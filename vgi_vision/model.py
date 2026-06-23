"""Model lifecycle + pure inference: load the ONNX classifier once, classify bytes.

VGI keeps the worker process alive across queries, so the expensive thing an
inference worker does -- loading the ONNX model + labels -- happens **once** and is
amortised over every row of every query. This module centralises that caching:
scalar/table functions only ask "classify these bytes" and get predictions back.

Model & licensing
-----------------
* Classifier: **MobileNetV2 (opset 12)** from the ONNX Model Zoo, ImageNet-1k
  (1000 classes). Weights license: **Apache-2.0**. Inference runtime:
  **onnxruntime** (MIT). Both are permissively licensed and safe for a
  commercial marketplace -- unlike Ultralytics/YOLOv8 (AGPL-3.0).
* The model + labels are **downloaded on first use and cached** under
  ``~/.cache/vgi-vision`` (overridable via ``VGI_VISION_CACHE_DIR``); they are
  never committed to the repo. ``make fetch-model`` / the first query / worker
  warm-up all trigger the download.

Robustness / security
---------------------
Images are **untrusted**. Every decode is bounded (a hard pixel cap defeats
decompression bombs) and wrapped so a malformed/huge/empty blob yields ``None``
(-> SQL NULL / no rows) rather than crashing the worker.

Everything here is lazy: importing the module is cheap; nothing downloads or
loads until the first classification (or an explicit :func:`warm_up`).
"""

from __future__ import annotations

import io
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# ---------------------------------------------------------------------------
# Model identity & sources (all permissively licensed -- see module docstring).
# ---------------------------------------------------------------------------

MODEL_NAME = "mobilenetv2-12"
MODEL_DISPLAY_NAME = "MobileNetV2 (opset 12, ImageNet-1k)"
MODEL_LICENSE = "Apache-2.0"
RUNTIME_LICENSE = "MIT (onnxruntime)"
NUM_CLASSES = 1000

# HuggingFace mirror of the (deprecated) ONNX Model Zoo. Apache-2.0 weights.
_MODEL_HF_REPO = "onnxmodelzoo/mobilenetv2-12"
_MODEL_FILENAME = "mobilenetv2-12.onnx"
# Direct fallback URL (HF resolve) used when huggingface_hub is unavailable.
_MODEL_URL = f"https://huggingface.co/{_MODEL_HF_REPO}/resolve/main/{_MODEL_FILENAME}"

# ImageNet-1k synset labels (1000 lines: "n01440764 tench, Tinca tinca").
_LABELS_FILENAME = "synset.txt"
_LABELS_URL = "https://raw.githubusercontent.com/onnx/models/main/validated/vision/classification/synset.txt"

# ONNX I/O: input tensor "input" (N,3,224,224) float; output "output" (N,1000) logits.
_INPUT_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Hard decode cap (decompression-bomb / OOM guard). 64 MP comfortably covers real
# photos while bounding the work a single malicious blob can force.
_MAX_PIXELS = 64 * 1024 * 1024

_lock = threading.Lock()


class ModelNotAvailableError(RuntimeError):
    """The model/labels could not be downloaded or loaded.

    Carries an actionable hint so the DuckDB-side error tells the user how to fix
    it (e.g. run ``make fetch-model`` or check network access).
    """


# ---------------------------------------------------------------------------
# Cache directory + downloads
# ---------------------------------------------------------------------------


def cache_dir() -> Path:
    """Directory where the model + labels are cached (created on demand)."""
    d = Path(os.environ.get("VGI_VISION_CACHE_DIR", Path.home() / ".cache" / "vgi-vision"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` atomically (via a temp file + rename)."""
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (https only)
            data = resp.read()
        tmp.write_bytes(data)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def model_path() -> Path:
    """Path to the cached ONNX model, downloading it on first use."""
    explicit = os.environ.get("VGI_VISION_MODEL")
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise ModelNotAvailableError(f"VGI_VISION_MODEL={explicit!r} does not exist.")
        return p
    dest = cache_dir() / _MODEL_FILENAME
    if dest.exists():
        return dest
    with _lock:
        if dest.exists():  # re-check after acquiring the lock
            return dest
        # Prefer huggingface_hub (resumable, cached); fall back to a plain URL fetch.
        try:
            from huggingface_hub import hf_hub_download

            got = hf_hub_download(repo_id=_MODEL_HF_REPO, filename=_MODEL_FILENAME)
            # hf_hub_download returns a path in its own cache; copy into ours so the
            # rest of the worker has a single, stable cache location.
            dest.write_bytes(Path(got).read_bytes())
            return dest
        except Exception:  # noqa: BLE001 -- fall back to direct download
            pass
        try:
            _download(_MODEL_URL, dest)
        except Exception as exc:  # noqa: BLE001
            raise ModelNotAvailableError(
                f"Could not download the {MODEL_DISPLAY_NAME} model. "
                f"Run `make fetch-model`, or fetch {_MODEL_URL} to "
                f"{dest}, or set VGI_VISION_MODEL to a local .onnx path. "
                f"(original error: {exc})"
            ) from exc
        return dest


def labels_path() -> Path:
    """Path to the cached ImageNet synset labels, downloading on first use."""
    explicit = os.environ.get("VGI_VISION_LABELS")
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise ModelNotAvailableError(f"VGI_VISION_LABELS={explicit!r} does not exist.")
        return p
    dest = cache_dir() / _LABELS_FILENAME
    if dest.exists():
        return dest
    with _lock:
        if dest.exists():
            return dest
        try:
            _download(_LABELS_URL, dest)
        except Exception as exc:  # noqa: BLE001
            raise ModelNotAvailableError(
                f"Could not download ImageNet labels from {_LABELS_URL}. (original error: {exc})"
            ) from exc
        return dest


# ---------------------------------------------------------------------------
# Cached model + labels
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _session() -> ort.InferenceSession:
    """The ONNX Runtime session, created once per process."""
    path = model_path()
    opts = ort.SessionOptions()
    # Single-threaded intra-op keeps the worker cooperative under DuckDB's own
    # parallelism and makes results bit-stable across runs.
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    try:
        return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        raise ModelNotAvailableError(f"Failed to load ONNX model {path}: {exc}") from exc


@lru_cache(maxsize=1)
def labels() -> list[str]:
    """The model's 1000 ImageNet class labels, indexed by class id.

    Each synset line is ``"<synset_id> <comma, separated, names>"``; we keep the
    first human-readable name (before the first comma) for a clean SQL label.
    """
    raw = labels_path().read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        # Drop the leading "nXXXXXXXX " synset id, keep the first common name.
        parts = line.split(" ", 1)
        names = parts[1] if len(parts) == 2 else parts[0]
        out.append(names.split(",")[0].strip())
    return out


@lru_cache(maxsize=1)
def _input_name() -> str:
    return _session().get_inputs()[0].name


# ---------------------------------------------------------------------------
# Preprocessing + inference (pure, over raw bytes)
# ---------------------------------------------------------------------------


def _preprocess(data: bytes) -> np.ndarray | None:
    """Decode + normalize image bytes to a (1,3,224,224) float32 tensor.

    Returns ``None`` for empty/malformed/over-large images (the caller maps that
    to SQL NULL / no rows). Never raises on bad input.
    """
    if not data:
        return None
    try:
        with Image.open(io.BytesIO(data)) as src:
            # Bound the decode: reject decompression bombs before paying for them.
            w, h = src.size
            if w <= 0 or h <= 0 or (w * h) > _MAX_PIXELS:
                return None
            rgb = src.convert("RGB").resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
    except Exception:  # noqa: BLE001 -- any decode failure -> NULL, never crash
        return None
    if arr.shape != (_INPUT_SIZE, _INPUT_SIZE, 3):
        return None
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    # HWC -> CHW -> NCHW
    chw = np.transpose(arr, (2, 0, 1))
    return chw[np.newaxis, :, :, :].astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis."""
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def classify_image(data: bytes | None, top_k: int = 5) -> list[tuple[str, float]] | None:
    """Classify image ``data``; return up to ``top_k`` ``(label, confidence)``.

    * ``None``/empty/malformed/over-large image -> ``None`` (SQL NULL / no rows).
    * Confidences are softmax probabilities in ``[0, 1]``, sorted descending.
    * ``top_k`` is clamped to ``[1, NUM_CLASSES]``.

    Never raises on bad image bytes; a genuine model-load failure (network/disk)
    does raise :class:`ModelNotAvailableError` so the user gets an actionable error.
    """
    if data is None:
        return None
    tensor = _preprocess(data)
    if tensor is None:
        return None
    k = max(1, min(int(top_k), NUM_CLASSES))
    sess = _session()
    try:
        logits = sess.run(None, {_input_name(): tensor})[0][0]
    except Exception:  # noqa: BLE001 -- defensive: never crash on a single row
        return None
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    names = labels()
    # argpartition for the top-k, then sort just those k by confidence desc.
    top_idx = np.argpartition(probs, -k)[-k:]
    top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]
    return [(names[int(i)], float(probs[int(i)])) for i in top_idx]


def top_label_for(data: bytes | None) -> str | None:
    """The single #1 predicted label for ``data`` (or ``None``)."""
    preds = classify_image(data, top_k=1)
    if not preds:
        return None
    return preds[0][0]


def class_table() -> list[tuple[int, str]]:
    """The model's label set as ``(idx, label)`` rows."""
    return list(enumerate(labels()))


# ---------------------------------------------------------------------------
# Startup warm-up
# ---------------------------------------------------------------------------


def warm_up() -> None:
    """Load the model + labels once, eagerly, at worker startup.

    Everything here is lazy by design, so the *first* query of every ATTACH
    otherwise pays the model-load cost (download + ORT session init) inline. Under
    the end-to-end SQL suite that load happens mid-assertion on the first file -- a
    window in which a worker-pool teardown SIGTERM or a heavily-loaded host can kill
    the run and record a spurious failure. Warming here moves the one-time cost to
    process spawn, so each per-file first query is fast and that window shrinks to
    near zero. It only populates caches; it never changes any output.

    Best-effort: if the model can't be fetched (e.g. offline), warm-up is silent --
    the relevant function raises its own actionable error if actually invoked.
    """
    try:
        _session()
        labels()
        # A tiny real inference primes ORT's lazy graph optimisation too.
        dummy = np.zeros((1, 3, _INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)
        _session().run(None, {_input_name(): dummy})
    except Exception:  # noqa: BLE001 -- never block startup
        pass
