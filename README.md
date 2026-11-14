<p align="center">
  <img src="https://query.farm/media-kit/products/vgi/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# Image Classification (ImageNet) in DuckDB

> **vgi-vision** · a [Query.Farm](https://query.farm) VGI worker · powered by ONNX Runtime

A [VGI](https://query.farm) worker that brings **image classification** to
DuckDB/SQL. Point it at a column of image bytes (`BLOB`) — or a file path — and
get **ImageNet-1k** predictions back: a single top label per row, or the top-K
labels with confidence scores, as plain SQL scalar and table functions.
Inference runs out-of-process on **MobileNetV2** via
[ONNX Runtime](https://onnxruntime.ai/) (no PyTorch); the model and labels
download on first use and cache locally. It is built to be
**commercial-marketplace-safe** — deliberately **no** AGPL-licensed
Ultralytics/YOLO — so every dependency is permissively licensed.

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'vision' (TYPE vgi, LOCATION 'uvx --from git+https://github.com/Query-farm/vgi-vision@v0.4.0 vgi-vision-worker');

SELECT id, vision.top_label(image)                       FROM photos;
SELECT * FROM vision.classify('/photos/cat.jpg');                       -- top 5
SELECT * FROM vision.classify('/photos/cat.jpg', 10);                   -- top 10
SELECT * FROM vision.image_classes();                                   -- 1000 labels

-- Combine them: keep only the photos whose top label is a dog breed
-- (ImageNet class indexes 151-268), looked up in the class list.
SELECT p.id, c.label AS breed
FROM photos AS p
JOIN vision.image_classes AS c ON c.label = vision.top_label(p.image)
WHERE c.idx BETWEEN 151 AND 268
ORDER BY p.id;
```

`classify` takes its image as a literal (a `BLOB` or a file path); DuckDB does not
accept a `(SELECT ...)` subquery or a lateral column reference as a table-function
argument, so use the `top_label` scalar to label a whole column.

Inference runs out-of-process on a permissively-licensed ONNX model via
[onnxruntime](https://onnxruntime.ai/). The model + labels are downloaded on
first use and cached; nothing large is committed to the repo.

---

## Quick start

The worker isn't on PyPI; [uv](https://docs.astral.sh/uv/) runs it straight from a tagged release on GitHub. `uvx` builds an isolated environment on the first ATTACH and reuses it afterwards:

```sql
INSTALL vgi FROM community;
LOAD vgi;

ATTACH 'vision' AS vision (TYPE vgi, LOCATION 'uvx --from git+https://github.com/Query-farm/vgi-vision@v0.4.0 vgi-vision-worker');

-- Predict the single most likely ImageNet label for an image BLOB.
SELECT vision.main.top_label('\x89\x50\x4E\x47\x0D\x0A\x1A\x0A\x00\x00\x00\x0D\x49\x48\x44\x52\x00\x00\x00\x40\x00\x00\x00\x40\x08\x02\x00\x00\x00\x25\x0B\xE6\x89\x00\x00\x00\xE1\x49\x44\x41\x54\x78\xDA\xED\xDA\xB1\x0D\x83\x30\x10\x85\x61\xB0\x18\x21\x43\x30\x4C\x26\xA0\x66\x04\xC6\xC8\x08\x29\x52\x65\x82\x0C\xC3\x10\x19\x22\x05\x4D\xA4\x20\x41\x7C\xCF\x1C\x27\xFF\x57\x02\xB2\xDE\x87\x8D\x31\x16\x6D\x3F\x3C\x9A\xC8\x95\x9A\xE0\x05\x00\x00\x00\x00\x00\x00\x00\xF0\xAC\x6E\xCF\x45\xAF\xE1\xEE\x95\xEF\xFA\x1C\x19\x42\x00\x00\x00\x00\xE0\xFD\x1E\xC8\xA8\x79\x7A\xFF\x1E\xEC\x6F\x97\x00\x80\xD5\xE8\xDF\xA7\xB4\x8C\xEE\x98\xE8\xE5\x18\xE9\xE0\xF4\xD9\xD7\x97\x05\xE4\xA5\x91\x18\x92\x57\x7A\x95\xA1\xFA\xF7\x80\xFD\x16\x1A\x5B\xA8\xBB\x07\x54\x33\x89\xA5\x1D\xD6\x42\x00\x00\x00\x08\x0C\x50\xAD\x28\x2D\xED\x54\x3F\x84\xEC\x9D\x60\x6C\x81\x87\xD8\x76\x0B\xED\x1D\x98\x1C\x9F\x42\xC9\x1C\x20\x1B\x42\xFF\xA6\x51\xCD\x60\xCA\x8F\xFA\x25\xD3\xE6\xD2\xF2\xBC\xBB\x12\x9B\x8C\x18\xFB\x42\xE5\xB2\x32\x8D\x02\x00\x00\x00\x80\xBE\x5A\x7E\xFA\x03\x00\x00\x00\x00\x00\x00\x00\xE4\xD7\x07\x2B\x8B\x39\x99\xCE\x48\xA8\x86\x00\x00\x00\x00\x49\x45\x4E\x44\xAE\x42\x60\x82'::BLOB) AS label;

-- Top-5 (label, confidence) predictions for an image BLOB, confidence-descending.
SELECT label, round(confidence, 4) AS confidence FROM vision.main.classify('\x89\x50\x4E\x47\x0D\x0A\x1A\x0A\x00\x00\x00\x0D\x49\x48\x44\x52\x00\x00\x00\x40\x00\x00\x00\x40\x08\x02\x00\x00\x00\x25\x0B\xE6\x89\x00\x00\x00\xE1\x49\x44\x41\x54\x78\xDA\xED\xDA\xB1\x0D\x83\x30\x10\x85\x61\xB0\x18\x21\x43\x30\x4C\x26\xA0\x66\x04\xC6\xC8\x08\x29\x52\x65\x82\x0C\xC3\x10\x19\x22\x05\x4D\xA4\x20\x41\x7C\xCF\x1C\x27\xFF\x57\x02\xB2\xDE\x87\x8D\x31\x16\x6D\x3F\x3C\x9A\xC8\x95\x9A\xE0\x05\x00\x00\x00\x00\x00\x00\x00\xF0\xAC\x6E\xCF\x45\xAF\xE1\xEE\x95\xEF\xFA\x1C\x19\x42\x00\x00\x00\x00\xE0\xFD\x1E\xC8\xA8\x79\x7A\xFF\x1E\xEC\x6F\x97\x00\x80\xD5\xE8\xDF\xA7\xB4\x8C\xEE\x98\xE8\xE5\x18\xE9\xE0\xF4\xD9\xD7\x97\x05\xE4\xA5\x91\x18\x92\x57\x7A\x95\xA1\xFA\xF7\x80\xFD\x16\x1A\x5B\xA8\xBB\x07\x54\x33\x89\xA5\x1D\xD6\x42\x00\x00\x00\x08\x0C\x50\xAD\x28\x2D\xED\x54\x3F\x84\xEC\x9D\x60\x6C\x81\x87\xD8\x76\x0B\xED\x1D\x98\x1C\x9F\x42\xC9\x1C\x20\x1B\x42\xFF\xA6\x51\xCD\x60\xCA\x8F\xFA\x25\xD3\xE6\xD2\xF2\xBC\xBB\x12\x9B\x8C\x18\xFB\x42\xE5\xB2\x32\x8D\x02\x00\x00\x00\x80\xBE\x5A\x7E\xFA\x03\x00\x00\x00\x00\x00\x00\x00\xE4\xD7\x07\x2B\x8B\x39\x99\xCE\x48\xA8\x86\x00\x00\x00\x00\x49\x45\x4E\x44\xAE\x42\x60\x82'::BLOB) ORDER BY confidence DESC;
```

Change `@v0.4.0` to move to a newer release.

The MobileNetV2 model and labels download on first use and are cached locally; see [Models & licensing](#models--licensing).

### Other ways to run it

- **Docker** (the image bundles the worker and its dependencies):

  ```sql
  ATTACH 'vision' AS vision (TYPE vgi, LOCATION 'docker://ghcr.io/query-farm/vgi-vision:0.4.0');
  ```

- **From a local checkout** (`git clone https://github.com/Query-farm/vgi-vision`), point `LOCATION` at `uv run --project /path/to/vgi-vision vgi-vision-worker`.

### Build from source

To build and test the worker yourself from a checkout, see [Development](#development).

## Models & licensing

> **This worker is built specifically to be safe for a commercial marketplace.**
> It deliberately **does not** use Ultralytics / YOLOv8, whose code is
> **AGPL-3.0** — unacceptable for proprietary/commercial distribution (the same
> reason `vgi-pdf` rejected PyMuPDF). Everything below is permissively licensed.

| Component | What | License |
|-----------|------|---------|
| **Classifier weights** | **MobileNetV2 (opset 12)**, trained on ImageNet-1k (1000 classes). Sourced from the ONNX Model Zoo, mirrored on Hugging Face as [`onnxmodelzoo/mobilenetv2-12`](https://huggingface.co/onnxmodelzoo/mobilenetv2-12). | **Apache-2.0** |
| **Inference runtime** | [onnxruntime](https://github.com/microsoft/onnxruntime) | **MIT** |
| **Image decoding** | [Pillow](https://python-pillow.org/) | **MIT-CMU (HPND)** |
| **Numerics** | [NumPy](https://numpy.org/) | **BSD-3-Clause** |
| **Labels** | ImageNet-1k synset list ([`synset.txt`](https://github.com/onnx/models/blob/main/validated/vision/classification/synset.txt)) | Apache-2.0 (ONNX Model Zoo) |
| **This worker** | vgi-vision | **MIT** (see `LICENSE`) |

The exact model file is `mobilenetv2-12.onnx` (~13.3 MB). It is **downloaded on
first use** to `~/.cache/vgi-vision/` (override with `VGI_VISION_CACHE_DIR`, or
point `VGI_VISION_MODEL` / `VGI_VISION_LABELS` at local copies). Pre-fetch it
with `make fetch-model`. The weights are **never committed** (gitignored).

### Object detection — future work

Object detection (`detect_objects`, `count_objects`, `has_object`) is **not
shipped in v1**. The popular detection models (YOLOv5/v8) carry AGPL-3.0 or
otherwise non-permissive terms, and the clearly-permissive alternatives
(e.g. an SSD-MobileNet exported from the Apache-2.0 TensorFlow Object Detection
API) lack a single stable, well-labelled ONNX artifact we were comfortable
bundling for a commercial product. Rather than ship a model with murky
provenance, v1 ships **classification only**. Detection can be added later behind
the same `classify`-style surface once a vetted Apache/BSD/MIT detection model +
labels are pinned.

---

## Function surface

### Classification

| Function | Kind | Returns | Notes |
|----------|------|---------|-------|
| `classify(image[, top_k])` | table | `(label VARCHAR, confidence DOUBLE)` | Top-k predictions (default `top_k=5`), ordered by `confidence` desc. `image` is a BLOB; a `classify(path[, top_k])` VARCHAR overload reads a file off disk. |
| `top_label(image)` | scalar | `VARCHAR` | The #1 predicted label. BLOB input; `top_label(path)` VARCHAR overload reads a file. |
| `image_classes()` | table | `(idx INT, label VARCHAR)` | The model's full 1000-class ImageNet label set. Only 998 labels are distinct: `crane` and `maillot` each appear at two indexes. |

`confidence` values are softmax probabilities in `[0, 1]`.

### NULL / robustness semantics

Images are **untrusted**. The worker never crashes on bad input:

- `NULL` image → `NULL` (`top_label`) / **no rows** (`classify`).
- Malformed / non-image / empty bytes → `NULL` / no rows.
- Over-large images (a decompression-bomb guard caps decode at 64 MP) → `NULL` / no rows.
- A genuine model-load failure (offline, disk) raises a clear, actionable error
  telling you how to fetch the model — it is *not* silently swallowed per row.

### Performance / warm-up

The ONNX session + labels load **once per worker process** and are cached for its
lifetime, so the cost is amortised across every row of every query. The worker
also **warms the model at startup** (`vision_worker.py`'s `run()` calls
`model.warm_up()`), so the first query of an `ATTACH` is fast and the end-to-end
SQL suite stays deterministic under load.

---

## Development

```sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev
make fetch-model            # download + cache the model and labels (one-off)
uv run --no-sync pytest -q  # unit + integration tests
make test-sql               # DuckDB end-to-end (.test) over a tiny generated PNG
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_vision/
```

Model-dependent tests skip cleanly when the model isn't present (e.g. offline),
so a bare checkout still goes green; they run for real once `make fetch-model`
(or any first query) has populated the cache.

## Layout

```
vision_worker.py        # worker entry point (PEP 723 header) + startup warm-up
vgi_vision/
  model.py              # pure inference: download/cache, warm-up, preprocess, classify
  scalars.py            # top_label(image) / top_label(path)
  tables.py             # classify(...), image_classes()
  schema_utils.py       # field() column-comment helper
tests/                  # pytest: pure logic + in-process scalar/table + Client E2E
test/sql/               # DuckDB sqllogictest .test files
```

---

## Authorship & License

Written by [Query.Farm](https://query.farm).

Copyright 2026 Query Farm LLC - https://query.farm

