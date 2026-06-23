<p align="center">
  <img src="docs/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# vgi-vision

Image classification on image blobs, exposed as DuckDB SQL functions through
[VGI](https://github.com/Query-farm/vgi-python). Point it at a column of image
bytes and get ImageNet predictions back, in SQL:

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'vision' (TYPE vgi, LOCATION 'uv run vision_worker.py');

SELECT id, vision.top_label(image)                       FROM photos;
SELECT * FROM vision.classify((SELECT image FROM photos LIMIT 1));      -- top 5
SELECT * FROM vision.classify((SELECT image FROM photos LIMIT 1), 10);  -- top 10
SELECT * FROM vision.image_classes();                                   -- 1000 labels
```

Inference runs out-of-process on a permissively-licensed ONNX model via
[onnxruntime](https://onnxruntime.ai/). The model + labels are downloaded on
first use and cached; nothing large is committed to the repo.

---

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
| `image_classes()` | table | `(idx INT, label VARCHAR)` | The model's full 1000-class ImageNet label set. |

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

Written by [Query.Farm](https://query.farm) — every VGI worker is designed and built by Query.Farm.

Copyright 2026 Query Farm LLC - https://query.farm

