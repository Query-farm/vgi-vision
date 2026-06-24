# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.4",
#     "onnxruntime>=1.17",
#     "pillow>=10",
#     "numpy>=1.26",
#     "huggingface-hub>=0.23",
# ]
# ///
"""VGI worker exposing image classification (ImageNet) to DuckDB/SQL.

Assembles the scalar + table functions in ``vgi_vision`` into a single ``vision``
catalog and runs the worker over stdio (DuckDB subprocess). Inference runs on a
permissively-licensed ONNX classifier (MobileNetV2, Apache-2.0 weights) via
onnxruntime (MIT) -- safe for a commercial marketplace, unlike AGPL YOLOv8.

Usage:
    uv run vision_worker.py               # serve over stdio (DuckDB subprocess)

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'vision' (TYPE vgi, LOCATION 'uv run vision_worker.py');

    SELECT vision.top_label(image)                 FROM photos;
    SELECT * FROM vision.classify((SELECT image FROM photos LIMIT 1));
    SELECT * FROM vision.classify((SELECT image FROM photos LIMIT 1), 10);
    SELECT * FROM vision.image_classes();          -- 1000 ImageNet labels

The model + labels are downloaded on first use and cached under
~/.cache/vgi-vision (override with VGI_VISION_CACHE_DIR). Pre-fetch them with
`make fetch-model`.
"""

from __future__ import annotations

from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_vision import model
from vgi_vision.scalars import SCALAR_FUNCTIONS
from vgi_vision.tables import TABLE_FUNCTIONS

_CATALOG_DESCRIPTION_LLM = (
    "Run image classification on image blobs (or image file paths) directly in SQL. "
    "Given the raw bytes of a PNG/JPEG/etc. image, returns ImageNet-1k labels: the single "
    "most likely label (`top_label`), the top-k (label, confidence) predictions "
    "(`classify`), or the model's full 1000-class label set (`image_classes`). The "
    "classifier is a permissively-licensed MobileNetV2 ONNX model. Use it to tag, filter, "
    "or group images by what they depict — e.g. find photos of cats, label a column of "
    "image blobs, or rank an image's most probable subjects."
)

_CATALOG_DESCRIPTION_MD = (
    "# vision\n\n"
    "Image classification (ImageNet-1k) on image blobs as DuckDB SQL functions, via VGI. "
    "Inference runs out-of-process on a permissively-licensed MobileNetV2 ONNX model "
    "(Apache-2.0 weights) through onnxruntime.\n\n"
    "- **`top_label(image)` / `top_label(path)`** — scalar: the #1 predicted label per row.\n"
    "- **`classify(image[, top_k])` / `classify(path[, top_k])`** — table: top-k "
    "`(label, confidence)` predictions, confidence descending.\n"
    "- **`image_classes()`** — table: the model's full `(idx, label)` label set (1000 rows).\n\n"
    "Untrusted/malformed images yield SQL NULL (or no rows), never a crash."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Image-classification functions over image blobs and file paths: predict the top "
    "ImageNet label (`top_label`), the top-k (label, confidence) predictions (`classify`), "
    "and enumerate the model's class set (`image_classes`)."
)

_SCHEMA_DESCRIPTION_MD = (
    "Image-classification functions: `top_label` (scalar), `classify` (table), and "
    "`image_classes` (table) over a MobileNetV2 ImageNet-1k ONNX model."
)

_CATALOG_TAGS = {
    "vgi.description_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.description_md": _CATALOG_DESCRIPTION_MD,
    "vgi.author": "Query.Farm",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": "https://github.com/Query-farm/vgi-vision/issues",
    "vgi.support_policy_url": "https://github.com/Query-farm/vgi-vision/blob/main/README.md",
}

_VISION_CATALOG = Catalog(
    name="vision",
    default_schema="main",
    comment="Image classification (ImageNet) on image blobs and file paths for SQL.",
    source_url="https://github.com/Query-farm/vgi-vision",
    tags=_CATALOG_TAGS,
    schemas=[
        Schema(
            name="main",
            comment="Image classification (ImageNet) on image blobs for SQL",
            tags={
                "vgi.description_llm": _SCHEMA_DESCRIPTION_LLM,
                "vgi.description_md": _SCHEMA_DESCRIPTION_MD,
            },
            functions=[*SCALAR_FUNCTIONS, *TABLE_FUNCTIONS],
        ),
    ],
)


class VisionWorker(Worker):
    """Worker process hosting the ``vision`` catalog."""

    catalog = _VISION_CATALOG

    def run(self, otel_config: Any = None) -> None:
        """Warm the ONNX model + labels once, then serve.

        Loading the model is lazy, so without this the first query of every ATTACH
        pays the download + ORT session-init cost inline -- a window in which a
        worker-pool teardown SIGTERM (or a heavily-loaded host) can kill the run
        mid-assertion and record a spurious E2E failure. Warming at spawn moves
        that one-time cost ahead of any query, keeping the SQL suite deterministic
        without changing a single output value. Best-effort; never fatal.
        """
        model.warm_up()
        super().run(otel_config=otel_config)


def main() -> None:
    """Run the vision worker process (stdio or, via flags, HTTP)."""
    VisionWorker.main()


if __name__ == "__main__":
    main()
