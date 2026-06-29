# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.5",
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

import json
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, Schema, View

from vgi_vision import model
from vgi_vision._examples import SAMPLE_IMAGE_BLOB
from vgi_vision.meta import keywords_json
from vgi_vision.scalars import SCALAR_FUNCTIONS
from vgi_vision.tables import TABLE_FUNCTIONS

_CATALOG_TITLE = "Image Classification (ImageNet)"

_CATALOG_KEYWORDS = [
    "image classification",
    "vision",
    "ImageNet",
    "label",
    "classify",
    "computer vision",
    "onnx",
    "mobilenet",
    "image tagging",
    "predict",
    "top_label",
    "image_classes",
    "photo",
    "object recognition",
    "machine learning",
    "inference",
]

_CATALOG_DOC_LLM = (
    "Run image classification on image blobs (or image file paths) directly in SQL. "
    "Given the raw bytes of a PNG/JPEG/etc. image, returns ImageNet-1k labels: the single "
    "most likely label (`top_label`), the top-k (label, confidence) predictions "
    "(`classify`), or the model's full 1000-class label set (`image_classes`). The "
    "classifier is a permissively-licensed MobileNetV2 ONNX model. Use it to tag, filter, "
    "or group images by what they depict — e.g. find photos of cats, label a column of "
    "image blobs, or rank an image's most probable subjects. Inputs are untrusted: a "
    "malformed/empty/over-large image yields SQL NULL (scalar) or no rows (table), never a "
    "worker crash. Inference is out-of-process; the model + labels download once and cache "
    "under `~/.cache/vgi-vision`."
)

_CATALOG_DOC_MD = (
    "# Vision: Image Classification (ImageNet) in SQL\n\n"
    "![ONNX Runtime]"
    "(https://raw.githubusercontent.com/microsoft/onnxruntime/main/docs/images/ONNX_Runtime_logo_dark.png)\n\n"
    "**Run deep-learning image classification directly in DuckDB SQL** — point a query at a "
    "column of image blobs (or image file paths) and get back ImageNet-1k labels, no Python "
    "notebook, model server, or ML pipeline required. The `vision` catalog turns computer-vision "
    "inference into ordinary SQL functions you can `SELECT`, filter, `JOIN`, and aggregate.\n\n"
    "This extension is for data engineers, analysts, and application developers who want to tag, "
    "filter, search, or group images by what they depict without leaving the database. Classic "
    "use cases include auto-labeling a column of photo blobs, finding every row whose image looks "
    "like a cat or a car, ranking an image's most probable subjects, and enriching a media table "
    "with predicted categories — all expressed as the SQL you already write. Because the model "
    "runs out-of-process behind VGI, untrusted inputs are safe: a malformed, empty, or oversized "
    "image yields SQL `NULL` (scalar) or no rows (table function), never a worker crash.\n\n"
    "Under the hood, inference is powered by [ONNX Runtime](https://onnxruntime.ai), Microsoft's "
    "high-performance, cross-platform inference engine for ONNX models "
    "([source on GitHub](https://github.com/microsoft/onnxruntime), "
    "[documentation](https://onnxruntime.ai/docs/)). The classifier is a permissively-licensed "
    "MobileNetV2 model trained on the [ImageNet](https://www.image-net.org) 1000-class dataset "
    "(Apache-2.0 weights served via ONNX Runtime's MIT license) — deliberately chosen to be safe "
    "for a commercial marketplace, unlike copyleft alternatives. The model and its 1000 labels "
    "are downloaded once on first use and cached locally, so subsequent queries pay only "
    "inference cost.\n\n"
    "## Functions\n\n"
    "- **`top_label(image)` / `top_label(path)`** — scalar: the single #1 predicted ImageNet "
    "label per row. Ideal for a `SELECT vision.top_label(photo) FROM media` enrichment column.\n"
    "- **`classify(image[, top_k])` / `classify(path[, top_k])`** — table function returning the "
    "top-k `(label, confidence)` predictions, confidence descending (default top-5).\n"
    "- **`image_classes()`** — table function listing the model's full `(idx, label)` label set "
    "(all 1000 ImageNet classes); also exposed as the parenthesis-free view "
    "`vision.main.image_classes` for discovery and joins.\n\n"
    "## Notes\n\n"
    "Every image argument is accepted either as a `BLOB` column or via a filesystem-path "
    "overload. Untrusted or malformed images yield SQL `NULL` (scalar) or no rows (table), never "
    "a crash, making the functions safe to apply across an entire column of user-supplied data."
)

_SCHEMA_TITLE = "Vision — main schema"

_SCHEMA_KEYWORDS = [
    "image classification",
    "top_label",
    "classify",
    "image_classes",
    "ImageNet",
    "vision",
    "label",
    "confidence",
    "onnx",
    "mobilenet",
    "blob",
    "file path",
]

_SCHEMA_DOC_LLM = (
    "Image-classification functions over image blobs and file paths. `top_label` (scalar) "
    "returns the single most likely ImageNet label per row; `classify` (table) returns the "
    "top-k `(label, confidence)` predictions confidence-descending; `image_classes` (table) "
    "enumerates the model's full 1000-class label set. Reach for this schema to tag, filter, "
    "or rank images by what they depict, or to discover the available class labels."
)

_SCHEMA_DOC_MD = (
    "## Vision functions\n\n"
    "Image classification over a MobileNetV2 ImageNet-1k ONNX model.\n\n"
    "| function | kind | returns |\n"
    "|---|---|---|\n"
    "| `top_label(image\\|path)` | scalar | the #1 predicted label |\n"
    "| `classify(image\\|path[, top_k])` | table | top-k `(label, confidence)` |\n"
    "| `image_classes()` | table | the full `(idx, label)` label set |\n\n"
    "Images may be BLOB columns or filesystem paths; bad images yield NULL / no rows."
)

# Representative catalog-qualified SQL for the schema (VGI506). The classify
# example uses a BLOB literal so it is self-contained (no `photos` table, no
# subquery argument, which DuckDB rejects for table functions).
_SCHEMA_EXAMPLE_QUERIES = (
    f"SELECT vision.main.top_label('{SAMPLE_IMAGE_BLOB}'::BLOB) AS label;\n"
    f"SELECT * FROM vision.main.classify('{SAMPLE_IMAGE_BLOB}'::BLOB);\n"
    f"SELECT * FROM vision.main.classify('{SAMPLE_IMAGE_BLOB}'::BLOB, 10);\n"
    "SELECT count(*) AS n_classes FROM vision.main.image_classes();\n"
    "SELECT idx, label FROM vision.main.image_classes() WHERE idx < 5 ORDER BY idx;"
)

# A parameterless table function always returns the same rows, so it is also
# exposed as a regular view of the same name (VGI311): `SELECT * FROM
# vision.main.image_classes` (no parentheses) scans the `image_classes()` table
# function. The view and the table function share the name in different DuckDB
# namespaces (relation vs. function), so both call styles work.
_CLASSES_VIEW_TITLE = "ImageNet Class List (view)"

_CLASSES_VIEW_DOC_LLM = (
    "The classifier's entire ImageNet-1k label set as a queryable view of `(idx, label)` rows — "
    "all 1000 classes the model can predict — so it reads like a normal table: "
    "`SELECT * FROM vision.main.image_classes` (no parentheses). It is a thin wrapper over the "
    "`image_classes()` table function. Use it to discover or validate which labels "
    "`top_label`/`classify` can return, or to join predicted labels against the canonical class "
    "list. `idx` is the 0-based position into the model's output vector; `label` is the "
    "human-readable class name. Always returns exactly 1000 rows."
)

_CLASSES_VIEW_DOC_MD = (
    "## `image_classes` (view)\n\n"
    "A view over the `image_classes()` table function listing **every ImageNet-1k class** the "
    "model can predict (1000 rows). Query it without parentheses: "
    "`SELECT * FROM vision.main.image_classes`.\n\n"
    "### Columns\n\n"
    "| column | type | description |\n"
    "|---|---|---|\n"
    "| `idx` | INTEGER | 0-based class index into the model output |\n"
    "| `label` | VARCHAR | ImageNet class label |\n\n"
    "### Notes\n\n"
    "Useful to validate or join against predicted labels from `top_label` / `classify`."
)

_CLASSES_VIEW_KEYWORDS = [
    "image classes",
    "labels",
    "ImageNet",
    "class list",
    "vocabulary",
    "categories",
    "discovery",
    "enumerate",
    "idx",
    "label",
    "1000 classes",
    "vision",
    "view",
]

_CLASSES_VIEW_EXAMPLES = json.dumps(
    [
        {
            "description": "Count the ImageNet classes the model can predict (1000).",
            "sql": "SELECT count(*) AS n FROM vision.main.image_classes",
        },
        {
            "description": "Peek at the first five (idx, label) class rows.",
            "sql": "SELECT idx, label FROM vision.main.image_classes WHERE idx < 5 ORDER BY idx",
        },
    ]
)

# Catalog-level guaranteed-runnable examples (VGI509). Self-contained BLOB literal +
# no-arg discovery; expected_result omitted on purpose (the linter only needs them
# to execute, and pinning exact predictions/labels would be brittle). Built with
# ``json.dumps`` so the BLOB literal's backslash-x escapes are valid JSON (VGI507).
_CATALOG_EXECUTABLE_EXAMPLES = json.dumps(
    [
        {
            "description": "Predict the single most likely ImageNet label for an image BLOB.",
            "sql": f"SELECT vision.main.top_label('{SAMPLE_IMAGE_BLOB}'::BLOB) AS label",
        },
        {
            "description": "Top-5 (label, confidence) predictions for an image BLOB.",
            "sql": f"SELECT * FROM vision.main.classify('{SAMPLE_IMAGE_BLOB}'::BLOB)",
        },
        {
            "description": "Count the classes the model can predict (1000 ImageNet labels).",
            "sql": "SELECT count(*) AS n FROM vision.main.image_classes()",
        },
    ]
)

_CATALOG_TAGS = {
    "vgi.title": _CATALOG_TITLE,
    "vgi.keywords": keywords_json(_CATALOG_KEYWORDS),
    "vgi.doc_llm": _CATALOG_DOC_LLM,
    "vgi.doc_md": _CATALOG_DOC_MD,
    "vgi.executable_examples": _CATALOG_EXECUTABLE_EXAMPLES,
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
                "vgi.title": _SCHEMA_TITLE,
                "vgi.keywords": keywords_json(_SCHEMA_KEYWORDS),
                "vgi.doc_llm": _SCHEMA_DOC_LLM,
                "vgi.doc_md": _SCHEMA_DOC_MD,
                "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "computer-vision",
                "category": "image-classification",
                "topic": "imagenet-labeling",
            },
            functions=[*SCALAR_FUNCTIONS, *TABLE_FUNCTIONS],
            views=[
                View(
                    name="image_classes",
                    definition="SELECT idx, label FROM vision.main.image_classes()",
                    comment="The model's ImageNet label set: (idx, label), 1000 rows",
                    column_comments={
                        "idx": "0-based class index into the model's output vector.",
                        "label": "Human-readable ImageNet class label.",
                    },
                    tags={
                        "vgi.title": _CLASSES_VIEW_TITLE,
                        "vgi.keywords": keywords_json(_CLASSES_VIEW_KEYWORDS),
                        "vgi.doc_llm": _CLASSES_VIEW_DOC_LLM,
                        "vgi.doc_md": _CLASSES_VIEW_DOC_MD,
                        "vgi.executable_examples": _CLASSES_VIEW_EXAMPLES,
                        "vgi.example_queries": _CLASSES_VIEW_EXAMPLES,
                        # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                        "domain": "computer-vision",
                        "category": "image-classification",
                        "topic": "imagenet-labeling",
                    },
                ),
            ],
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
