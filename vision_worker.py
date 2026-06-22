# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python",
#     "onnxruntime>=1.17",
#     "pillow>=10",
#     "numpy>=1.26",
#     "huggingface-hub>=0.23",
# ]
#
# [tool.uv.sources]
# vgi-python = { path = "../vgi-python" }
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

_VISION_CATALOG = Catalog(
    name="vision",
    default_schema="main",
    schemas=[
        Schema(
            name="main",
            comment="Image classification (ImageNet) on image blobs for SQL",
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
