# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.16.0",
#     "onnxruntime>=1.17",
#     "pillow>=10",
#     "numpy>=1.26",
#     "huggingface-hub>=0.23",
# ]
# ///
"""Repo-root PEP 723 entry-point shim for the vision worker.

The catalog + :class:`VisionWorker` + :func:`main` now live in the wheel-importable
:mod:`vgi_vision.worker` (so the built distribution and the ``vgi-vision-worker``
console script carry the worker). This file stays a thin shim that re-exports them,
keeping ``uv run vision_worker.py`` (Makefile / ci/run-integration.sh / tests) working
unchanged as the stdio/`--http`/`--unix` entry point DuckDB drives.

Usage:
    uv run vision_worker.py               # serve over stdio (DuckDB subprocess)

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'vision' (TYPE vgi, LOCATION 'uv run vision_worker.py');
"""

from __future__ import annotations

from vgi_vision.worker import VisionWorker, main

__all__ = ["VisionWorker", "main"]


if __name__ == "__main__":
    main()
