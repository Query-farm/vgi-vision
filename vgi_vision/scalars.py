"""Per-row scalar vision functions.

``top_label`` is a true DuckDB **scalar** -- one image (per row) in, one label out
-- so it can be used inline in any projection or predicate:

    SELECT id, top_label(image) FROM photos;
    SELECT * FROM photos WHERE top_label(image) = 'tabby';

Argument syntax: VGI *scalar* functions take **positional** arguments and resolve
overloads by *arity* (``name := value`` is a table-function / macro feature). The
image is accepted as a BLOB; a VARCHAR-path overload (``top_label(path)``) reads
the file off disk for convenience.

NULL / robustness semantics: a NULL input row yields NULL output. Images are
**untrusted** -- a malformed/empty/over-large blob yields NULL rather than
crashing the worker (the per-row guard lives in :mod:`vgi_vision.model`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pyarrow as pa
from vgi.arguments import Param, Returns
from vgi.metadata import FunctionExample
from vgi.scalar_function import ScalarFunction

from . import model

# Cap on bytes read from a path overload -- a path is still untrusted input.
_MAX_FILE_BYTES = 64 * 1024 * 1024


def _read_path(path: str | None) -> bytes | None:
    """Read image bytes from a filesystem path, or None on any failure."""
    if not path:
        return None
    try:
        p = Path(path)
        if not p.is_file() or p.stat().st_size > _MAX_FILE_BYTES:
            return None
        return p.read_bytes()
    except Exception:  # noqa: BLE001 -- missing/unreadable file -> NULL
        return None


class TopLabelFunction(ScalarFunction):
    """``top_label(image)`` -- the #1 predicted ImageNet label for a BLOB image."""

    class Meta:
        name = "top_label"
        description = "The #1 predicted ImageNet label for an image BLOB (NULL if not an image)"
        categories = ["vision", "classification"]
        examples = [
            FunctionExample(
                sql="SELECT vision.top_label(image) FROM photos",
                description="Top predicted label for each image",
            ),
        ]

    @classmethod
    def compute(
        cls, image: Annotated[pa.BinaryArray, Param(doc="Image bytes (PNG/JPEG/...).")]
    ) -> Annotated[pa.StringArray, Returns()]:
        out = [None if b is None else model.top_label_for(b) for b in image.to_pylist()]
        return pa.array(out, type=pa.string())


class TopLabelPathFunction(ScalarFunction):
    """``top_label(path)`` -- like ``top_label(image)`` but reads a file path."""

    class Meta:
        name = "top_label"
        description = "The #1 predicted ImageNet label for an image file path (NULL if unreadable)"
        categories = ["vision", "classification"]
        examples = [
            FunctionExample(
                sql="SELECT vision.top_label('/tmp/cat.jpg')",
                description="Top predicted label for an image on disk",
            ),
        ]

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc="Filesystem path to an image file.")]
    ) -> Annotated[pa.StringArray, Returns()]:
        out = [None if p is None else model.top_label_for(_read_path(p)) for p in path.to_pylist()]
        return pa.array(out, type=pa.string())


SCALAR_FUNCTIONS: list[type] = [
    TopLabelFunction,
    TopLabelPathFunction,
]
