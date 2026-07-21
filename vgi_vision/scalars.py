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

import json
from pathlib import Path
from typing import Annotated

import pyarrow as pa
from vgi.arguments import Param, Returns
from vgi.scalar_function import ScalarFunction

from . import model
from ._examples import SAMPLE_IMAGE_BLOB
from .meta import object_tags

# Cap on bytes read from a path overload -- a path is still untrusted input.
_MAX_FILE_BYTES = 64 * 1024 * 1024

# The committed image fixture, as a path, for the path-overload examples. The
# linter launches the worker with the repo root as its working directory, so this
# relative path resolves to the same deterministic PNG the BLOB literal embeds.
_SAMPLE_IMAGE_PATH = "test/sql/data/sample.png"

_TOP_LABEL_DOC_LLM = (
    "Return the single most likely ImageNet-1k class label for an image, as a scalar — "
    "one image per row in, one `VARCHAR` label out — so it composes anywhere in a "
    "`SELECT` projection or a `WHERE`/`GROUP BY` predicate. Use it to tag a column of "
    "images, filter rows to a subject (`WHERE top_label(img) = 'tabby'`), or bucket "
    "images by what they depict.\n\n"
    "Two arity-1 overloads share the name and resolve by input type: a `BLOB` overload "
    "(`top_label(image)`) over raw image bytes, and a `VARCHAR` overload "
    "(`top_label(path)`) that reads the image off a filesystem path. Inputs are untrusted: "
    "a NULL, malformed, empty, over-large, or unreadable image yields SQL `NULL` rather "
    "than raising. For the top-k predictions with confidences, use `classify` instead."
)

_TOP_LABEL_DOC_MD = (
    "## `top_label`\n\n"
    "Scalar function returning the **#1 predicted ImageNet label** for an image.\n\n"
    "### Overloads\n\n"
    "- `top_label(image BLOB) -> VARCHAR` — classify raw image bytes.\n"
    "- `top_label(path VARCHAR) -> VARCHAR` — read the image at a filesystem path, then classify.\n\n"
    "### Returns\n\n"
    "A single ImageNet class label (e.g. `tabby`, `envelope`), or `NULL` for a "
    "NULL/malformed/unreadable image.\n\n"
    "### Notes\n\n"
    "Use in projections or predicates. For ranked `(label, confidence)` predictions, "
    "use the `classify` table function."
)

_TOP_LABEL_KEYWORDS = [
    "top label",
    "classify",
    "predict",
    "image label",
    "ImageNet",
    "tag image",
    "most likely class",
    "scalar",
    "vision",
    "top_label",
    "label image",
]

# Described, guaranteed-runnable examples (VGI503/509/515). Both overloads of the
# shared function name carry the SAME aggregated example set (VGI515's "aggregate by
# function name" rule) so whichever overload row DuckDB surfaces the tags on, every
# example is present and described. Built with ``json.dumps`` so the BLOB literal's
# backslash-x escapes are correctly JSON-escaped (raw ``\x`` is invalid JSON; VGI507).
_TOP_LABEL_EXAMPLES_DATA = [
    {
        "description": "Predict the single most likely ImageNet label for an image BLOB literal.",
        "sql": f"SELECT vision.main.top_label('{SAMPLE_IMAGE_BLOB}'::BLOB) AS label",
    },
    {
        "description": "Predict the top label for an image read from a filesystem path.",
        "sql": f"SELECT vision.main.top_label('{_SAMPLE_IMAGE_PATH}') AS label",
    },
]
_TOP_LABEL_EXAMPLES = json.dumps(_TOP_LABEL_EXAMPLES_DATA)


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
        """VGI function metadata."""

        name = "top_label"
        description = "The #1 predicted ImageNet label for an image BLOB (NULL if not an image)"
        categories = ["vision", "classification"]
        tags = {
            **object_tags(
                title="Top Image Label (BLOB)",
                doc_llm=_TOP_LABEL_DOC_LLM,
                doc_md=_TOP_LABEL_DOC_MD,
                keywords=_TOP_LABEL_KEYWORDS,
            ),
            "vgi.category": "Image Classification",
            "vgi.example_queries": _TOP_LABEL_EXAMPLES,
            "vgi.executable_examples": _TOP_LABEL_EXAMPLES,
        }

    @classmethod
    def compute(
        cls, image: Annotated[pa.BinaryArray, Param(doc="Image bytes (PNG/JPEG/...).")]
    ) -> Annotated[pa.StringArray, Returns()]:
        """Return the top label for each image BLOB in the array."""
        out = [None if b is None else model.top_label_for(b) for b in image.to_pylist()]
        return pa.array(out, type=pa.string())


class TopLabelPathFunction(ScalarFunction):
    """``top_label(path)`` -- like ``top_label(image)`` but reads a file path."""

    class Meta:
        """VGI function metadata."""

        name = "top_label"
        description = "The #1 predicted ImageNet label for an image file path (NULL if unreadable)"
        categories = ["vision", "classification"]
        tags = {
            **object_tags(
                title="Top Image Label (from File Path)",
                doc_llm=_TOP_LABEL_DOC_LLM,
                doc_md=_TOP_LABEL_DOC_MD,
                keywords=_TOP_LABEL_KEYWORDS,
            ),
            "vgi.category": "Image Classification",
            "vgi.example_queries": _TOP_LABEL_EXAMPLES,
            "vgi.executable_examples": _TOP_LABEL_EXAMPLES,
        }

    @classmethod
    def compute(
        cls, path: Annotated[pa.StringArray, Param(doc="Filesystem path to an image file.")]
    ) -> Annotated[pa.StringArray, Returns()]:
        """Return the top label for the image at each filesystem path in the array."""
        out = [None if p is None else model.top_label_for(_read_path(p)) for p in path.to_pylist()]
        return pa.array(out, type=pa.string())


SCALAR_FUNCTIONS: list[type] = [
    TopLabelFunction,
    TopLabelPathFunction,
]
