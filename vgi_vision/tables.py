"""Set-returning vision table functions.

``classify`` explodes one image into its top-k ``(label, confidence)`` predictions,
and ``image_classes`` enumerates the model's whole label set -- both naturally
**many rows out**, so they are table functions rather than scalars.

    SELECT * FROM vision.classify(image);             -- top 5, confidence desc
    SELECT * FROM vision.classify(image, 10);          -- top 10
    SELECT * FROM vision.classify('/tmp/cat.jpg');     -- VARCHAR path overload
    SELECT * FROM vision.image_classes();              -- 1000 ImageNet labels

Argument syntax: a *source* table function takes its inputs **positionally**. The
optional ``top_k`` cannot be a single-class default (positional args can't be
optional), so each arity is its own class sharing the function ``name`` --
``classify(image)`` and ``classify(image, top_k)`` -- the same arity-overload idiom
the scalar functions use. ``classify`` is offered for both a BLOB image and a
VARCHAR file path.

Robustness: a NULL / malformed / over-large image emits **no rows** (never crashes
the worker); the per-image guard lives in :mod:`vgi_vision.model`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pyarrow as pa
from vgi import Arg
from vgi.metadata import FunctionExample
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi_rpc import ArrowSerializableDataclass
from vgi_rpc.rpc import OutputCollector

from . import model
from ._examples import SAMPLE_IMAGE_BLOB
from .meta import object_tags
from .scalars import _SAMPLE_IMAGE_PATH, _read_path
from .schema_utils import field

_DEFAULT_TOP_K = 5

# Rows emitted per process() tick. Bounded so the scan cursor (``offset``) is
# observable across the HTTP limit-1 continuation boundary: correctness no longer
# depends on the whole result fitting inside a single producer batch. See
# ScanState below and CLAUDE.md "HTTP continuation" for the why.
ROWS_PER_TICK = 64


@dataclass(kw_only=True)
class ScanState(ArrowSerializableDataclass):
    """Externalized scan cursor for the vision table functions.

    Over the stateless HTTP transport the framework wire-serializes a producer's
    per-scan state through a continuation token after each ``process()`` tick (the
    client returns it; the worker resumes by deserializing it). A position-less
    state that emits *all* rows in one ``out.emit()`` and finishes therefore
    restarts from row 0 on every HTTP resume and loops forever once the output
    exceeds one producer batch. Carrying an explicit cursor here fixes that.

    Fields (all wire-serialize through the continuation token):

    * ``started`` -- flips once the (possibly heavy) source has been read and the
      full result batch materialized into ``rows_ipc``. Distinguishes
      "not yet computed" from "computed an empty result".
    * ``offset`` -- index of the next unemitted row; advanced at each emit.
    * ``rows_ipc`` -- the full materialized result as Arrow IPC stream bytes.
    """

    started: bool = False
    offset: int = 0
    rows_ipc: bytes = b""


def result_to_ipc(batch: pa.RecordBatch) -> bytes:
    """Serialize a single RecordBatch to Arrow IPC stream bytes (for ScanState)."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:  # type: ignore[no-untyped-call]
        writer.write_batch(batch)
    result: bytes = sink.getvalue().to_pybytes()
    return result


def ipc_to_table(value: bytes) -> pa.Table:
    """Deserialize Arrow IPC stream bytes (from ScanState) back to a Table."""
    reader = pa.ipc.open_stream(pa.BufferReader(value))  # type: ignore[no-untyped-call]
    return reader.read_all()


def _emit_cursor(state: ScanState, out: OutputCollector, schema: pa.Schema) -> None:
    """Emit one bounded ``ROWS_PER_TICK`` slice from ``state.offset``; finish when drained.

    ``state.started`` must already be set (``rows_ipc`` materialized). Advances
    ``state.offset`` past the emitted slice so a resumed tick (post wire round-trip)
    sees the new position and never re-emits row 0. An empty/zero-row result
    finishes immediately (``0 >= 0``).
    """
    table = ipc_to_table(state.rows_ipc)
    total = table.num_rows
    if state.offset >= total:
        out.finish()
        return
    end = min(state.offset + ROWS_PER_TICK, total)
    chunk = table.slice(state.offset, end - state.offset)
    # Advance the cursor BEFORE emitting: over http, emit() may suspend the tick
    # (limit-1 continuation boundary) and the framework wire-serializes the state
    # as it stands -- the advanced offset must already be recorded so the resumed
    # tick continues past this slice instead of re-emitting it.
    state.offset = end
    out.emit(chunk.combine_chunks().to_batches()[0])
    if state.offset >= total:
        out.finish()


_CLASSIFY_SCHEMA = pa.schema(
    [
        field("label", pa.string(), "Predicted ImageNet class label.", nullable=False),
        field("confidence", pa.float64(), "Softmax probability in [0, 1].", nullable=False),
    ]
)

_CLASSES_SCHEMA = pa.schema(
    [
        field("idx", pa.int32(), "0-based class index into the model's output.", nullable=False),
        field("label", pa.string(), "ImageNet class label.", nullable=False),
    ]
)

# Markdown column docs for the dynamic table-function schemas. DuckDB can't expose a
# table function's returned columns statically, so each function advertises them via a
# `vgi.result_columns_md` tag (consumed by describe/listing tooling and the metadata linter).
_CLASSIFY_COLUMNS_MD = (
    "| column | type | description |\n"
    "|---|---|---|\n"
    "| `label` | VARCHAR | Predicted ImageNet class label. |\n"
    "| `confidence` | DOUBLE | Softmax probability in [0, 1]; rows are confidence descending. |"
)

_CLASSES_COLUMNS_MD = (
    "| column | type | description |\n"
    "|---|---|---|\n"
    "| `idx` | INTEGER | 0-based class index into the model's output. |\n"
    "| `label` | VARCHAR | ImageNet class label. |"
)


_CLASSIFY_DOC_LLM = (
    "Explode one image into its **top-k ImageNet `(label, confidence)` predictions**, one "
    "row per prediction, ordered confidence-descending. Use it when you want ranked "
    "candidate labels with probabilities — not just the single best label (`top_label`) — "
    "e.g. to show the model's top guesses, threshold on confidence, or pick among close "
    "alternatives.\n\n"
    "It is a *source* table function: pass the image **positionally**. Overloads resolve "
    "by argument type and arity — `classify(image)` / `classify(image, top_k)` over a "
    "**BLOB**, and `classify(path)` / `classify(path, top_k)` over a **VARCHAR** filesystem "
    "path; `top_k` defaults to 5. A table-function argument cannot be a `(SELECT ...)` "
    "subquery (DuckDB rejects it), so pass a BLOB column/literal or a path string directly. "
    "A NULL/malformed/over-large/unreadable image emits **no rows** rather than crashing. "
    "Returns columns `label VARCHAR` and `confidence DOUBLE` (softmax probability in [0, 1])."
)

_CLASSIFY_DOC_MD = (
    "## `classify`\n\n"
    "Table function returning the **top-k `(label, confidence)`** ImageNet predictions for "
    "an image, confidence-descending.\n\n"
    "### Overloads\n\n"
    "- `classify(image BLOB[, top_k INTEGER])`\n"
    "- `classify(path VARCHAR[, top_k INTEGER])` — read the image off disk first.\n\n"
    "`top_k` defaults to 5.\n\n"
    "### Returns\n\n"
    "| column | type | description |\n"
    "|---|---|---|\n"
    "| `label` | VARCHAR | predicted ImageNet class label |\n"
    "| `confidence` | DOUBLE | softmax probability in [0, 1] |\n\n"
    "### Notes\n\n"
    "A NULL/malformed image yields **no rows**. A table-function argument cannot be a "
    "`(SELECT ...)` subquery — pass a BLOB column/literal or a path string."
)

_CLASSIFY_KEYWORDS = [
    "classify",
    "classification",
    "predict",
    "top-k",
    "label",
    "confidence",
    "probability",
    "ImageNet",
    "vision",
    "ranked predictions",
    "image labels",
    "softmax",
]

_CLASSES_DOC_LLM = (
    "Enumerate the classifier's **entire ImageNet-1k label set** as `(idx, label)` rows — "
    "all 1000 classes the model can predict. Takes no arguments. Use it to discover or "
    "validate which labels `top_label`/`classify` can return, to join predicted labels "
    "against the canonical class list, or to populate a dropdown/lookup of recognizable "
    "subjects. `idx` is the 0-based position into the model's output vector; `label` is the "
    "human-readable class name. Always returns exactly 1000 rows (the model is fixed)."
)

_CLASSES_DOC_MD = (
    "## `image_classes`\n\n"
    "No-argument discovery table function listing **every ImageNet-1k class** the model "
    "can predict (1000 rows).\n\n"
    "### Returns\n\n"
    "| column | type | description |\n"
    "|---|---|---|\n"
    "| `idx` | INTEGER | 0-based class index into the model output |\n"
    "| `label` | VARCHAR | ImageNet class label |\n\n"
    "### Notes\n\n"
    "Useful to validate or join against predicted labels from `top_label` / `classify`."
)

_CLASSES_KEYWORDS = [
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
]

# Per-function guaranteed-runnable examples (VGI509). Self-contained: BLOB literals
# for the BLOB overloads, the committed fixture path for the path overloads, and the
# no-arg discovery function. expected_result omitted (execution-only check). Built with
# ``json.dumps`` so the BLOB literal's backslash-x escapes are valid JSON (VGI507).
_CLASSIFY_BLOB_EXAMPLES = json.dumps(
    [
        {
            "description": "Top-5 (label, confidence) predictions for an image BLOB literal.",
            "sql": f"SELECT * FROM vision.main.classify('{SAMPLE_IMAGE_BLOB}'::BLOB)",
        }
    ]
)
_CLASSIFY_BLOB_TOPK_EXAMPLES = json.dumps(
    [
        {
            "description": "Top-3 (label, confidence) predictions for an image BLOB literal.",
            "sql": f"SELECT * FROM vision.main.classify('{SAMPLE_IMAGE_BLOB}'::BLOB, 3)",
        }
    ]
)
_CLASSIFY_PATH_EXAMPLES = json.dumps(
    [
        {
            "description": "Top-5 predictions for an image read from a filesystem path.",
            "sql": f"SELECT * FROM vision.main.classify('{_SAMPLE_IMAGE_PATH}')",
        }
    ]
)
_CLASSIFY_PATH_TOPK_EXAMPLES = json.dumps(
    [
        {
            "description": "Top-3 predictions for an image read from a filesystem path.",
            "sql": f"SELECT * FROM vision.main.classify('{_SAMPLE_IMAGE_PATH}', 3)",
        }
    ]
)
_CLASSES_EXAMPLES = json.dumps(
    [
        {
            "description": "Count the ImageNet classes the model can predict (1000).",
            "sql": "SELECT count(*) AS n FROM vision.main.image_classes()",
        },
        {
            "description": "Peek at the first five (idx, label) class rows.",
            "sql": "SELECT idx, label FROM vision.main.image_classes() WHERE idx < 5 ORDER BY idx",
        },
    ]
)


def _classify_batch(preds: list[tuple[str, float]] | None, schema: pa.Schema) -> pa.RecordBatch:
    """Build the full ``(label, confidence)`` batch for a set of predictions.

    A NULL/unclassifiable image (``preds`` is ``None``/empty) yields a zero-row
    batch -- the cursor then finishes with no rows, preserving the early-out
    contract. The cursor (not this helper) does the emit/finish.
    """
    if not preds:
        return pa.RecordBatch.from_pydict({"label": [], "confidence": []}, schema=schema)
    return pa.RecordBatch.from_pydict(
        {"label": [p[0] for p in preds], "confidence": [p[1] for p in preds]},
        schema=schema,
    )


def _process_classify(
    state: ScanState, out: OutputCollector, schema: pa.Schema, preds: list[tuple[str, float]] | None
) -> None:
    """Cursor-driven classify tick: materialize predictions on the first tick, then stream slices."""
    if not state.started:
        state.rows_ipc = result_to_ipc(_classify_batch(preds, schema))
        state.started = True
        state.offset = 0
    _emit_cursor(state, out, schema)


# ---------------------------------------------------------------------------
# classify(image) / classify(image, top_k)  -- BLOB overloads
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True, kw_only=True)
class _ClassifyBlobArgs:
    image: Annotated[bytes, Arg(0, doc="Image bytes (PNG/JPEG/...).")]


@dataclass(slots=True, frozen=True, kw_only=True)
class _ClassifyBlobTopKArgs:
    image: Annotated[bytes, Arg(0, doc="Image bytes (PNG/JPEG/...).")]
    top_k: Annotated[int, Arg(1, doc="Number of predictions to return.", ge=1)]


@init_single_worker
@bind_fixed_schema
class ClassifyFunction(TableFunctionGenerator[_ClassifyBlobArgs, ScanState]):
    """``classify(image)`` -- top-5 ImageNet predictions, confidence descending."""

    FunctionArguments = _ClassifyBlobArgs
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CLASSIFY_SCHEMA

    class Meta:
        """VGI function metadata."""

        name = "classify"
        description = "Top-5 ImageNet predictions (label, confidence) for an image BLOB"
        categories = ["vision", "classification"]
        tags = {
            **object_tags(
                title="Classify Image (BLOB, top-5)",
                doc_llm=_CLASSIFY_DOC_LLM,
                doc_md=_CLASSIFY_DOC_MD,
                keywords=_CLASSIFY_KEYWORDS,
            ),
            "vgi.result_columns_md": _CLASSIFY_COLUMNS_MD,
            "vgi.executable_examples": _CLASSIFY_BLOB_EXAMPLES,
        }
        examples = [
            FunctionExample(
                sql=f"SELECT * FROM vision.main.classify('{SAMPLE_IMAGE_BLOB}'::BLOB)",
                description="Top-5 predictions for an image BLOB literal",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_ClassifyBlobArgs]) -> TableCardinality:
        """Estimate the output row count (the default top-k)."""
        return TableCardinality(estimate=_DEFAULT_TOP_K, max=_DEFAULT_TOP_K)

    @classmethod
    def initial_state(cls, params: ProcessParams[_ClassifyBlobArgs]) -> ScanState:
        """Fresh scan cursor for this image's predictions."""
        return ScanState()

    @classmethod
    def process(cls, params: ProcessParams[_ClassifyBlobArgs], state: ScanState, out: OutputCollector) -> None:
        """Classify the image BLOB and stream the top-5 predictions via the cursor."""
        preds = None if state.started else model.classify_image(params.args.image, top_k=_DEFAULT_TOP_K)
        _process_classify(state, out, params.output_schema, preds)


@init_single_worker
@bind_fixed_schema
class ClassifyTopKFunction(TableFunctionGenerator[_ClassifyBlobTopKArgs, ScanState]):
    """``classify(image, top_k)`` -- top-k ImageNet predictions, confidence desc."""

    FunctionArguments = _ClassifyBlobTopKArgs
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CLASSIFY_SCHEMA

    class Meta:
        """VGI function metadata."""

        name = "classify"
        description = "Top-k ImageNet predictions (label, confidence) for an image BLOB"
        categories = ["vision", "classification"]
        tags = {
            **object_tags(
                title="Classify Image (BLOB, top-k)",
                doc_llm=_CLASSIFY_DOC_LLM,
                doc_md=_CLASSIFY_DOC_MD,
                keywords=_CLASSIFY_KEYWORDS,
            ),
            "vgi.result_columns_md": _CLASSIFY_COLUMNS_MD,
            "vgi.executable_examples": _CLASSIFY_BLOB_TOPK_EXAMPLES,
        }
        examples = [
            FunctionExample(
                sql=f"SELECT * FROM vision.main.classify('{SAMPLE_IMAGE_BLOB}'::BLOB, 10)",
                description="Top-10 predictions for an image BLOB literal",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_ClassifyBlobTopKArgs]) -> TableCardinality:
        """Estimate the output row count (the requested top-k)."""
        k = max(1, params.args.top_k)
        return TableCardinality(estimate=k, max=k)

    @classmethod
    def initial_state(cls, params: ProcessParams[_ClassifyBlobTopKArgs]) -> ScanState:
        """Fresh scan cursor for this image's predictions."""
        return ScanState()

    @classmethod
    def process(cls, params: ProcessParams[_ClassifyBlobTopKArgs], state: ScanState, out: OutputCollector) -> None:
        """Classify the image BLOB and stream the top-k predictions via the cursor."""
        preds = None if state.started else model.classify_image(params.args.image, top_k=params.args.top_k)
        _process_classify(state, out, params.output_schema, preds)


# ---------------------------------------------------------------------------
# classify(path) / classify(path, top_k)  -- VARCHAR path overloads
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True, kw_only=True)
class _ClassifyPathArgs:
    path: Annotated[str, Arg(0, doc="Filesystem path to an image file.")]


@dataclass(slots=True, frozen=True, kw_only=True)
class _ClassifyPathTopKArgs:
    path: Annotated[str, Arg(0, doc="Filesystem path to an image file.")]
    top_k: Annotated[int, Arg(1, doc="Number of predictions to return.", ge=1)]


@init_single_worker
@bind_fixed_schema
class ClassifyPathFunction(TableFunctionGenerator[_ClassifyPathArgs, ScanState]):
    """``classify(path)`` -- top-5 predictions for an image read off disk."""

    FunctionArguments = _ClassifyPathArgs
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CLASSIFY_SCHEMA

    class Meta:
        """VGI function metadata."""

        name = "classify"
        description = "Top-5 ImageNet predictions for an image file path"
        categories = ["vision", "classification"]
        tags = {
            **object_tags(
                title="Classify Image (File Path, top-5)",
                doc_llm=_CLASSIFY_DOC_LLM,
                doc_md=_CLASSIFY_DOC_MD,
                keywords=_CLASSIFY_KEYWORDS,
            ),
            "vgi.result_columns_md": _CLASSIFY_COLUMNS_MD,
            "vgi.executable_examples": _CLASSIFY_PATH_EXAMPLES,
        }
        examples = [
            FunctionExample(
                sql=f"SELECT * FROM vision.main.classify('{_SAMPLE_IMAGE_PATH}')",
                description="Top-5 predictions for an image on disk",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_ClassifyPathArgs]) -> TableCardinality:
        """Estimate the output row count (the default top-k)."""
        return TableCardinality(estimate=_DEFAULT_TOP_K, max=_DEFAULT_TOP_K)

    @classmethod
    def initial_state(cls, params: ProcessParams[_ClassifyPathArgs]) -> ScanState:
        """Fresh scan cursor for this image's predictions."""
        return ScanState()

    @classmethod
    def process(cls, params: ProcessParams[_ClassifyPathArgs], state: ScanState, out: OutputCollector) -> None:
        """Read the image off disk, classify it, and stream the top-5 predictions via the cursor."""
        preds = None if state.started else model.classify_image(_read_path(params.args.path), top_k=_DEFAULT_TOP_K)
        _process_classify(state, out, params.output_schema, preds)


@init_single_worker
@bind_fixed_schema
class ClassifyPathTopKFunction(TableFunctionGenerator[_ClassifyPathTopKArgs, ScanState]):
    """``classify(path, top_k)`` -- top-k predictions for an image read off disk."""

    FunctionArguments = _ClassifyPathTopKArgs
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CLASSIFY_SCHEMA

    class Meta:
        """VGI function metadata."""

        name = "classify"
        description = "Top-k ImageNet predictions for an image file path"
        categories = ["vision", "classification"]
        tags = {
            **object_tags(
                title="Classify Image (File Path, top-k)",
                doc_llm=_CLASSIFY_DOC_LLM,
                doc_md=_CLASSIFY_DOC_MD,
                keywords=_CLASSIFY_KEYWORDS,
            ),
            "vgi.result_columns_md": _CLASSIFY_COLUMNS_MD,
            "vgi.executable_examples": _CLASSIFY_PATH_TOPK_EXAMPLES,
        }
        examples = [
            FunctionExample(
                sql=f"SELECT * FROM vision.main.classify('{_SAMPLE_IMAGE_PATH}', 10)",
                description="Top-10 predictions for an image on disk",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_ClassifyPathTopKArgs]) -> TableCardinality:
        """Estimate the output row count (the requested top-k)."""
        k = max(1, params.args.top_k)
        return TableCardinality(estimate=k, max=k)

    @classmethod
    def initial_state(cls, params: ProcessParams[_ClassifyPathTopKArgs]) -> ScanState:
        """Fresh scan cursor for this image's predictions."""
        return ScanState()

    @classmethod
    def process(cls, params: ProcessParams[_ClassifyPathTopKArgs], state: ScanState, out: OutputCollector) -> None:
        """Read the image off disk, classify it, and stream the top-k predictions via the cursor."""
        preds = None if state.started else model.classify_image(_read_path(params.args.path), top_k=params.args.top_k)
        _process_classify(state, out, params.output_schema, preds)


# ---------------------------------------------------------------------------
# image_classes()  -- the model's label set
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class _NoArgs:
    """A discovery table function that takes no arguments."""


@init_single_worker
@bind_fixed_schema
class ImageClassesFunction(TableFunctionGenerator[_NoArgs, ScanState]):
    """``image_classes()`` -- every ``(idx, label)`` the classifier can predict."""

    FunctionArguments = _NoArgs
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CLASSES_SCHEMA

    class Meta:
        """VGI function metadata."""

        name = "image_classes"
        description = "The model's ImageNet label set: (idx, label), 1000 rows"
        categories = ["vision", "classification"]
        tags = {
            **object_tags(
                title="ImageNet Class List",
                doc_llm=_CLASSES_DOC_LLM,
                doc_md=_CLASSES_DOC_MD,
                keywords=_CLASSES_KEYWORDS,
            ),
            "vgi.result_columns_md": _CLASSES_COLUMNS_MD,
            "vgi.executable_examples": _CLASSES_EXAMPLES,
        }
        examples = [
            FunctionExample(
                sql="SELECT count(*) AS n FROM vision.main.image_classes()",
                description="How many classes the model predicts (1000)",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_NoArgs]) -> TableCardinality:
        """Estimate the output row count (the model's full label set)."""
        return TableCardinality(estimate=model.NUM_CLASSES, max=model.NUM_CLASSES)

    @classmethod
    def initial_state(cls, params: ProcessParams[_NoArgs]) -> ScanState:
        """Fresh scan cursor for the label-set enumeration."""
        return ScanState()

    @classmethod
    def process(cls, params: ProcessParams[_NoArgs], state: ScanState, out: OutputCollector) -> None:
        """Stream every ``(idx, label)`` the classifier can predict, ``ROWS_PER_TICK`` at a time.

        The label set is ~1000 rows -- well over one producer batch -- so the cursor
        is *required*: on the first tick we materialize the full table into
        ``state.rows_ipc``, then emit bounded slices so the offset survives each HTTP
        continuation round-trip.
        """
        if not state.started:
            rows = model.class_table()
            batch = pa.RecordBatch.from_pydict(
                {"idx": [r[0] for r in rows], "label": [r[1] for r in rows]},
                schema=params.output_schema,
            )
            state.rows_ipc = result_to_ipc(batch)
            state.started = True
            state.offset = 0
        _emit_cursor(state, out, params.output_schema)


TABLE_FUNCTIONS: list[type] = [
    ClassifyFunction,
    ClassifyTopKFunction,
    ClassifyPathFunction,
    ClassifyPathTopKFunction,
    ImageClassesFunction,
]
