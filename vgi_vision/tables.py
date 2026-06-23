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

from dataclasses import dataclass
from typing import Annotated, ClassVar

import pyarrow as pa
from vgi import Arg
from vgi.metadata import FunctionExample
from vgi.table_function import (
    BindParams,
    OutputCollector,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)

from . import model
from .scalars import _read_path
from .schema_utils import field

_DEFAULT_TOP_K = 5

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


def _emit_classify(preds: list[tuple[str, float]] | None, out: OutputCollector, schema: pa.Schema) -> None:
    """Emit one row per prediction (or nothing for a NULL/unclassifiable image)."""
    if not preds:
        out.emit(pa.RecordBatch.from_pydict({"label": [], "confidence": []}, schema=schema))
        out.finish()
        return
    out.emit(
        pa.RecordBatch.from_pydict(
            {"label": [p[0] for p in preds], "confidence": [p[1] for p in preds]},
            schema=schema,
        )
    )
    out.finish()


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
class ClassifyFunction(TableFunctionGenerator[_ClassifyBlobArgs]):
    """``classify(image)`` -- top-5 ImageNet predictions, confidence descending."""

    FunctionArguments = _ClassifyBlobArgs
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CLASSIFY_SCHEMA

    class Meta:
        name = "classify"
        description = "Top-5 ImageNet predictions (label, confidence) for an image BLOB"
        categories = ["vision", "classification"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM vision.classify((SELECT image FROM photos LIMIT 1))",
                description="Top-5 predictions for an image",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_ClassifyBlobArgs]) -> TableCardinality:
        return TableCardinality(estimate=_DEFAULT_TOP_K, max=_DEFAULT_TOP_K)

    @classmethod
    def process(cls, params: ProcessParams[_ClassifyBlobArgs], state: None, out: OutputCollector) -> None:
        preds = model.classify_image(params.args.image, top_k=_DEFAULT_TOP_K)
        _emit_classify(preds, out, params.output_schema)


@init_single_worker
@bind_fixed_schema
class ClassifyTopKFunction(TableFunctionGenerator[_ClassifyBlobTopKArgs]):
    """``classify(image, top_k)`` -- top-k ImageNet predictions, confidence desc."""

    FunctionArguments = _ClassifyBlobTopKArgs
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CLASSIFY_SCHEMA

    class Meta:
        name = "classify"
        description = "Top-k ImageNet predictions (label, confidence) for an image BLOB"
        categories = ["vision", "classification"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM vision.classify((SELECT image FROM photos LIMIT 1), 10)",
                description="Top-10 predictions for an image",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_ClassifyBlobTopKArgs]) -> TableCardinality:
        k = max(1, params.args.top_k)
        return TableCardinality(estimate=k, max=k)

    @classmethod
    def process(cls, params: ProcessParams[_ClassifyBlobTopKArgs], state: None, out: OutputCollector) -> None:
        preds = model.classify_image(params.args.image, top_k=params.args.top_k)
        _emit_classify(preds, out, params.output_schema)


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
class ClassifyPathFunction(TableFunctionGenerator[_ClassifyPathArgs]):
    """``classify(path)`` -- top-5 predictions for an image read off disk."""

    FunctionArguments = _ClassifyPathArgs
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CLASSIFY_SCHEMA

    class Meta:
        name = "classify"
        description = "Top-5 ImageNet predictions for an image file path"
        categories = ["vision", "classification"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM vision.classify('/tmp/cat.jpg')",
                description="Top-5 predictions for an image on disk",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_ClassifyPathArgs]) -> TableCardinality:
        return TableCardinality(estimate=_DEFAULT_TOP_K, max=_DEFAULT_TOP_K)

    @classmethod
    def process(cls, params: ProcessParams[_ClassifyPathArgs], state: None, out: OutputCollector) -> None:
        preds = model.classify_image(_read_path(params.args.path), top_k=_DEFAULT_TOP_K)
        _emit_classify(preds, out, params.output_schema)


@init_single_worker
@bind_fixed_schema
class ClassifyPathTopKFunction(TableFunctionGenerator[_ClassifyPathTopKArgs]):
    """``classify(path, top_k)`` -- top-k predictions for an image read off disk."""

    FunctionArguments = _ClassifyPathTopKArgs
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CLASSIFY_SCHEMA

    class Meta:
        name = "classify"
        description = "Top-k ImageNet predictions for an image file path"
        categories = ["vision", "classification"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM vision.classify('/tmp/cat.jpg', 10)",
                description="Top-10 predictions for an image on disk",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_ClassifyPathTopKArgs]) -> TableCardinality:
        k = max(1, params.args.top_k)
        return TableCardinality(estimate=k, max=k)

    @classmethod
    def process(cls, params: ProcessParams[_ClassifyPathTopKArgs], state: None, out: OutputCollector) -> None:
        preds = model.classify_image(_read_path(params.args.path), top_k=params.args.top_k)
        _emit_classify(preds, out, params.output_schema)


# ---------------------------------------------------------------------------
# image_classes()  -- the model's label set
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class _NoArgs:
    """A discovery table function that takes no arguments."""


@init_single_worker
@bind_fixed_schema
class ImageClassesFunction(TableFunctionGenerator[_NoArgs]):
    """``image_classes()`` -- every ``(idx, label)`` the classifier can predict."""

    FunctionArguments = _NoArgs
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CLASSES_SCHEMA

    class Meta:
        name = "image_classes"
        description = "The model's ImageNet label set: (idx, label), 1000 rows"
        categories = ["vision", "classification"]
        examples = [
            FunctionExample(
                sql="SELECT count(*) FROM vision.image_classes()",
                description="How many classes the model predicts (1000)",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_NoArgs]) -> TableCardinality:
        return TableCardinality(estimate=model.NUM_CLASSES, max=model.NUM_CLASSES)

    @classmethod
    def process(cls, params: ProcessParams[_NoArgs], state: None, out: OutputCollector) -> None:
        rows = model.class_table()
        out.emit(
            pa.RecordBatch.from_pydict(
                {"idx": [r[0] for r in rows], "label": [r[1] for r in rows]},
                schema=params.output_schema,
            )
        )
        out.finish()


TABLE_FUNCTIONS: list[type] = [
    ClassifyFunction,
    ClassifyTopKFunction,
    ClassifyPathFunction,
    ClassifyPathTopKFunction,
    ImageClassesFunction,
]
