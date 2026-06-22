"""In-process VGI invocation + test fixtures for the vision worker suite.

Drives a table function through the real bind -> init -> process lifecycle without
spawning a worker process, so most tests stay fast and debuggable. Also provides
small deterministic test images and a guard that skips model-dependent tests when
the classifier can't be loaded (e.g. offline), so a bare checkout stays green.

Adapted from the vgi-conform / vgi-nlp worker test suites.
"""

from __future__ import annotations

import io
from typing import Any

import pyarrow as pa
import pytest
from PIL import Image
from vgi.arguments import Arguments
from vgi.function_storage import BoundStorage, FunctionStorage, FunctionStorageSqlite
from vgi.invocation import FunctionType
from vgi.protocol import BindRequest, InitRequest
from vgi.table_function import ProcessParams

from vgi_vision import model

# ---------------------------------------------------------------------------
# Deterministic test images
# ---------------------------------------------------------------------------


def png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> bytes:
    """A solid-color PNG of the given size, as bytes."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def tiny_png() -> bytes:
    """A 1x1 PNG -- smallest valid image, exercises the resize path."""
    return png_bytes((255, 0, 0), size=(1, 1))


# ---------------------------------------------------------------------------
# Model availability gate
# ---------------------------------------------------------------------------


def model_available() -> bool:
    """True if the ONNX model + labels can be loaded (download already happened)."""
    try:
        model.warm_up()
        # warm_up swallows errors; confirm the session actually came up.
        return model.classify_image(png_bytes((8, 8, 8)), top_k=1) is not None
    except Exception:  # noqa: BLE001
        return False


needs_model = pytest.mark.skipif(not model_available(), reason="ONNX classifier model not available")


# ---------------------------------------------------------------------------
# In-process table-function driver
# ---------------------------------------------------------------------------


def test_storage() -> FunctionStorage:
    """Real in-memory FunctionStorage for the function lifecycle in tests."""
    return FunctionStorageSqlite(":memory:")


class MockOutputCollector:
    """Captures emitted batches for assertions."""

    def __init__(self, output_schema: pa.Schema) -> None:
        self.output_schema = output_schema
        self.batches: list[pa.RecordBatch] = []
        self._finished = False

    def emit(
        self,
        batch: pa.RecordBatch,
        partition_values: dict[str, Any] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.batches.append(batch)

    def finish(self) -> None:
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def emit_client_log_message(self, msg: Any) -> None:
        pass


def invoke_table_function(
    func_cls: type,
    *,
    named: dict[str, pa.Scalar] | None = None,
    positional: tuple[pa.Scalar, ...] = (),
) -> pa.Table:
    """Run a (source) table function through bind -> init -> process -> table."""
    args = Arguments(positional=positional, named=named or {})

    bind_req = BindRequest(
        function_name=func_cls.Meta.name,
        arguments=args,
        function_type=FunctionType.TABLE,
    )
    bind_resp = func_cls.bind(bind_req)

    init_req = InitRequest(bind_call=bind_req, output_schema=bind_resp.output_schema)
    init_resp = func_cls.global_init(init_req)

    storage = test_storage()
    params = ProcessParams(
        args=func_cls._parse_arguments(func_cls.FunctionArguments, args),
        init_call=init_req,
        init_response=init_resp,
        output_schema=bind_resp.output_schema,
        settings={},
        secrets={},
        storage=BoundStorage(storage, init_resp.execution_id),
    )

    state = func_cls.initial_state(params)
    out = MockOutputCollector(bind_resp.output_schema)

    while not out.finished:
        func_cls.process(params, state, out)

    return pa.Table.from_batches(out.batches, schema=bind_resp.output_schema)
