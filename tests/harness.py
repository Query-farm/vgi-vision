"""In-process VGI invocation + test fixtures for the vision worker suite.

Drives a table function through the real bind -> init -> process lifecycle without
spawning a worker process, so most tests stay fast and debuggable. Also provides
small deterministic test images and a guard that skips model-dependent tests when
the classifier can't be loaded (e.g. offline), so a bare checkout stays green.

Adapted from the vgi-conform / vgi-nlp worker test suites.
"""

from __future__ import annotations

import contextlib
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
    """Captures emitted batches for assertions.

    ``batch_limit`` models the HTTP transport's per-response producer batch limit:
    once that many batches have been emitted in a single ``process()`` tick, further
    emits raise ``_BatchLimitReached`` so the driver can suspend, wire-serialize the
    state, and resume (exactly as the http server does across a continuation token).
    ``None`` (the default) means unbounded -- the in-process behaviour.
    """

    def __init__(self, output_schema: pa.Schema, batch_limit: int | None = None) -> None:
        self.output_schema = output_schema
        self.batches: list[pa.RecordBatch] = []
        self._finished = False
        self._batch_limit = batch_limit
        self._emitted_this_tick = 0

    def begin_tick(self) -> None:
        self._emitted_this_tick = 0

    def emit(
        self,
        batch: pa.RecordBatch,
        partition_values: dict[str, Any] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.batches.append(batch)
        self._emitted_this_tick += 1
        if self._batch_limit is not None and self._emitted_this_tick >= self._batch_limit:
            raise _BatchLimitReached

    def finish(self) -> None:
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def emit_client_log_message(self, msg: Any) -> None:
        pass


class _BatchLimitReached(Exception):
    """Internal: the per-tick producer batch limit was hit; suspend + resume."""


def invoke_table_function(
    func_cls: type,
    *,
    named: dict[str, pa.Scalar] | None = None,
    positional: tuple[pa.Scalar, ...] = (),
    serialize_state: bool = False,
    max_ticks: int = 1000,
) -> pa.Table:
    """Run a (source) table function through bind -> init -> process -> table.

    When ``serialize_state=True`` the driver faithfully models the stateless HTTP
    transport: each ``process()`` tick may emit at most ONE producer batch (the
    limit-1 continuation boundary), after which the per-scan state is wire-
    serialized and deserialized
    (``type(state).deserialize_from_bytes(state.serialize_to_bytes())``) before the
    next tick resumes. A position-less state that re-emits row 0 on every resume
    loops forever; the ``max_ticks`` guard turns that into a loud failure instead
    of an infinite hang. With a cursor state the offset survives each round-trip
    and the scan terminates after ~ceil(rows / ROWS_PER_TICK) ticks.
    """
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
    # Over http, each response carries at most one producer batch; model that with
    # batch_limit=1 so the cursor must be observable across the boundary.
    out = MockOutputCollector(bind_resp.output_schema, batch_limit=1 if serialize_state else None)

    ticks = 0
    while not out.finished:
        if serialize_state and state is not None:
            state = type(state).deserialize_from_bytes(state.serialize_to_bytes())
        out.begin_tick()
        # _BatchLimitReached suspends the tick mid-flight exactly as the http server
        # does once it has filled a response with one batch; the loop then resumes
        # with the (serialized) state.
        with contextlib.suppress(_BatchLimitReached):
            func_cls.process(params, state, out)
        ticks += 1
        if ticks > max_ticks:
            raise AssertionError(
                f"{func_cls.__name__} did not finish after {max_ticks} ticks "
                f"(serialize_state={serialize_state}): the scan cursor is not "
                f"surviving the continuation boundary (likely re-emitting from row 0)."
            )

    return pa.Table.from_batches(out.batches, schema=bind_resp.output_schema)
