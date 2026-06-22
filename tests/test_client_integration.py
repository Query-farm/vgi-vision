"""End-to-end tests through ``vgi.client.Client``, spawning the real worker.

These exercise the full Arrow-IPC round trip the way DuckDB would: the worker runs
as a subprocess and we drive it over stdin/stdout. Gated on the model being
available, so a bare/offline checkout stays green.
"""

from __future__ import annotations

import os
import shlex
import sys

import pyarrow as pa
from vgi import Arguments
from vgi.client import Client

from .harness import needs_model, png_bytes

_WORKER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vision_worker.py")


def _client() -> Client:
    # Launch the worker with the same interpreter running the tests, so it shares
    # the installed deps (rather than going through `uv run`). Client wants a
    # shell-style command string.
    return Client(f"{shlex.quote(sys.executable)} {shlex.quote(_WORKER)}")


@needs_model
def test_classify_table_end_to_end() -> None:
    img = png_bytes((60, 120, 180))
    with _client() as client:
        batches = list(
            client.table_function(
                function_name="classify",
                arguments=Arguments(positional=[pa.scalar(img, type=pa.binary())]),
            )
        )
    table = pa.Table.from_batches(batches)
    assert table.column_names == ["label", "confidence"]
    assert table.num_rows == 5
    confs = table.column("confidence").to_pylist()
    assert confs == sorted(confs, reverse=True)


@needs_model
def test_top_label_scalar_end_to_end() -> None:
    # The image arrives as the (sole) input column; its BINARY type selects the
    # blob overload of top_label. No positional const arg is passed -- the column
    # is the argument (mirrors DuckDB's `top_label(image)` over a BLOB column).
    batch = pa.RecordBatch.from_pydict(
        {"image": pa.array([png_bytes((200, 30, 30)), None], type=pa.binary())}
    )
    with _client() as client:
        results = list(
            client.scalar_function(
                function_name="top_label",
                input=iter([batch]),
                arguments=Arguments(positional=[]),
            )
        )
    out = results[0]["result"].to_pylist()
    assert isinstance(out[0], str) and out[0]
    assert out[1] is None


@needs_model
def test_image_classes_end_to_end() -> None:
    with _client() as client:
        batches = list(client.table_function(function_name="image_classes"))
    table = pa.Table.from_batches(batches)
    assert table.num_rows == 1000
