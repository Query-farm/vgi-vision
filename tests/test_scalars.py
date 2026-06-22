"""Tests for the scalar vision functions (top_label).

Calls compute() directly with Arrow arrays -- the same path the worker drives per
batch. Robustness (NULL / garbage in a batch -> NULL out, good rows unaffected)
runs without the model; label assertions are gated on the model being present.
"""

from __future__ import annotations

import pyarrow as pa

from vgi_vision.scalars import TopLabelFunction, TopLabelPathFunction

from .harness import needs_model, png_bytes

GARBAGE = b"not an image at all \x00\x01\xff"


class TestTopLabelRobustness:
    def test_null_and_garbage_passthrough(self) -> None:
        arr = pa.array([None, GARBAGE, b""], type=pa.binary())
        out = TopLabelFunction.compute(arr)
        assert out.to_pylist() == [None, None, None]

    @needs_model
    def test_garbage_beside_good(self) -> None:
        # A garbage blob next to a real image: garbage -> NULL, good -> a label.
        arr = pa.array([GARBAGE, png_bytes((10, 120, 250)), None], type=pa.binary())
        out = TopLabelFunction.compute(arr).to_pylist()
        assert out[0] is None
        assert isinstance(out[1], str) and out[1]
        assert out[2] is None


@needs_model
class TestTopLabel:
    def test_returns_string(self) -> None:
        arr = pa.array([png_bytes((50, 60, 70))], type=pa.binary())
        out = TopLabelFunction.compute(arr).to_pylist()
        assert isinstance(out[0], str) and out[0]


class TestTopLabelPath:
    def test_missing_path_is_null(self) -> None:
        arr = pa.array([None, "/no/such/file.png"], type=pa.string())
        out = TopLabelPathFunction.compute(arr).to_pylist()
        assert out == [None, None]

    @needs_model
    def test_reads_real_file(self, tmp_path) -> None:
        p = tmp_path / "img.png"
        p.write_bytes(png_bytes((90, 90, 90)))
        arr = pa.array([str(p)], type=pa.string())
        out = TopLabelPathFunction.compute(arr).to_pylist()
        assert isinstance(out[0], str) and out[0]
