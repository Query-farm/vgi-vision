"""Integration tests for the vision table functions.

Drives ``classify``, its overloads, and ``image_classes`` through the real
bind -> init -> process lifecycle in-process (no worker subprocess). Model-gated
where inference is required.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

from vgi_vision import tables
from vgi_vision.tables import (
    ClassifyFunction,
    ClassifyTopKFunction,
    ImageClassesFunction,
)

from .harness import invoke_table_function, needs_model, png_bytes

GARBAGE = b"definitely not an image \x00\xff\x01"


class TestImageClasses:
    @needs_model
    def test_count_is_1000(self) -> None:
        table = invoke_table_function(ImageClassesFunction)
        assert table.column_names == ["idx", "label"]
        assert table.num_rows == 1000

    @needs_model
    def test_idx_is_sequential(self) -> None:
        table = invoke_table_function(ImageClassesFunction)
        idx = table.column("idx").to_pylist()
        assert idx == list(range(1000))


class TestClassify:
    @needs_model
    def test_top5_default(self) -> None:
        table = invoke_table_function(
            ClassifyFunction, positional=(pa.scalar(png_bytes((40, 80, 160)), type=pa.binary()),)
        )
        assert table.column_names == ["label", "confidence"]
        assert table.num_rows == 5
        confs = table.column("confidence").to_pylist()
        assert confs == sorted(confs, reverse=True)
        assert all(0.0 <= c <= 1.0 for c in confs)

    @needs_model
    def test_top_k_override(self) -> None:
        table = invoke_table_function(
            ClassifyTopKFunction,
            positional=(
                pa.scalar(png_bytes((10, 10, 10)), type=pa.binary()),
                pa.scalar(3, type=pa.int64()),
            ),
        )
        assert table.num_rows == 3

    def test_garbage_image_no_rows(self) -> None:
        # Hostile input through the table path: no rows, no crash. Needs no model
        # (the decode guard rejects it before inference).
        table = invoke_table_function(ClassifyFunction, positional=(pa.scalar(GARBAGE, type=pa.binary()),))
        assert table.column_names == ["label", "confidence"]
        assert table.num_rows == 0

    def test_empty_image_no_rows(self) -> None:
        table = invoke_table_function(ClassifyFunction, positional=(pa.scalar(b"", type=pa.binary()),))
        assert table.num_rows == 0


# ---------------------------------------------------------------------------
# HTTP-continuation regression: the scan cursor must survive being wire-
# serialized/deserialized between every process() tick (as the stateless HTTP
# transport does across its limit-1 continuation boundary). On the OLD
# emit-all-then-finish code with `state: None`, a serialized resume re-emits from
# row 0 forever -- these tests would overrun the tick guard (hang -> AssertionError)
# or see duplicated rows. On the cursor code the offset survives and the scan
# terminates with each row emitted exactly once.
# ---------------------------------------------------------------------------


class TestScanStateRoundTrip:
    """image_classes() (~1000 rows >> ROWS_PER_TICK) drives the real cursor path."""

    @needs_model
    def test_serialize_between_ticks_matches_single_shot(self) -> None:
        plain = invoke_table_function(ImageClassesFunction)
        rt = invoke_table_function(ImageClassesFunction, serialize_state=True)

        assert plain.num_rows == 1000
        # Identical rows AND order, despite a wire round-trip on every tick.
        assert rt.to_pylist() == plain.to_pylist()
        # No duplicates: idx is the dense 0..999 range exactly once.
        idx = rt.column("idx").to_pylist()
        assert idx == list(range(1000))
        assert len(set(idx)) == 1000

    @needs_model
    def test_terminates_in_expected_tick_count(self) -> None:
        # ceil(1000 / ROWS_PER_TICK) emitting ticks + 1 finishing tick at most.
        expected = math.ceil(1000 / tables.ROWS_PER_TICK)
        # A tight guard: if the cursor regressed, this overruns and raises.
        table = invoke_table_function(ImageClassesFunction, serialize_state=True, max_ticks=expected + 2)
        assert table.num_rows == 1000

    @needs_model
    def test_small_chunk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force many tiny slices so the cursor must advance through ~500 ticks,
        # each across a serialize boundary -- a stress test of offset survival.
        monkeypatch.setattr(tables, "ROWS_PER_TICK", 2)
        rt = invoke_table_function(ImageClassesFunction, serialize_state=True)
        assert rt.column("idx").to_pylist() == list(range(1000))


class TestCursorSurvivesContinuation:
    """Offline classify round-trip: monkeypatch classify_image to span many ticks.

    Runs without the ONNX weights so the regression is covered even on a bare
    checkout (the image_classes tests above are model-gated).
    """

    def test_many_synthetic_preds_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n = 200  # >> ROWS_PER_TICK (64): forces multiple continuation slices.
        synthetic = [(f"label-{i:04d}", 1.0 - i / n) for i in range(n)]
        monkeypatch.setattr(tables.model, "classify_image", lambda *a, **k: synthetic)

        img = pa.scalar(png_bytes((1, 2, 3)), type=pa.binary())
        plain = invoke_table_function(ClassifyFunction, positional=(img,))
        rt = invoke_table_function(ClassifyFunction, positional=(img,), serialize_state=True)

        assert plain.num_rows == n
        assert rt.num_rows == n
        # Byte-identical rows AND order across the wire round-trip.
        assert rt.to_pylist() == plain.to_pylist()
        labels = rt.column("label").to_pylist()
        assert labels == [p[0] for p in synthetic]
        assert len(set(labels)) == n  # no dupes

    def test_small_chunk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tables, "ROWS_PER_TICK", 2)
        synthetic = [(f"c{i}", 0.5) for i in range(50)]
        monkeypatch.setattr(tables.model, "classify_image", lambda *a, **k: synthetic)
        img = pa.scalar(png_bytes((9, 9, 9)), type=pa.binary())
        rt = invoke_table_function(
            ClassifyTopKFunction, positional=(img, pa.scalar(50, type=pa.int64())), serialize_state=True
        )
        assert rt.column("label").to_pylist() == [p[0] for p in synthetic]

    def test_no_preds_still_terminates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The NULL/empty-image early-out path: zero rows, finishes immediately,
        # even with serialize-between-ticks (0 >= 0).
        monkeypatch.setattr(tables.model, "classify_image", lambda *a, **k: None)
        img = pa.scalar(png_bytes((0, 0, 0)), type=pa.binary())
        rt = invoke_table_function(ClassifyFunction, positional=(img,), serialize_state=True, max_ticks=5)
        assert rt.num_rows == 0
