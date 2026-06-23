"""Integration tests for the vision table functions.

Drives ``classify``, its overloads, and ``image_classes`` through the real
bind -> init -> process lifecycle in-process (no worker subprocess). Model-gated
where inference is required.
"""

from __future__ import annotations

import pyarrow as pa

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
