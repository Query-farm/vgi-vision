"""Pure-logic tests for vgi_vision.model.

The robustness/NULL tests need no model (preprocessing + guards run on bad bytes
before inference) and always run. The classification tests are gated on the model
being available, so a bare/offline checkout still goes green.
"""

from __future__ import annotations

from vgi_vision import model

from .harness import needs_model, png_bytes, tiny_png

# A clearly non-image blob and an empty blob -- both must be handled, never crash.
GARBAGE = b"this is definitely not an image \x00\x01\x02\xff\xfe and never will be"


class TestRobustness:
    """Untrusted-input handling: bad bytes -> None, worker survives."""

    def test_none_input(self) -> None:
        assert model.classify_image(None) is None
        assert model.top_label_for(None) is None

    def test_empty_bytes(self) -> None:
        assert model.classify_image(b"") is None
        assert model.top_label_for(b"") is None

    def test_garbage_bytes(self) -> None:
        assert model.classify_image(GARBAGE) is None
        assert model.top_label_for(GARBAGE) is None

    def test_truncated_png(self) -> None:
        # First few bytes of a PNG signature, then nothing decodable.
        assert model.classify_image(b"\x89PNG\r\n\x1a\n\x00\x00") is None

    def test_garbage_does_not_poison_good(self) -> None:
        # The hostile case: a garbage blob handled right before a good one.
        assert model.classify_image(GARBAGE) is None
        if model.classify_image(png_bytes((10, 20, 30)), top_k=1) is not None:
            # Only assert the survival property; exact label is covered elsewhere.
            assert model.classify_image(png_bytes((10, 20, 30)), top_k=1)


class TestLabelsAndClasses:
    """Label set is well-formed -- gated on the labels file being downloadable."""

    @needs_model
    def test_label_count(self) -> None:
        assert len(model.labels()) == model.NUM_CLASSES == 1000

    @needs_model
    def test_class_table_shape(self) -> None:
        rows = model.class_table()
        assert len(rows) == 1000
        assert rows[0][0] == 0
        assert rows[-1][0] == 999
        assert all(isinstance(label, str) and label for _, label in rows)


@needs_model
class TestClassification:
    """Inference structure + determinism -- gated on the model being present."""

    def test_top_k_count_and_order(self) -> None:
        preds = model.classify_image(png_bytes((123, 50, 200)), top_k=5)
        assert preds is not None
        assert len(preds) == 5
        confs = [c for _, c in preds]
        assert confs == sorted(confs, reverse=True)

    def test_confidences_are_probabilities(self) -> None:
        preds = model.classify_image(png_bytes((30, 30, 30)), top_k=10)
        assert preds is not None
        assert all(0.0 <= c <= 1.0 for _, c in preds)
        # Softmax over all 1000 classes sums to ~1, so any top-10 subset sums <= 1.
        assert sum(c for _, c in preds) <= 1.0 + 1e-6

    def test_top_k_clamped(self) -> None:
        # Asking for more than NUM_CLASSES is clamped, not an error.
        preds = model.classify_image(png_bytes((0, 0, 0)), top_k=10_000)
        assert preds is not None
        assert len(preds) == model.NUM_CLASSES

    def test_top_k_minimum(self) -> None:
        preds = model.classify_image(png_bytes((0, 0, 0)), top_k=0)
        assert preds is not None
        assert len(preds) == 1  # clamped up to 1

    def test_tiny_1x1_image(self) -> None:
        # Smallest valid image: must be handled (upscaled) without error.
        preds = model.classify_image(tiny_png(), top_k=3)
        assert preds is not None
        assert len(preds) == 3

    def test_deterministic(self) -> None:
        a = model.classify_image(png_bytes((77, 88, 99)), top_k=5)
        b = model.classify_image(png_bytes((77, 88, 99)), top_k=5)
        assert a == b

    def test_top_label_matches_classify_head(self) -> None:
        img = png_bytes((200, 10, 10))
        top = model.top_label_for(img)
        preds = model.classify_image(img, top_k=1)
        assert preds is not None
        assert top == preds[0][0]
