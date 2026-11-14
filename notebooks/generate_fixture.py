"""Regenerate the small deterministic checkerboard and malformed-input controls."""

from pathlib import Path

from PIL import Image


def main():
    """Write a 224×224 RGB eight-by-eight checkerboard and invalid byte sequence."""
    directory = Path(__file__).resolve().parent / "fixtures"
    directory.mkdir(exist_ok=True)
    image = Image.new("RGB", (224, 224))
    image.putdata(
        [(240, 240, 240) if (x // 28 + y // 28) % 2 else (25, 25, 25) for y in range(224) for x in range(224)]
    )
    image.save(directory / "checkerboard.png", optimize=False)
    (directory / "invalid.bin").write_bytes(b"This is a synthetic malformed image control.\x00\xff")


if __name__ == "__main__":
    main()
