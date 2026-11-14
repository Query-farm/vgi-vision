"""Verify pinned public model assets and configure the worker's supported overrides."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen


class ModelAssets:
    """Download immutable model files atomically, without changing the worker."""

    def __init__(self):
        directory = Path(__file__).resolve().parent
        self.manifest = json.loads((directory / "model-manifest.json").read_text())
        cache = directory / "model-cache"
        cache.mkdir(exist_ok=True)
        for entry in self.manifest["files"]:
            destination = cache / entry["name"]
            if destination.exists() and self.valid(destination, entry):
                continue
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache, delete=False) as output:
                    temporary = Path(output.name)
                    with urlopen(entry["url"], timeout=120) as response:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                if not self.valid(temporary, entry):
                    raise ValueError(f"Model asset failed size/hash verification: {entry['name']}")
                temporary.replace(destination)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        settings = {
            "VGI_VISION_MODEL": str(cache / "mobilenetv2-12.onnx"),
            "VGI_VISION_LABELS": str(cache / "synset.txt"),
            "VGI_VISION_CACHE_DIR": str(cache),
        }
        self.previous = {key: os.environ.get(key) for key in settings}
        os.environ.update(settings)

    @staticmethod
    def valid(path, entry):
        """Check the exact bytes against the committed model manifest."""
        return (
            path.stat().st_size == entry["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        )

    def close(self):
        """Restore inherited model settings after the SQL worker has stopped."""
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
