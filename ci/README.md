# CI: the vgi-vision worker integration suite

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the unit tests
and this repo's sqllogictest suite (`test/sql/*.test`) against the vgi-vision
VGI worker through the **real DuckDB `vgi` extension** on every push / PR.

## How it works (no C++ build)

CI drives a **prebuilt** standalone `haybarn-unittest` and installs the
**signed** `vgi` extension from the Haybarn community channel:

1. **Install the worker** — `uv sync --frozen`. `vision_worker.py` is a PEP 723
   stdio worker spawned via `uv run vision_worker.py`. On first inference it
   downloads + caches the MobileNetV2-12 ONNX model (Apache-2.0) and ImageNet
   labels; CI runners have network access, so the E2E suite fetches them at run
   time.
2. **Download the runner** — the matching `haybarn_unittest-*` asset per platform.
3. **Preprocess** — [`preprocess-require.awk`](preprocess-require.awk) injects a
   signed `INSTALL vgi FROM community;` before each bare `LOAD vgi;`.
4. **Run** — [`run-integration.sh`](run-integration.sh) stages the preprocessed
   tree (including the `test/sql/data/sample.png` fixture), points
   `VGI_VISION_WORKER` at `uv run vision_worker.py`, and runs the suite.

## Run it locally

```bash
uv sync --python 3.13
HAYBARN_UNITTEST=/path/to/haybarn-unittest \
VGI_VISION_WORKER="uv run --python 3.13 vision_worker.py" \
  ci/run-integration.sh
```
