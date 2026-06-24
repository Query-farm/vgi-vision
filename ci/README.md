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
   tree (including the `test/sql/data/sample.png` fixture), resolves the ATTACH
   `LOCATION` per `$TRANSPORT` (see below), warms the extension cache, and runs
   the suite in a single `haybarn-unittest` invocation.

## Transport matrix (subprocess | http | unix)

The same `test/sql/*.test` suite is exercised over all three VGI transports. The
vgi extension picks the transport from the `LOCATION` string the `.test` files
`ATTACH`, and `run-integration.sh` builds that string from `$TRANSPORT`:

| `TRANSPORT`  | `VGI_VISION_WORKER` (LOCATION)      | How the worker is reached |
|--------------|------------------------------------|---------------------------|
| `subprocess` | `uv run … vision_worker.py`        | extension spawns the worker per query; Arrow IPC over stdin/stdout (default) |
| `http`       | `http://127.0.0.1:<port>`          | harness boots `vision_worker.py --http --port 0 --port-file <f>`, waits for the port-file, ATTACHes that URL |
| `unix`       | `unix:///tmp/vision-<pid>.sock`     | harness boots `vision_worker.py --unix <sock>`, waits for the socket, ATTACHes it |

The CI `integration` job is a `transport: [subprocess, http, unix]` × `os`
matrix; each leg runs `ci/run-integration.sh` with `TRANSPORT=<t>`.

- **http** needs DuckDB's `httpfs` extension (the vgi HTTP transport is built on
  it), so the script injects `INSTALL httpfs FROM core; LOAD httpfs;` into each
  staged file on the http leg only. It also needs the worker's `http` extra
  (waitress) — CI installs it via `uv sync --frozen --extra http`, and the PEP
  723 header in `vision_worker.py` lists `vgi-python[http]` so `uv run` resolves
  it. The worker is booted with cwd = the staging dir so VARCHAR-path fixture
  overloads (`classify('test/sql/data/sample.png')`) resolve.
- **Silent-skip guard**: DuckDB's sqllogictest runner auto-SKIPs (exit 0!) any
  statement whose error contains "HTTP" / "Unable to connect" — a broken http
  setup would green while testing nothing. The script therefore requires the
  runner to print `All tests passed (N …)` with N > 0 and fails the leg
  otherwise.

## Run it locally

```bash
uv sync --python 3.13 --extra http
# subprocess (default), then http, then unix:
HAYBARN_UNITTEST=/path/to/haybarn-unittest TRANSPORT=subprocess ci/run-integration.sh
HAYBARN_UNITTEST=/path/to/haybarn-unittest TRANSPORT=http       ci/run-integration.sh
HAYBARN_UNITTEST=/path/to/haybarn-unittest TRANSPORT=unix       ci/run-integration.sh
```
