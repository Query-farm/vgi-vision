# vgi-vision — dev and test targets.
#
# Usage:
#   make fetch-model # download + cache the ONNX classifier and ImageNet labels
#   make test        # unit/integration (pytest) + end-to-end SQL (haybarn-unittest)
#   make test-unit   # pytest only
#   make test-sql    # DuckDB sqllogictest .test files via haybarn-unittest
#
# test-sql is self-contained: it points VGI_VISION_WORKER at the worker run as a
# uv stdio subprocess (exactly how DuckDB drives it after ATTACH) and runs the
# files under test/sql/. haybarn-unittest is a uv tool:
#   uv tool install haybarn-unittest   # installs ~/.local/bin/haybarn-unittest

# Worker command DuckDB uses for ATTACH (overridable).
WORKER_STDIO    ?= uv run --python 3.13 vision_worker.py

# haybarn-unittest lives in the uv tools bin; keep it on PATH.
HAYBARN_BIN     ?= $(HOME)/.local/bin
TEST_DIR         = .
TEST_PATTERN     = test/sql/*

.PHONY: test test-unit test-sql lint fetch-model

# Download the model + labels into the cache (no-op if already present). The
# worker also downloads on first use; this just front-loads it for CI / offline.
fetch-model:
	uv run --no-sync python -c "from vgi_vision import model; print(model.model_path()); print(model.labels_path()); print(len(model.labels()), 'labels')"

test: test-unit test-sql

test-unit:
	uv run pytest -q

test-sql:
	PATH="$(HAYBARN_BIN):$$PATH" \
		VGI_VISION_WORKER="$(WORKER_STDIO)" \
		haybarn-unittest --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy vgi_vision/
