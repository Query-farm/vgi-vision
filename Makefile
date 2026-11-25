# vgi-vision — dev and test targets.
#
# Usage:
#   make fetch-model # download + cache the ONNX classifier and ImageNet labels
#   make test        # unit/integration (pytest) + end-to-end SQL (haybarn-unittest)
#   make test-unit   # pytest only
#   make test-sql    # DuckDB sqllogictest .test files via haybarn-unittest

# SQL tests: ci/run-integration.sh (adapted from the vgi-starter template) runs
# test/sql/*.test through the published haybarn-unittest with the signed community
# `vgi` extension, over stdio, launch (AF_UNIX, pooled) and http. CI runs the same
# script per transport. `make test-sql TRANSPORTS=launch` runs one.
HAYBARN_VERSION  ?= 2.0.1rc3
WORKER           ?= $(CURDIR)/.venv/bin/vgi-vision-worker
TOOLS            := $(CURDIR)/.tools
HAYBARN_UNITTEST := $(TOOLS)/bin/haybarn-unittest
TRANSPORTS       ?= stdio launch http

.PHONY: build tools check test test-unit test-sql lint fetch-model

build: ## Install the worker (with every extra) into .venv
	uv sync --locked --all-extras

tools: ## Install the pinned SQL test runner into .tools
	@test -x $(HAYBARN_UNITTEST) || { uv venv -q $(TOOLS) && uv pip install -q -p $(TOOLS) haybarn-unittest==$(HAYBARN_VERSION); }

test-sql: build tools ## SQL tests through DuckDB over each of $(TRANSPORTS)
	@for t in $(TRANSPORTS); do \
		WORKER="$(WORKER)" HAYBARN_UNITTEST="$(HAYBARN_UNITTEST)" TRANSPORT=$$t ci/run-integration.sh || exit 1; \
	done

# Download the model + labels into the cache (no-op if already present). The
# worker also downloads on first use; this just front-loads it for CI / offline.
fetch-model:
	uv run --no-sync python -c "from vgi_vision import model; print(model.model_path()); print(model.labels_path()); print(len(model.labels()), 'labels')"

test: test-unit test-sql

test-unit:
	uv run pytest -q

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy vgi_vision/

check: lint test ## Everything CI runs
