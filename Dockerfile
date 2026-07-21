# Copyright 2026 Query Farm LLC - https://query.farm
#
# Single image serving BOTH transports of the vgi-vision worker:
#   docker run ... IMG            -> HTTP server on $PORT (default 8000; /health, VGI RPC)
#   docker run -i ... IMG stdio   -> stdio worker DuckDB spawns on-host
# See docker-entrypoint.sh. Inference runs on a permissively-licensed MobileNetV2 ONNX
# classifier (onnxruntime); the model + ImageNet labels are baked into the image so the
# first query and the /health boot are fast (no first-use download).
# syntax=docker/dockerfile:1
FROM python:3.13-slim

ARG VERSION=0.0.0
ARG GIT_COMMIT=unknown
ARG SOURCE_URL=https://github.com/Query-farm/vgi-vision

LABEL org.opencontainers.image.title="vgi-vision" \
      org.opencontainers.image.description="Image classification (ImageNet) on image blobs as DuckDB SQL functions via VGI (stdio + HTTP)" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.licenses="MIT" \
      farm.query.vgi.transports='["http","stdio"]'

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    VGI_VISION_CACHE_DIR=/app/.vision_cache

WORKDIR /app

# curl backs the HEALTHCHECK and the CI /health smoke.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install the worker + HTTP-serving extra from the source tree.
COPY pyproject.toml README.md LICENSE ./
COPY vgi_vision ./vgi_vision
RUN pip install '.[serve]'

# Pre-download the ONNX classifier + ImageNet labels into the baked cache so the first
# classify query (and the /health boot warm-up) never pays the download inline.
RUN python -c "from vgi_vision import model; model.warm_up(); print(model.model_path()); print(model.labels_path()); print(len(model.labels()), 'labels')"

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=8s \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
