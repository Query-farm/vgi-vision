# CLAUDE.md — vgi-vision

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://query.farm) worker that runs **image classification** on image
blobs as DuckDB SQL functions. Inference runs out-of-process on a
permissively-licensed ONNX model (MobileNetV2-12, ImageNet-1k) via `onnxruntime`.
`vision_worker.py` assembles every function into one `vision` catalog (single
`main` schema) over stdio. Sibling style/tooling to `vgi-conform`, `vgi-nlp`,
`vgi-audio` (the blob-input sibling — copy its `.test` patterns).

## Layout

```
vision_worker.py       repo-root stdio entry point; PEP 723 inline deps; warm-up; main()
vgi_vision/
  model.py             pure inference: download/cache, warm_up, preprocess, classify_image — no Arrow/VGI
  scalars.py           per-row scalar top_label (BLOB + VARCHAR-path arity overloads)
  tables.py            classify(image[, top_k]) / classify(path[, top_k]) / image_classes()
  schema_utils.py      pa.Field comment / column-doc helper
tests/                 pytest: test_model (pure), test_tables (in-proc), test_scalars + test_client_integration (Client RPC)
test/sql/*.test        haybarn-unittest sqllogictest — authoritative E2E
test/sql/data/         committed tiny deterministic fixtures (sample.png)
Makefile               fetch-model / test / test-unit / test-sql / lint
```

To add a function: put pure logic in `model.py` (total — never raises on garbage
image bytes; returns `None`), wrap it as a scalar or table function in the
matching module, register it in `vision_worker.py`'s catalog.

## THE load-bearing decision: licensing

This worker is for a **commercial marketplace**, so model licensing is the whole
point. **Do not** reintroduce Ultralytics / YOLOv8 — they are **AGPL-3.0**. The
shipped stack is all permissive: MobileNetV2-12 weights (**Apache-2.0**, from the
ONNX Model Zoo via the `onnxmodelzoo/mobilenetv2-12` HF mirror), `onnxruntime`
(MIT), Pillow (HPND), NumPy (BSD). Any new model **must** be Apache/BSD/MIT and
its exact name + license documented in the README "Models & licensing" section.
Detection is intentionally **not** shipped — see the README's "future work" note.

## Scalars vs table functions — core convention

VGI **scalar functions are positional-only** (`name := value` is table-function
only), and overloads resolve by **arity + type**:

- `top_label` is a scalar with two arity-1 overloads sharing `Meta.name`: a BLOB
  overload (`top_label(image)`) and a VARCHAR-path overload (`top_label(path)`).
  The framework disambiguates by the input **column type** (binary vs string).
- `classify` is a **source table function** taking its image **positionally** (a
  `bytes` `Arg(0)` → BLOB, or `str` `Arg(0)` → path). The optional `top_k` can't
  be a single-class default (positional args can't be optional), so each arity is
  its own `TableFunctionGenerator` subclass: `classify(image)` / `classify(image,
  top_k)` / `classify(path)` / `classify(path, top_k)`.
- `image_classes()` is a no-arg discovery table function.

LIST/STRUCT-style fixed schemas use `bind_fixed_schema` + `FIXED_SCHEMA` and the
`field()` helper for column comments.

## Sharp edges (learned the hard way)

1. **`haybarn-unittest` skips `require vgi`** — under haybarn the extension is
   built in, so the `.test` files use `statement ok` + `LOAD vgi;` (never
   `require vgi`) and `require-env VGI_VISION_WORKER`.
2. **A table function argument cannot be a `(SELECT ...)` subquery.** DuckDB
   rejects `classify((SELECT content FROM read_blob('x.png')))` in the `.test`
   context with "Table function cannot contain subqueries". The SQL E2E therefore
   drives `classify` via the **VARCHAR path overload** (`classify('x.png')`), the
   same trick `vgi-audio` uses. (The BLOB overload is still exercised from Python
   and via literal `'...'::BLOB` in `hostile.test`.)
3. **BLOB→hex string escapes**: in `'...'::BLOB` literals only `\xNN` escapes are
   valid; `\r`/`\n` are rejected by DuckDB's string→blob conversion. Write a
   truncated-PNG fixture as full hex (`\x89\x50\x4E\x47...`).
4. **Row scan order is not guaranteed in SQL**, so the descending-confidence
   *emission* property is asserted in pytest (where `out.emit` order is
   observable), not in `.test`. SQL asserts the head-is-max invariant instead.
5. **Untrusted images**: every decode is wrapped and bounded (`_MAX_PIXELS` caps
   decompression bombs). Bad bytes → `None` → SQL NULL / no rows; the worker must
   never crash on a single hostile row. `test/sql/hostile.test` is the contract.
6. **Expensive init is cached + warmed**: the ORT session + labels load once per
   process (`@lru_cache`), and `vision_worker.run()` calls `model.warm_up()` at
   spawn so the first query of each ATTACH is fast and the E2E suite is stable.
7. **Model is downloaded, never committed.** `*.onnx` and `models/` are
   gitignored; the cache lives under `~/.cache/vgi-vision` (`VGI_VISION_CACHE_DIR`
   to override; `VGI_VISION_MODEL` / `VGI_VISION_LABELS` to point at local files).
   `make fetch-model` front-loads the download; the worker also fetches on first
   use. Model-dependent tests skip cleanly when it's absent (offline checkout).

## Verify

```sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev
make fetch-model
uv run --no-sync pytest -q
make test-sql
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_vision/
```
