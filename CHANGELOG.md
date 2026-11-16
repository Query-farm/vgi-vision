# Changelog

## 0.4.2 (2026-11-16)

- Bump vgi-python to 0.52.2 (`>=0.52.2,<0.53`).
- Python 3.14 (requires-python floor, CI, Docker image and tooling).
- SQL integration tests run on every transport (stdio, launch, http) through
  the starter's `ci/run-integration.sh`; `make check` runs all three.

## 0.4.1 — 2026-11-14

- The worker's own ruff run excludes `notebooks/` (`[tool.ruff] extend-exclude`).
  The notebooks are maintained by separate tooling with its own lint config, and
  CI's repo-wide `ruff format --check .` was failing on them.
- Examples show real output for every user, not just inside a checkout of this
  repo (vgi-lint VGI913/VGI516): the path-overload examples that read
  `test/sql/data/sample.png` are replaced by the embedded BLOB literal for
  `classify` and by `read_blob()` of the fixture published at the v0.4.0 tag
  for `top_label`. The composed classify examples and the
  most-likely-animal agent task use the BLOB literal.
