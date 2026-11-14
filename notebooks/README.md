# Image Classification example

| Notebook | Question | One-parameter experiment |
|---|---|---|
| [A top-K list is only part of an image classifier's answer](01_top_k_and_score_mass.ipynb) | How much of the model's full score distribution does a short list omit? | `TOP_K=5` → `20`, holding inputs/model fixed |

The notebook runs real **MobileNetV2-12 CPU inference through Haybarn/VGI** on an
unmodified CC0 cat photograph and an authored checkerboard, retaining malformed,
empty and missing controls. These illustrative images are not an accuracy benchmark.

Use **Open in Colab** and **Runtime → Run all**, or install [uv](https://docs.astral.sh/uv/)
and Git and run locally from this directory:

```sh
uv sync --locked
uv run --locked vgi-notebooks
uv run --locked vgi-notebooks 01_top_k_and_score_mass.ipynb --parameters '{"TOP_K":20}' --output-dir artifacts/top20
```

For Jupyter select `.venv/bin/python`. Colab's short visible bootstrap installs
only the shared base package; the locked `[runner]` extra is local tooling.
Unknown parameters fail, overrides require a separate destination, and failed
execution preserves the saved default. Use a fresh runtime if an older example
already replaced kernel libraries.

## Portable released runtime

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'vision' (TYPE vgi,
  LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-vision@v0.4.0 vgi-vision-worker');
```

This public release owns its dependencies in a separate Python 3.13 worker
environment. The short launch succeeds without the previous notebook's transitive
override stack. The tested environment resolved vgi-python 0.51.1, vgi-rpc 0.53.1,
ONNX Runtime 1.32.0, Pillow 13.0.0, NumPy 2.5.4, Hugging Face Hub 2.4.0 and
PyArrow 26.0.1. The notebook lock does not lock that separate worker environment.

Initial retrieval needs internet for Git/packages, the VGI extension, verified
companions and approximately **14 MB of model assets**. No worker checkout, root
virtual environment, external inference service or credential is required.
Inference uses the CPU provider with one intra-op and one inter-op thread. macOS
arm64 was executed; other platforms were not. Saved SQL credentials are isolated
in a fresh temporary directory with persistent secrets disabled.

[Shared v0.3.29 presentation](https://github.com/Query-farm/vgi-example-notebooks/blob/v0.3.29/docs/standard.md) supplies the exact catalog-derived H1 title/subtitle,
branding, visible SQL launcher, folded presentation code and SVG figures. Keep
`charts/` beside the saved notebook for GitHub image links; interactive runs
retain native SVG. Logs, model caches, review PNGs and experiment copies are
ignored. The fresh compatible-kernel test preloads eight common libraries before
executing the visible base-only setup and confirms their versions remain intact;
the hosted Colab browser UI was not manually operated.

## Model and image provenance

[model-manifest.json](model-manifest.json) pins the MobileNetV2-12 ONNX model to
Hugging Face revision `23a2a25eaaa6fcd61865414c8a1acf13dda66b7a` and the ImageNet
synset list to ONNX Model Zoo revision `c5cae0f1942c8d7727372c7b1f6b485aa39e4421`.
Both artifacts use Apache-2.0; see the [model card](https://huggingface.co/onnxmodelzoo/mobilenetv2-12).
The unchanged `model_assets.py` verifies exact size/SHA256, downloads into unique
temporary files, atomically replaces cache files and sets the worker's supported
model/label/cache overrides. Setup restores prior settings after worker cleanup.
Weights are never committed. When loaded through shared `prepare`, the cache lives
beside the immutable companion files under `.vgi-examples/vision/`.

[Image provenance](fixtures/provenance.json) and [attribution](fixtures/LICENSE.md)
identify **Chelsea the cat**, by **Stefan van der Walt**, released under CC0 through
scikit-image. Its original PNG bytes remain unchanged. `generate_fixture.py`
recreates the checkerboard and malformed bytes deterministically; it never
changes the photograph. The notebook verifies all three fixture hashes.

## Coverage and the release/current distinction

Complete actual released/current catalogs identify **three named surfaces**:
`top_label`, `classify` and `image_classes`. Seven overload rows and the same-named
`image_classes` catalog table are retained; overloads and table/function aliases
count once. This single focused story meaningfully covers **3/3** with visible
SQL, usable output and result-derived interpretation. It does not claim that
every overload or access kind is executed. There are no auth/write APIs omitted.

Release v0.4.0 resolves to `79393adf13923a964895ee9708f3beabbf35d469`; reviewed
current source is `aa30023808f21a34c91bcfdd0a581836dd757ff8`. Model/preprocessing
logic and root dependencies are unchanged. Current source adds metadata examples
and removes an erroneous trusted `UNIQUE(label)` declaration from the browsable
catalog table. The release still declares it despite **1,000 class indexes and
998 distinct labels**: “crane” and “maillot” repeat. The notebook deliberately uses
the callable `image_classes()` interface and materializes an ordinary local table.
It audits duplicates and avoids joining predictions to indexes by label.

Both complete raw constraint exports remain in `coverage/`; callable and column
signatures match. Current-compatible coverage refers to this reviewed callable
path, not a separately executed current-main notebook or a claim that the release's
catalog-table declaration is safe. `coverage.json` binds reviewed mappings to
actual catalogs and hashed execution receipts.

Existing green [source CI](https://github.com/Query-farm/vgi-vision/actions/runs/43587336715)
is reused. This change touches only notebooks: no worker rebuild, repeated full
suite, new worker release, image publication or deployment. The original local
checkout's source revision and untracked notebook state are preserved during
publication; only unchanged notebook files are synchronized afterward.

## Read score mass without treating it as accuracy

The worker converts decoded images to RGB, bilinearly resizes directly to 224×224,
scales channels to 0–1 and applies ImageNet normalization. It does not preserve the
photograph's aspect ratio. Scalar `top_label` accepts a column; table `classify`
requires a literal path/blob. The notebook materializes each invocation once.

The full reference has 1,000 class entries per valid image. The fixed top-five
chart retains their original scores on a common 0–100% scale. The cumulative
chart always shows ranks 1–50 with the same fraction scale; only the selected-K
markers move. Full-distribution fingerprints let the experiment verify all 2,000
score rows remain identical, including duplicate display names.

A non-NULL label is neither an out-of-distribution test nor proof that the named
object appears. Softmax mass is not calibrated correctness, model confidence in
an unknown-class sense, error probability or benchmark accuracy. Increasing K
reveals more of the same distribution; it does not improve the underlying model.
Malformed, empty and missing controls stay distinct from scored valid images.
Model loading failures remain errors, never fabricated predictions.

Fourteen SQL checks verify input handling, the 1,000-index vocabulary, scalar/table
agreement, finite bounded scores, totals within 1e-10, multiset shortlist membership
and nondecreasing cumulative mass. Different CPU/library versions can change
floating-point scores; the model bytes and tested runtime versions are explicit.
