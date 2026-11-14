# Vision notebooks

| Notebook | Question | Default / experiment |
|---|---|---|
| [Top-K and score mass](01_top_k_and_score_mass.ipynb) | How much of a classifier's full distribution does a short list omit? | `TOP_K=5` → `20`, holding images and model fixed |

The saved example runs **real MobileNetV2-12 CPU inference through Haybarn/VGI**. Inputs are an unmodified CC0 cat photograph, a generated checkerboard and deliberately invalid/missing controls. The two images are illustrative, not an accuracy benchmark. It distinguishes successful image decoding, the model's forced choice among ImageNet classes, and the softmax mass excluded by a short list.

## Run from notebooks alone

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then download/copy this complete directory. No root checkout, worker build or root virtual environment is required. First use needs internet access for dependencies, the community VGI extension, the public Git worker and approximately **14 MB** of model assets. The worker uses Python 3.13; uv can provision it. The separate locked notebook environment supports Python 3.13–3.14. Inference uses ONNX Runtime's CPU provider with one intra-op and one inter-op thread.

```sh
cd notebooks
uv sync --frozen
uv run --frozen python execute.py
uv run --frozen python execute.py --parameters '{"TOP_K":20}' --output-dir validation/top20
```

The runner uses a temporary kernel and notebook-directory working directory, writes outputs only after success, and requires a separate destination for overrides. It copies documentation, fixtures, assets and environment companions into experiment folders. Validation artifacts and `model-cache/` are ignored; the default notebook retains embedded outputs. To verify a fresh model download, copy this directory without `.venv/`, `validation/` or `model-cache/`, then run the commands again.

For Jupyter, register/select this environment and restart/run all:

```sh
uv run --frozen python -m ipykernel install --user --name vision-notebooks --display-name 'Python (Vision notebooks)'
```

The visible literal ATTACH fetches Git revision `b603c7867c2c6a00211688b729250b85c69f5507`, with tested pins for VGI SDK/RPC, ONNX Runtime, Pillow, NumPy, Hugging Face Hub and Arrow. The notebook engine is published Haybarn `2.0.0rc11`. Upgrading requires editing the visible launcher and rerunning both parameter cases; changing the notebook lock alone does not upgrade the worker.

## Models, inputs and interpretation

[model-manifest.json](model-manifest.json) pins the MobileNetV2-12 ONNX file to Hugging Face revision `23a2a25eaaa6fcd61865414c8a1acf13dda66b7a` and the ImageNet synset list to ONNX Model Zoo Git revision `c5cae0f1942c8d7727372c7b1f6b485aa39e4421`. Both are Apache-2.0 artifacts; the [model card](https://huggingface.co/onnxmodelzoo/mobilenetv2-12) documents the model. `model_assets.py` verifies byte sizes and SHA-256 before setting the worker's documented `VGI_VISION_MODEL`, `VGI_VISION_LABELS` and cache overrides. Downloads use unique temporary files and atomic replacement, so simultaneous preparations cannot expose a partial model. No weights are committed. Overrides are restored after cleanup. The Haybarn connection uses empty temporary secret storage and disables persistent secrets.

[Image provenance](fixtures/provenance.json) and [attribution](fixtures/LICENSE.md) identify the original **Chelsea** photograph by **Stefan van der Walt**, released under CC0 via scikit-image. The image is copied without modification from a pinned source revision. `generate_fixture.py` recreates the checkerboard and invalid byte fixture deterministically; it never changes the photograph:

```sh
uv run --frozen python generate_fixture.py
```

The worker converts images to RGB, bilinearly resizes to 224×224 without preserving aspect ratio, and applies ImageNet channel normalization. Classification uses a fixed 1,000-class vocabulary. A non-NULL label is not an out-of-distribution check, and a softmax score is not calibrated correctness. `classify` needs literal file/blob input; the scalar `top_label` is used for the image column. Full reference results and each K-sized source call are materialized once before display. The class-index catalog has two duplicate display names; the notebook avoids a label-to-index join that would multiply rows.

The experiment changes only report length. Full-reference scores, scalar labels, input gallery and top-five chart must remain identical, while returned rows grow and retained mass cannot decrease. The cumulative chart uses fixed ranks 1–50 and a fixed mass scale. No score is renormalized over the displayed shortlist. Malformed, empty and missing inputs stay distinct from scored images; model-load failures fail execution rather than fabricating outputs. CPU/platform/library changes may produce small numerical differences, so the selected runtime/model pins matter.
