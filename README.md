# peripheral-blood-cell-classification

Peripheral blood cell classification, from a PyTorch prototype on the desktop
to C++ inference on an edge device (Raspberry Pi).

- **Phase 1:** fine-tune a pretrained CNN on 8 cell classes; report balanced
  accuracy, macro-F1, per-class recall, confusion matrix and calibration
  (ECE, temperature scaling).
- **Phase 2:** export to ONNX, quantize to INT8, and run it with the ONNX
  Runtime C++ API on a Raspberry Pi, comparing accuracy and latency
  against the desktop.

Work is tracked in GitHub Issues under the Phase 1 and Phase 2 milestones.

## Quick start

Needs [uv](https://docs.astral.sh/uv/). It installs Python 3.12 from
`.python-version` if missing.

```bash
uv sync --extra cpu --extra export --extra viz   # cu126 instead of cpu for an NVIDIA GPU
uv run pytest                        # synthetic-data tests, no download

# after unzipping the dataset into data/ (see data/README.md)
uv run bloodcell-split
uv run bloodcell-train --arch mobilenet_v3_small --epochs 15
uv run bloodcell-eval runs/<run>/best.pt
uv run bloodcell-export runs/<run>/best.pt   # -> best.onnx + best.json
uv run bloodcell-plot runs/<run>             # figures, see docs/figures/STYLE.md
```

Edge build: [edge/README.md](edge/README.md).

## Results

Two small ImageNet-pretrained CNNs, fine-tuned for 15 epochs with the same
recipe: AdamW, one-cycle learning rate peaking at 1e-3, batch 64, label
smoothing 0.05, seed 0. They were trained with CUDA on a laptop GPU. The
checkpoint and the temperature are both chosen on the validation split. The
test split (2,562 images, stratified, deduplicated) is scored once.

| Model | Params | Test acc. | Balanced acc. | Macro-F1 | ECE, raw → T-scaled |
|---|--:|--:|--:|--:|--:|
| mobilenet_v3_small | 1.5 M | 98.8% | 98.9% | 98.9% | 4.4% → 0.6% |
| efficientnet_b0 | 4.0 M | 99.2% | 99.3% | 99.3% | 4.3% → 0.4% |
| *Acevedo et al. 2019, VGG-16 / Inception v3* | | *96% / 95%* | | | |

The paper's row is a reference point, not a like-for-like comparison. It
used its own split, and the duplicates were not removed.

- **The two models are effectively tied.** EfficientNet-B0 makes 21 errors
  on the test set and MobileNetV3-Small makes 31. With one seed on one
  split, that gap is within run-to-run noise. MobileNetV3-Small has under
  half the parameters, so it is the candidate for the Raspberry Pi.
- **Most errors are immature granulocytes (IG).** For both models, most
  mistakes are IG confused with neutrophils. IG covers several stages of
  neutrophil maturation, so this boundary is genuinely fuzzy.
- **Both models are underconfident before calibration.** The fitted
  temperatures are below 1 (0.61 and 0.64), partly because label smoothing
  trains the model never to aim for 100%. Temperature scaling doesn't
  change any prediction, and it cuts ECE to under 1%.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/per-class-recall-dark.png">
  <img alt="Test recall by class for both models: 100% on platelets and eosinophils, 98% or higher on every class except immature granulocyte, the weakest for both at 96.1% (MobileNetV3-Small) and 98.2% (EfficientNet-B0)." src="docs/figures/per-class-recall.png" width="640">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/training-curves-dark.png">
  <img alt="Training curves over 15 epochs: both models reach about 99% validation balanced accuracy by epoch 10, and validation ECE settles near 4.4% before calibration." src="docs/figures/training-curves.png" width="800">
</picture>

MobileNetV3-Small in detail:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/mobilenet_v3_small/confusion-matrix-dark.png">
  <img alt="MobileNetV3-Small confusion matrix on the test set: every class at 96% or higher on the diagonal; the largest error is 3.5% of immature granulocytes predicted as neutrophils." src="docs/figures/mobilenet_v3_small/confusion-matrix.png" width="560">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/mobilenet_v3_small/reliability-dark.png">
  <img alt="MobileNetV3-Small reliability diagram: temperature scaling (T = 0.61) cuts ECE from 4.4% to 0.6%. Most test images sit in the top confidence bin; the lower bins hold only a few images each, so their points are noisy." src="docs/figures/mobilenet_v3_small/reliability.png" width="480">
</picture>

These are internal scores. The split is image-level, and every image comes
from one lab and one analyser (see the limitation under Dataset), so they
say little about performance at another site.

## Dataset

Acevedo A, Merino A, Alférez S, Molina Á, Boldú L, Rodellar J. *A dataset of
microscopic peripheral blood cell images for development of automatic
recognition systems.* Data in Brief 30 (2020) 105474.
Mendeley Data, DOI [10.17632/snkd93bnjr.1](https://doi.org/10.17632/snkd93bnjr.1),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 17,092
images of normal cells in 8 classes, acquired with a CellaVision DM96 at the
Hospital Clinic of Barcelona.

Before splitting, 16 byte-identical duplicates are removed, plus both copies
of one image labelled as both eosinophil and neutrophil, leaving 17,074
images ([ADR 0004](docs/decisions/0004-deduplicate-before-splitting.md)).

This repository does not redistribute the images. `data/splits.csv` lists
the dataset's file names and class labels unchanged, plus the split each
image was assigned to.

**Limitation:** the dataset has no patient or smear IDs, so the split is
image-level and internal test scores are likely optimistic. External
validation on a second dataset (Raabin-WBC) is planned.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and checks. The
project is built with Claude Code as a coding assistant. The design,
validation and review are mine, and every change goes through tests and CI
before it is merged.

## License

Code: [BSD-3-Clause](LICENSE). The dataset keeps its own CC BY 4.0 license.
Models are fine-tuned from torchvision's ImageNet-pretrained weights, which
come with their own terms; see the
[torchvision model docs](https://docs.pytorch.org/vision/stable/models.html).
