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
Results go here once measured.

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

| Model | Test acc. | Balanced acc. | Macro-F1 | ECE (raw / T-scaled) |
|---|---|---|---|---|
| mobilenet_v3_small | | | | |
| efficientnet_b0 | | | | |

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
