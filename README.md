# peripheral-blood-cell-classification

Classifying white blood cells, platelets and erythroblasts in blood smear
images: a PyTorch prototype, then C++ inference on a Raspberry Pi.

## The finding

Two small pretrained CNNs score 99% balanced accuracy on the PBC dataset's
own test split. On images from another lab (Raabin-WBC), without
retraining, the same checkpoints drop to 39% and 57%. The other lab's
stain is pinker, and MobileNet calls 54% of its monocytes eosinophils, the
class defined by pink-orange granules. Normalising the colour, or training
the model to ignore it, removes that error but not the drop: the errors
move to other classes.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/external-recall-dark.png">
  <img alt="Recall by class on Raabin-WBC for both models. EfficientNet-B0: monocyte 85%, lymphocyte 74%, neutrophil 59%, eosinophil 37%, basophil 27%. MobileNetV3-Small: eosinophil 79%, neutrophil 60%, lymphocyte 37%, monocyte 12%, basophil 5%." src="docs/figures/external-recall.png" width="640">
</picture>

| Balanced accuracy | Params | PBC test | Raabin-WBC |
|---|--:|--:|--:|
| MobileNetV3-Small | 1.5 M | 98.9% | 38.5% |
| EfficientNet-B0 | 4.0 M | 99.3% | 56.5% |

- **The internal tie hides a real difference.** The models are 10 test
  images apart on PBC and 18 points apart on the other lab.
- **Calibration doesn't transfer.** Temperature scaling cuts ECE on PBC to
  under 1%, and roughly doubles it on Raabin-WBC. A confidence threshold
  tuned on one lab can't be trusted at another.
- **Most internal errors are one boundary:** metamyelocyte vs band
  neutrophil, neighbouring maturation stages that PBC puts in different
  classes.

Details, figures and caveats: [docs/results.md](docs/results.md).

## Status

- **Phase 1, desktop prototype:** done.
- **Phase 2, edge:** ONNX export works and the C++ ONNX Runtime program
  compiles in CI; INT8 quantization and Pi benchmarks are next
  ([#22](../../issues/22)).

## Quick start

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra cpu --extra export --extra viz   # cu126 for an NVIDIA GPU
uv run pytest                                    # synthetic data, no download

# after unzipping the datasets into data/ (see data/README.md)
uv run bloodcell-split
uv run bloodcell-train --arch mobilenet_v3_small --epochs 15
uv run bloodcell-eval runs/<run>/best.pt
uv run bloodcell-external runs/<run>/best.pt     # Raabin-WBC
uv run bloodcell-shift runs/<run>/best.pt        # colour and size corrections
uv run bloodcell-export runs/<run>/best.pt       # -> best.onnx + best.json
uv run bloodcell-plot runs/<run>
```

Edge build: [edge/README.md](edge/README.md). Design decisions:
[docs/decisions/](docs/decisions/).

## Data

[PBC](https://doi.org/10.17632/snkd93bnjr.1) (Acevedo et al., Data in Brief
2020, CC BY 4.0): 17,092 cell images in 8 classes from a CellaVision DM96,
Hospital Clinic of Barcelona. 18 duplicates are removed before splitting
([ADR 0004](docs/decisions/0004-deduplicate-before-splitting.md)). The
dataset has no patient IDs, so the split is image-level and internal scores
are likely optimistic; that is why the external test matters.

[Raabin-WBC](https://www.nature.com/articles/s41598-021-04426-x)
(Kouzehkanan et al., Sci Rep 2022) Test-A: 4,339 cells, 5 classes shared
with PBC. Neither dataset is redistributed here.

## Development

Workflow and checks: [CONTRIBUTING.md](CONTRIBUTING.md). Built with Claude
Code as a coding assistant. The design, validation and review are mine, and
every change goes through tests and CI before it is merged.

## License

Code: [BSD-3-Clause](LICENSE). The datasets keep their own licenses.
Models are fine-tuned from torchvision's ImageNet weights, which have
[their own terms](https://docs.pytorch.org/vision/stable/models.html).
