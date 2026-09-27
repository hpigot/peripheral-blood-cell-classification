# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow the project phases in CONTRIBUTING.md.

## [Unreleased]

### Added

- `bloodcell-shift`: test-time colour (Reinhard, Macenko) and size
  corrections on Raabin-WBC, with PBC controls; `--colour-jitter strong`
  for training. Neither recovers the external drop (docs/results.md).
- The ONNX export check runs on the whole PBC test split and writes
  `export_check.json`; the sidecar carries the fitted temperature, and the
  C++ program applies it.
- `bloodcell-score`: scores the C++ program's predictions with the same
  metrics as `bloodcell-eval`, optionally against Python's.
- `edge/parity.sh`, run in CI: checks the C++ model input against
  torchvision's value by value, and the predictions end to end.

### Fixed

- ONNX export crashed on a Windows console, and wrote the weights to a
  separate `.onnx.data` file that a device copy would miss.
- The C++ resize didn't antialias like Pillow's, which changed 7 of 2,562
  test predictions; it now matches exactly (ADR 0002).

## [0.1.0] - 2026-09-26

Phase 1: desktop prototype. Results are in [docs/results.md](docs/results.md).

### Added

- Stratified 70/15/15 split of the PBC dataset after removing duplicates,
  committed as `data/splits.csv` (ADR 0004).
- Training and evaluation of ImageNet-pretrained MobileNetV3-Small and
  EfficientNet-B0: balanced accuracy, macro-F1, per-class recall,
  confusion matrix, and ECE before and after temperature scaling
  (ADR 0005).
- Error analysis: per-image test predictions, top confused class pairs,
  and a figure of the most confident mistakes.
- `bloodcell-external`: scores a checkpoint on Raabin-WBC Test-A, another
  lab's images, without retraining.
- `bloodcell-plot`: figures in one style with light and dark variants.
- Provenance in every run and report: git SHA, dirty flag, split hash,
  package versions and device.
- ONNX export with a numerical check against PyTorch, and a C++ ONNX
  Runtime classifier for the edge that compiles in CI.
- CI for lint, types, tests, secrets and the C++ build; pre-commit hooks
  that keep datasets and weights out of git.

[Unreleased]: https://github.com/hpigot/peripheral-blood-cell-classification/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hpigot/peripheral-blood-cell-classification/releases/tag/v0.1.0
