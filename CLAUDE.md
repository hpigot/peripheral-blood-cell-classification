# peripheral-blood-cell-classification

Blood cell classifier: PyTorch training in `src/bloodcell/`, C++ ONNX Runtime
inference for a Raspberry Pi in `edge/cpp/`. Workflow and commit rules are
in CONTRIBUTING.md. Follow them.

## Commands

- Setup: `uv sync --extra cpu --extra export --extra viz`
- Before every commit: `uv run pre-commit run --all-files`, `uv run mypy`,
  `uv run pytest`
- Add dependencies with `uv add`, never pip, so `uv.lock` stays in sync.
  torch and torchvision come from the `cpu`/`cu126` extras, not the base
  dependencies.

## Invariants

- **Preprocessing contract.** `INPUT_SIZE`, `MEAN` and `STD` in `model.py`,
  the sidecar JSON written by `export.py`, and `kSize`/`kMean`/`kStd` in
  `edge/cpp/preprocess.hpp` must agree. If you change one, change all three
  in the same commit. `edge/parity.sh` (CI) checks the C++ model input
  against torchvision's, value by value.
- `data.py` and `metrics.py` must import without torch. Put torch code in
  `dataset.py`, `model.py`, `train.py`, `evaluate.py` or `export.py`.
- `data/splits.csv` defines the test set. Don't regenerate it unless an
  issue says so.
- The split is image-level because the PBC dataset has no patient IDs. Don't
  describe internal test scores as generalization. External validation is
  separate work.

## Figures

- Every result figure comes from `src/bloodcell/plots.py` and follows
  docs/figures/STYLE.md: at most 3 coloured series, colour fixed per model,
  and light and dark variants. Add a new plot type there, not in a notebook.

## Tests

- Synthetic data only: tiny images written to `tmp_path`, and
  `pretrained=False`. Tests never download the dataset or weights.
- Tests that need torch or onnxruntime start with `pytest.importorskip`.

## Don't

- Don't commit anything under `data/` except `README.md` and `splits.csv`,
  and don't commit weights (`*.pt`, `*.onnx`). A pre-commit hook blocks both.
- Don't commit personal config such as `.claude/settings.local.json`.
