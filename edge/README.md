# Edge inference

C++ classifier using the ONNX Runtime C++ API and OpenCV, intended for a
Raspberry Pi (the exact model is still to be decided). Preprocessing
(`cpp/preprocess.hpp`) gives the same model input as `src/bloodcell/model.py`,
including Pillow's antialiased resize; class names come from the `.json`
sidecar written by `bloodcell-export`.

## Build

Needs CMake, a C++17 compiler, OpenCV, and a prebuilt ONNX Runtime release
for the target's architecture (from
[github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)).

```bash
cmake -S edge/cpp -B build -DONNXRUNTIME_ROOT=<path to onnxruntime release>
cmake --build build
```

## Run

```bash
./build/bloodcell_infer <model.onnx> <image_or_dir> [--threads N] > preds.tsv
```

stdout is `path<TAB>label<TAB>confidence`; latency stats (p50/p95, img/s)
go to stderr. Confidence is temperature-scaled with the `temperature` from
the sidecar. The images can sit anywhere, as long as each is in a folder
named after its class, as in the dataset.

Score it with the same metrics as `bloodcell-eval`, optionally checking it
against Python's predictions on the same images:

```bash
uv run bloodcell-score preds.tsv --against runs/<run>/test_predictions.csv
```

## Parity with Python

`parity.sh <build dir>` checks, on synthetic images, that the model input
matches torchvision's value by value (`bloodcell_preprocess` writes it),
and that labels and confidences match Python's end to end. CI runs it. On
PBC's test split both baselines give the same label as Python on all
2,562 images, with confidences matching to the 4 decimals Python writes.
