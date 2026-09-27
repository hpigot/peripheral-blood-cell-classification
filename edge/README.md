# Edge inference

C++ classifier using the ONNX Runtime C++ API and OpenCV, intended for a
Raspberry Pi (the exact model is still to be decided). Preprocessing
mirrors `src/bloodcell/model.py`; class names come from the `.json` sidecar
written by `bloodcell-export`.

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
the sidecar. Scoring `preds.tsv` against the split CSV is a Phase 2 task.
