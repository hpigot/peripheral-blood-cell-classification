# 6. Keep FP32 on the edge; post-training INT8 costs too much accuracy

Status: accepted

## Context

INT8 weights and activations make a model about 3.5× smaller, and on CPUs
with fast 8-bit arithmetic they can make it faster. The cost is accuracy:
each activation is rounded to 256 levels over a range fixed in advance
from calibration images. #36 asked what static post-training quantization
(PTQ) with ONNX Runtime costs on the two baselines.

Setup (`bloodcell-quantize`): QDQ format, per-channel INT8 weights, UINT8
activations, ranges calibrated on 500 random training images (seed 0).
Settings were compared on the validation split only; the test split was
scored once, with the setting chosen for each model.

| Validation balanced accuracy | MobileNetV3-Small | EfficientNet-B0 |
|---|--:|--:|
| FP32 | 99.3% | 99.3% |
| INT8, min-max ranges | 61.6% | 93.7% |
| INT8, ranges clipped at the 99.9th percentile | 91.2% | out of memory |
| INT8, the 24 worst tensors kept in FP32 (best ranges above) | 95.9% | 94.7% |

Other settings did worse: tighter or looser percentiles (99.999 to 98),
entropy calibration, quantizing only the convolutions, and keeping the
ReLU paths in FP32. MobileNetV3's activations have a few extreme values
that stretch the ranges; its ReLU outputs keep 0–5 dB of their signal
after quantization, and the loss is spread across the network rather
than in a few layers. ONNX Runtime's percentile calibration keeps every
activation of every calibration image in memory, which doesn't fit for
EfficientNet on a 16 GB machine.

## Decision

The edge runs the FP32 models. INT8 would have been adopted within about
1 point of FP32; the best all-INT8 models lose 6 to 7 points on the test
split:

| PBC test | Size | Balanced accuracy | ECE | Same class as FP32 |
|---|--:|--:|--:|--:|
| MobileNetV3-Small FP32 | 6.4 MB | 98.9% | 0.6% | |
| MobileNetV3-Small INT8 (percentile 99.9) | 1.8 MB | 91.6% | 2.7% | 94.5% |
| EfficientNet-B0 FP32 | 16.6 MB | 99.3% | 0.4% | |
| EfficientNet-B0 INT8 (min-max) | 5.2 MB | 93.3% | 1.6% | 95.5% |

`bloodcell-quantize` stays, with each model's best setting, so the INT8
files exist as a speed reference for the Pi benchmark (#37).

## Consequences

- Nothing about INT8 is needed for a working edge build; the Pi keeps the
  preprocessing contract and parity check as they are.
- INT8 isn't faster on this x86 laptop either: in the C++ program
  MobileNet takes 16.8 ms per image against 6.1 ms in FP32. The Pi's ARM
  cores may differ, which #37 measures.
- INT8 output is less reproducible. EfficientNet INT8 logits for an image
  change by up to 0.5 with the other images in its batch, and the C++
  program's INT8 confidences differ from Python's by up to 0.4 (labels
  agree on at least 99.8% of images). The FP32 models match exactly on
  both counts.
- Quantization-aware training would probably recover most of the loss, but
  it's a retraining project of its own. Revisit it only if FP32 turns out
  too slow on the Pi.
