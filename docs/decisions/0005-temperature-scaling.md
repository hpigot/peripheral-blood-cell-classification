# 5. Calibrate with temperature scaling

Status: accepted

## Context

A classifier's softmax output looks like a probability, but nothing in
training makes it one. If the model says 70%, it should be right about 70%
of the time: that's what would let a lab send low-confidence cells to a
person for review. We train with label smoothing (0.05), which pulls
targets away from 100%, so the models come out underconfident. At an
average stated confidence of 96%, MobileNetV3-Small was right on 2,326 of
2,329 test images.

## Decision

- Fit one temperature T on the validation split by minimising negative
  log-likelihood (`metrics.fit_temperature`: a log-spaced grid, then a finer
  grid around the best value; numpy only). Divide the logits by T before
  the softmax.
- Report expected calibration error (ECE) on the test split before and
  after: top-label confidence, 15 equal-width bins, weighted by count.
  `bloodcell-eval` writes both reports, and `bloodcell-plot` draws the
  reliability diagram with a count strip under it.

Temperature scaling has a single parameter, so it can't overfit the 2,561
validation images and it never changes the predicted class: dividing every
logit by the same positive number keeps their order. Vector or matrix
scaling (per-class parameters) and isotonic regression can fix more
complex miscalibration, but they fit many more parameters to the same
data, and a reliability diagram that is already close to the diagonal
after one parameter gives them nothing to fix.

## Consequences

- Baselines: T = 0.61 (MobileNetV3-Small) and 0.64 (EfficientNet-B0).
  Test ECE falls from 4.4% to 0.6% and from 4.3% to 0.4%. Accuracy is
  unchanged by construction.
- T belongs to the trained model. The edge build must apply it too, either
  from the export sidecar or folded into the ONNX graph, or the Pi's
  probabilities will be the uncalibrated ones. That's Phase 2 work (#22).
- ECE depends on the binning, and bins with a handful of images are noisy.
  Read it together with the count strip, not as a single number.
- Calibration here is measured on the same lab and analyser as training.
  It needs checking again on external data (#10), where confidence usually
  degrades.
