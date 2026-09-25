# 2. Preprocessing contract between Python and C++

Status: accepted

## Context

The model is trained in Python with torchvision transforms and run on the Pi
by C++ code with OpenCV. If the two preprocess images differently (colour
order, resize, scaling, normalization, layout), accuracy drops silently. No
error is raised.

## Decision

Evaluation preprocessing is fixed: resize to 224×224 (bilinear), RGB, scale
to [0, 1], normalize with ImageNet mean/std, NCHW float32. It is defined in
`model.py` (`INPUT_SIZE`, `MEAN`, `STD`), written to the `.json` sidecar by
`bloodcell-export`, and mirrored in `edge/cpp/main.cpp`. Class names come
only from the sidecar.

## Consequences

- A change to preprocessing has to touch all three places in one commit.
- The C++ side still hardcodes the constants rather than reading them from
  the sidecar. A Phase 2 parity test (same images through PyTorch and
  `bloodcell_infer`, compare predictions) will catch drift.
- torchvision resizes PIL images with antialiasing and OpenCV's
  `INTER_LINEAR` doesn't, so small numerical differences are expected. The
  parity test decides whether they matter.
