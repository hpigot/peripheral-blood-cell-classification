# 3. uv, with torch as a per-machine extra

Status: accepted

## Context

Development runs on a CPU-only desktop, full training with CUDA on a
laptop GPU, and CI on CPU-only Linux runners. PyPI's default torch wheel is
not the right build for all three, and pip plus hand-written index URLs
isn't reproducible.

## Decision

Use uv with a committed `uv.lock`, and pin Python to 3.12 because torch
wheels for newer Pythons lag behind. torch and torchvision come from two
conflicting extras, `cpu` and `cu126`, each mapped to its PyTorch wheel
index. One lockfile covers both.

## Consequences

- Every install names its build: `uv sync --extra cpu` or `--extra cu126`.
- `data.py` and `metrics.py` must not import torch, because the base install
  doesn't include it.
- Moving to a newer CUDA build means changing the extra and index names in
  `pyproject.toml` and relocking.
