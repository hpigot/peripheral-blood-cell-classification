"""Classification and calibration metrics (numpy only, no torch needed)."""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Top-label confidence vs accuracy in equal-width bins (non-empty bins only)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in itertools.pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            bins.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "confidence": float(conf[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return bins


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: count-weighted |accuracy - confidence| over the reliability bins."""
    bins = reliability_bins(probs, y, n_bins)
    return float(sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in bins) / len(y))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimizing validation NLL (1-D grid + refinement, no torch)."""

    def nll(t: float) -> float:
        p = softmax(logits, t)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())

    grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 200))
    best = min(grid, key=nll)
    fine = np.linspace(best * 0.9, best * 1.1, 101)
    return float(min(fine, key=nll))


def report(logits: np.ndarray, y: np.ndarray, classes: list[str], temperature: float = 1.0) -> dict:
    probs = softmax(logits, temperature)
    pred = probs.argmax(axis=1)
    labels = list(range(len(classes)))
    recall = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "ece": expected_calibration_error(probs, y),
        "temperature": float(temperature),
        "per_class_recall": dict(zip(classes, map(float, recall), strict=True)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "reliability": reliability_bins(probs, y),
    }
