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
    return _bins(probs.max(axis=1), probs.argmax(axis=1) == y, n_bins)


def _bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict]:
    correct = correct.astype(float)
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


def _ece(bins: list[dict], n: int) -> float:
    return float(sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in bins) / n)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: count-weighted |accuracy - confidence| over the reliability bins."""
    return _ece(reliability_bins(probs, y, n_bins), len(y))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimizing validation NLL (1-D grid + refinement, no torch)."""

    def nll(t: float) -> float:
        p = softmax(logits, t)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())

    grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 200))
    best = min(grid, key=nll)
    fine = np.linspace(best * 0.9, best * 1.1, 101)
    return float(min(fine, key=nll))


def confused_pairs(cm: np.ndarray | list, classes: list[str], k: int = 5) -> list[dict]:
    """The k largest off-diagonal cells, as true class, predicted class, count and share."""
    counts = np.asarray(cm)
    pairs = [
        {
            "true": classes[i],
            "predicted": classes[j],
            "count": int(counts[i, j]),
            "share_of_true": float(counts[i, j] / counts[i].sum()),
        }
        for i, j in zip(*np.nonzero(counts), strict=True)
        if i != j
    ]
    return sorted(pairs, key=lambda p: (-p["count"], p["true"], p["predicted"]))[:k]


def report(logits: np.ndarray, y: np.ndarray, classes: list[str], temperature: float = 1.0) -> dict:
    probs = softmax(logits, temperature)
    out = label_report(y, probs.argmax(axis=1), probs.max(axis=1), classes)
    return {**out, "temperature": float(temperature)}


def label_report(y: np.ndarray, pred: np.ndarray, conf: np.ndarray, classes: list[str]) -> dict:
    """The report from predicted labels and top-label confidences alone (e.g. from C++)."""
    labels = list(range(len(classes)))
    recall = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    bins = _bins(conf, pred == y)
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "ece": _ece(bins, len(y)),
        "per_class_recall": dict(zip(classes, map(float, recall), strict=True)),
        "confusion_matrix": cm.tolist(),
        "confused_pairs": confused_pairs(cm, classes),
        "reliability": bins,
    }
