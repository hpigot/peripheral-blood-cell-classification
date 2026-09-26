"""External validation: score a PBC checkpoint on Raabin-WBC, without retraining.

Raabin-WBC (Kouzehkanan et al., Sci Rep 2022) comes from another lab, with a
different microscope, camera and staining, and covers five of PBC's eight
classes. The internal test split can't measure that kind of shift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import Sample
from .metrics import report
from .provenance import device_name, file_sha256, provenance

# The label files only hold numbers. This mapping reproduces the class counts
# the Raabin-WBC paper gives for Test-A (neutrophil 2,660, lymphocyte 1,034,
# monocyte 234, eosinophil 322, basophil 89), and a sample of images per
# label was checked by eye.
RAABIN_LABELS = {1: "neutrophil", 2: "lymphocyte", 3: "monocyte", 4: "eosinophil", 5: "basophil"}


def read_raabin(labels_json: Path, images: Path) -> list[Sample]:
    labels = json.loads(Path(labels_json).read_text())
    return [Sample(Path(images) / name, RAABIN_LABELS[k]) for name, k in sorted(labels.items())]


def score_external(
    logits: np.ndarray, labels: list[str], classes: list[str], temperature: float
) -> dict:
    """Score logits over the model's classes against labels from a subset of them.

    Two views: forced choice among the classes the external set has (the domain
    shift alone), and a free choice among all the model's classes, where
    predicting a class the set doesn't contain counts as wrong (as in use).
    """
    present = set(labels)
    shared = [c for c in classes if c in present]
    cols = [classes.index(c) for c in shared]
    y_shared = np.array([shared.index(label) for label in labels])
    y_all = np.array([classes.index(label) for label in labels])
    pred = logits.argmax(axis=1)
    cm = [
        [int(((y_all == classes.index(t)) & (pred == j)).sum()) for j in range(len(classes))]
        for t in shared
    ]
    recall = {t: cm[i][classes.index(t)] / sum(cm[i]) for i, t in enumerate(shared)}
    absent = [j for j, c in enumerate(classes) if c not in present]
    return {
        "n": len(labels),
        "temperature": float(temperature),
        "forced_choice": {
            "uncalibrated": report(logits[:, cols], y_shared, shared),
            "temperature_scaled": report(logits[:, cols], y_shared, shared, temperature),
        },
        "all_classes": {
            "accuracy": float((pred == y_all).mean()),
            "balanced_accuracy": float(np.mean(list(recall.values()))),
            "per_class_recall": recall,
            # rows: true classes of the external set; columns: every model class
            "confusion_matrix": cm,
            "columns": classes,
            "predicted_absent_class": float(np.isin(pred, absent).mean()),
        },
    }


def main(argv: list[str] | None = None) -> None:
    from torch.utils.data import DataLoader

    from .dataset import CellDataset
    from .model import load_checkpoint, pick_device, transforms
    from .train import predict_logits

    raabin = Path("data/raabin/Raabin-WBC")
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on Raabin-WBC Test-A.")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--images", type=Path, default=raabin / "Test")
    ap.add_argument("--labels", type=Path, default=raabin / "Test.json")
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args(argv)

    internal = a.checkpoint.with_name("test_report.json")
    if not internal.is_file():
        ap.error(f"{internal} not found: run bloodcell-eval first (it fits the temperature)")
    temperature = json.loads(internal.read_text())["temperature_scaled"]["temperature"]

    device = pick_device()
    model, classes = load_checkpoint(a.checkpoint, device)
    samples = read_raabin(a.labels, a.images)
    dl = DataLoader(CellDataset(samples, classes, transforms(train=False)), a.batch_size)
    logits, _ = predict_logits(model, dl, device)

    result = score_external(logits, [s.label for s in samples], classes, temperature)
    result["dataset"] = {
        "name": "Raabin-WBC Test-A, double-labelled",
        "labels_sha256": file_sha256(a.labels),
    }
    result["provenance"] = provenance(None, device_name(device))
    out = a.checkpoint.with_name("external_raabin.json")
    out.write_text(json.dumps(result, indent=2))

    forced, free = result["forced_choice"]["temperature_scaled"], result["all_classes"]
    print(
        f"forced choice  acc {forced['accuracy']:.4f}  bal-acc {forced['balanced_accuracy']:.4f}"
        f"  ECE {result['forced_choice']['uncalibrated']['ece']:.4f} -> {forced['ece']:.4f}"
    )
    print(
        f"all classes    acc {free['accuracy']:.4f}  bal-acc {free['balanced_accuracy']:.4f}"
        f"  predicted a class Raabin lacks: {free['predicted_absent_class']:.1%}"
    )
    print(f"full report -> {out}")


if __name__ == "__main__":
    main()
