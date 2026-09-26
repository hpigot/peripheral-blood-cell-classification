"""Evaluate a checkpoint on the test split; fit temperature on val, report before/after."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data import read_splits
from .dataset import CellDataset
from .metrics import fit_temperature, report, softmax
from .model import load_checkpoint, pick_device, transforms
from .provenance import device_name, provenance
from .train import predict_logits


def main(argv: list[str] | None = None) -> None:
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--root", type=Path, default=Path("data/PBC_dataset_normal_DIB"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args(argv)

    device = pick_device()
    model, classes = load_checkpoint(a.checkpoint, device)

    splits = read_splits(a.splits, a.root)
    tf = transforms(train=False)

    def logits_for(name):
        dl = DataLoader(CellDataset(splits[name], classes, tf), a.batch_size, shuffle=False)
        return predict_logits(model, dl, device)

    val_logits, val_y = logits_for("val")
    t = fit_temperature(val_logits, val_y)
    test_logits, test_y = logits_for("test")
    result = {
        "uncalibrated": report(test_logits, test_y, classes),
        "temperature_scaled": report(test_logits, test_y, classes, temperature=t),
        # the evaluating machine and code can differ from the training ones
        "provenance": provenance(a.splits, device_name(device)),
    }
    out = a.checkpoint.with_name("test_report.json")
    out.write_text(json.dumps(result, indent=2))

    # one row per test image, for error analysis (bloodcell-plot draws the worst ones)
    probs = softmax(test_logits, t)
    with open(a.checkpoint.with_name("test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "predicted", "confidence"])
        for s, row in zip(splits["test"], probs, strict=True):
            top = int(row.argmax())
            w.writerow(
                [s.path.relative_to(a.root).as_posix(), s.label, classes[top], f"{row[top]:.4f}"]
            )
    for k in ("uncalibrated", "temperature_scaled"):
        r = result[k]
        print(
            f"{k:20s} acc {r['accuracy']:.4f}  bal-acc {r['balanced_accuracy']:.4f}  "
            f"macro-F1 {r['macro_f1']:.4f}  ECE {r['ece']:.4f}"
        )
    print(f"temperature {t:.3f}; full report -> {out}")


if __name__ == "__main__":
    main()
