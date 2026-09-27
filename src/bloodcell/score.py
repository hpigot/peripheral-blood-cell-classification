"""Score the C++ program's predictions with the same metrics as bloodcell-eval."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import class_names, read_splits
from .metrics import label_report

# Largest allowed |C++ confidence - Python confidence|. With the same pixels the
# two differ by float rounding only (#35).
TOLERANCE = 1e-3


def _key(path: str) -> str:
    # <class>/<file>, as in splits.csv: the images can sit anywhere on the device
    return "/".join(Path(path.replace("\\", "/")).parts[-2:])


def read_preds(path: Path) -> dict[str, tuple[str, float]]:
    """``path<TAB>label<TAB>confidence`` lines, keyed by <class>/<file>."""
    out = {}
    for line in path.read_text().splitlines():
        p, label, conf = line.split("\t")
        out[_key(p)] = (label, float(conf))
    return out


def parity(preds: dict[str, tuple[str, float]], python_csv: Path) -> dict:
    """Agreement with bloodcell-eval's test_predictions.csv on the same images."""
    with open(python_csv, newline="") as f:
        ref = {r["path"]: (r["predicted"], float(r["confidence"])) for r in csv.DictReader(f)}
    keys = sorted(ref.keys() & preds.keys())
    return {
        "images": len(keys),
        "label_agreement": sum(preds[k][0] == ref[k][0] for k in keys) / len(keys),
        # the CSV rounds to 4 decimals
        "max_abs_confidence_diff": max(abs(preds[k][1] - ref[k][1]) for k in keys),
        "tolerance": TOLERANCE,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score bloodcell_infer's preds.tsv against a split.")
    ap.add_argument("preds", type=Path)
    ap.add_argument("--root", type=Path, default=Path("data/PBC_dataset_normal_DIB"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--against", type=Path, help="test_predictions.csv from bloodcell-eval, to check parity"
    )
    a = ap.parse_args(argv)

    splits = read_splits(a.splits, a.root)
    classes = class_names(splits)
    samples = splits[a.split]
    preds = read_preds(a.preds)
    keys = [s.path.relative_to(a.root).as_posix() for s in samples]
    missing = [k for k in keys if k not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} {a.split} images have no prediction, e.g. {missing[0]}")

    y = np.array([classes.index(s.label) for s in samples])
    pred = np.array([classes.index(preds[k][0]) for k in keys])
    conf = np.array([preds[k][1] for k in keys])
    result = {"split": a.split, **label_report(y, pred, conf, classes)}
    if a.against:
        result["parity"] = parity(preds, a.against)
    out = a.preds.with_name(a.preds.stem + "_report.json")
    out.write_text(json.dumps(result, indent=2))

    print(
        f"{a.split}: {result['n']} images  acc {result['accuracy']:.4f}  "
        f"bal-acc {result['balanced_accuracy']:.4f}  ECE {result['ece']:.4f}  -> {out}"
    )
    if a.against:
        p = result["parity"]
        print(
            f"vs Python: label agrees on {p['label_agreement']:.2%} of {p['images']}, "
            f"max |confidence diff| {p['max_abs_confidence_diff']:.1e}"
        )
        if p["label_agreement"] < 1.0 or p["max_abs_confidence_diff"] > TOLERANCE:
            raise SystemExit("C++ and Python disagree; check the preprocessing contract")


if __name__ == "__main__":
    main()
