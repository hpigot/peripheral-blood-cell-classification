"""Dataset discovery and reproducible stratified splits (no torch needed).

Expects an ImageFolder-style tree: ``root/<class_name>/*.jpg`` (the PBC
dataset unzips this way). Splits are written to a CSV so every run, and the
edge evaluation, uses exactly the same test images.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


def discover(root: Path) -> list[Sample]:
    """List every image under ``root/<class>/``; class = parent folder name."""
    root = Path(root)
    samples = [
        Sample(p, p.parent.name)
        for class_dir in sorted(d for d in root.iterdir() if d.is_dir())
        for p in sorted(class_dir.rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS and p.parent == class_dir
    ]
    if not samples:
        raise FileNotFoundError(f"no images found under {root}/<class>/")
    return samples


def stratified_split(
    samples: list[Sample], val: float = 0.15, test: float = 0.15, seed: int = 0
) -> dict[str, list[Sample]]:
    """Stratified train/val/test split, deterministic for a given seed."""
    labels = [s.label for s in samples]
    rest, test_s = train_test_split(samples, test_size=test, stratify=labels, random_state=seed)
    rest_labels = [s.label for s in rest]
    train_s, val_s = train_test_split(
        rest, test_size=val / (1 - test), stratify=rest_labels, random_state=seed
    )
    return {"train": train_s, "val": val_s, "test": test_s}


def write_splits(splits: dict[str, list[Sample]], out: Path, root: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "label"])
        for name in SPLITS:
            for s in splits[name]:
                w.writerow([name, s.path.relative_to(root).as_posix(), s.label])


def read_splits(csv_path: Path, root: Path) -> dict[str, list[Sample]]:
    splits: dict[str, list[Sample]] = {name: [] for name in SPLITS}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            splits[row["split"]].append(Sample(Path(root) / row["path"], row["label"]))
    return splits


def class_names(splits: dict[str, list[Sample]]) -> list[str]:
    return sorted({s.label for s in splits["train"]})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write a reproducible stratified split CSV.")
    ap.add_argument("--root", type=Path, default=Path("data/PBC_dataset_normal_DIB"))
    ap.add_argument("--out", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    splits = stratified_split(discover(a.root), a.val, a.test, a.seed)
    write_splits(splits, a.out, a.root)
    for name in SPLITS:
        print(f"{name:5s} {len(splits[name]):6d}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
