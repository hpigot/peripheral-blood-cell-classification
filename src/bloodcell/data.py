"""Dataset discovery and reproducible stratified splits (no torch needed).

Expects an ImageFolder-style tree: ``root/<class_name>/*.jpg`` (the PBC
dataset unzips this way). Splits are written to a CSV so every run, and the
edge evaluation, uses exactly the same test images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


@dataclass(frozen=True)
class Dropped:
    path: Path
    label: str
    reason: str


def discover(root: Path) -> list[Sample]:
    """List every image under ``root/<class>/``; class = parent folder name.

    Hidden files are skipped: the PBC zip contains a macOS ``.DS_Store`` file
    saved with a ``.jpg`` name.
    """
    root = Path(root)
    samples = [
        Sample(p, p.parent.name)
        for class_dir in sorted(d for d in root.iterdir() if d.is_dir())
        for p in sorted(class_dir.rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS and p.parent == class_dir and not p.name.startswith(".")
    ]
    if not samples:
        raise FileNotFoundError(f"no images found under {root}/<class>/")
    return samples


def _short(p: Path) -> str:
    return f"{p.parent.name}/{p.name}"


def clean(samples: list[Sample]) -> tuple[list[Sample], list[Dropped]]:
    """Drop unreadable images and byte-identical duplicates (see ADR 0004).

    The first copy of a same-class duplicate is kept. When copies carry
    different labels, the true label is unknown, so every copy is dropped.
    """
    drop: dict[Path, str] = {}
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        try:
            with Image.open(s.path) as im:
                im.verify()
        except Exception:
            drop[s.path] = "unreadable image"
            continue
        groups[hashlib.sha256(s.path.read_bytes()).hexdigest()].append(s)

    for copies in groups.values():
        labels = sorted({c.label for c in copies})
        if len(labels) > 1:
            for c in copies:
                drop[c.path] = f"same image labelled {' and '.join(labels)}"
        else:
            for c in copies[1:]:
                drop[c.path] = f"duplicate of {_short(copies[0].path)}"

    kept = [s for s in samples if s.path not in drop]
    dropped = [Dropped(s.path, s.label, drop[s.path]) for s in samples if s.path in drop]
    return kept, dropped


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
    kept, dropped = clean(discover(a.root))
    for d in dropped:
        print(f"dropped {_short(d.path)}: {d.reason}")
    splits = stratified_split(kept, a.val, a.test, a.seed)
    write_splits(splits, a.out, a.root)
    for name in SPLITS:
        print(f"{name:5s} {len(splits[name]):6d}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
