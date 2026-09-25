from collections import Counter
from pathlib import Path

from PIL import Image

from bloodcell.data import discover, read_splits, stratified_split, write_splits


def make_tree(root: Path, counts: dict[str, int]) -> None:
    for cls, n in counts.items():
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (8, 8), (i % 255, 0, 0)).save(d / f"{cls}_{i}.jpg")
    (root / "neutrophil" / "notes.txt").write_text("not an image")


def test_discover_uses_folder_names_and_skips_non_images(tmp_path):
    make_tree(tmp_path, {"neutrophil": 5, "basophil": 3})
    samples = discover(tmp_path)
    assert Counter(s.label for s in samples) == {"neutrophil": 5, "basophil": 3}


def test_split_is_stratified_deterministic_and_disjoint(tmp_path):
    make_tree(tmp_path, {"neutrophil": 60, "basophil": 20, "monocyte": 20})
    samples = discover(tmp_path)
    a = stratified_split(samples, seed=1)
    b = stratified_split(samples, seed=1)
    assert [s.path for s in a["test"]] == [s.path for s in b["test"]]
    paths = [{s.path for s in a[k]} for k in ("train", "val", "test")]
    assert not (paths[0] & paths[1]) and not (paths[0] & paths[2]) and not (paths[1] & paths[2])
    assert sum(map(len, paths)) == 100
    test_counts = Counter(s.label for s in a["test"])
    assert test_counts["neutrophil"] == 9 and test_counts["basophil"] == 3


def test_split_csv_roundtrip(tmp_path):
    root = tmp_path / "data"
    make_tree(root, {"neutrophil": 20, "basophil": 20})
    splits = stratified_split(discover(root), seed=0)
    write_splits(splits, tmp_path / "splits.csv", root)
    back = read_splits(tmp_path / "splits.csv", root)
    for k in splits:
        assert [s.path for s in back[k]] == [s.path for s in splits[k]]
