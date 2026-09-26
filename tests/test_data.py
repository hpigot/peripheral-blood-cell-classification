import random
from collections import Counter
from pathlib import Path

from PIL import Image

from bloodcell.data import clean, discover, read_splits, stratified_split, write_splits


def make_tree(root: Path, counts: dict[str, int]) -> None:
    for cls, n in counts.items():
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            # seeded noise, so every image is unique even after JPEG compression
            noise = random.Random(f"{cls}/{i}").randbytes(8 * 8 * 3)
            Image.frombytes("RGB", (8, 8), noise).save(d / f"{cls}_{i}.jpg")
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


def test_discover_skips_hidden_files(tmp_path):
    make_tree(tmp_path, {"neutrophil": 2})
    (tmp_path / "neutrophil" / ".DS_1.jpg").write_bytes(b"Bud1 (macOS folder metadata)")
    assert [s.path.name for s in discover(tmp_path)] == ["neutrophil_0.jpg", "neutrophil_1.jpg"]


def test_clean_keeps_first_copy_of_same_class_duplicates(tmp_path):
    make_tree(tmp_path, {"neutrophil": 2})
    d = tmp_path / "neutrophil"
    (d / "neutrophil_9.jpg").write_bytes((d / "neutrophil_0.jpg").read_bytes())
    kept, dropped = clean(discover(tmp_path))
    assert [s.path.name for s in kept] == ["neutrophil_0.jpg", "neutrophil_1.jpg"]
    assert [(x.path.name, x.reason) for x in dropped] == [
        ("neutrophil_9.jpg", "duplicate of neutrophil/neutrophil_0.jpg")
    ]


def test_clean_drops_every_copy_when_labels_conflict(tmp_path):
    make_tree(tmp_path, {"neutrophil": 2, "eosinophil": 1})
    src = tmp_path / "neutrophil" / "neutrophil_0.jpg"
    (tmp_path / "eosinophil" / "eo_x.jpg").write_bytes(src.read_bytes())
    kept, dropped = clean(discover(tmp_path))
    assert {s.path.name for s in kept} == {"neutrophil_1.jpg", "eosinophil_0.jpg"}
    assert sorted((x.path.name, x.reason) for x in dropped) == [
        ("eo_x.jpg", "same image labelled eosinophil and neutrophil"),
        ("neutrophil_0.jpg", "same image labelled eosinophil and neutrophil"),
    ]


def test_clean_drops_unreadable_images(tmp_path):
    make_tree(tmp_path, {"neutrophil": 1})
    (tmp_path / "neutrophil" / "broken.jpg").write_bytes(b"not a jpeg")
    kept, dropped = clean(discover(tmp_path))
    assert [s.path.name for s in kept] == ["neutrophil_0.jpg"]
    assert [(x.path.name, x.reason) for x in dropped] == [("broken.jpg", "unreadable image")]
