import pytest

pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from bloodcell.data import Sample  # noqa: E402
from bloodcell.dataset import CellDataset  # noqa: E402


def test_items_are_rgb_images_with_class_indices(tmp_path):
    Image.new("L", (8, 8)).save(tmp_path / "a.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    ds = CellDataset(
        [Sample(tmp_path / "a.png", "monocyte"), Sample(tmp_path / "b.png", "basophil")],
        classes=["basophil", "monocyte"],
    )
    assert len(ds) == 2
    img, label = ds[0]
    assert img.mode == "RGB" and label == 1
    assert ds[1][1] == 0
