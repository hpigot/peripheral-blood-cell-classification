import json
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from PIL import Image  # noqa: E402

from bloodcell import export, quantize  # noqa: E402
from bloodcell.data import discover, stratified_split, write_splits  # noqa: E402
from bloodcell.model import build_model  # noqa: E402

CLASSES = ["basophil", "neutrophil"]


def make_dataset(root):
    for cls in CLASSES:
        (root / cls).mkdir(parents=True)
        for i in range(11):
            noise = random.Random(f"{cls}/{i}").randbytes(32 * 32 * 3)
            Image.frombytes("RGB", (32, 32), noise).save(root / cls / f"{i}.png")


def test_quantize_writes_int8_model_sidecar_and_report(tmp_path):
    root = tmp_path / "images"
    make_dataset(root)
    splits = tmp_path / "splits.csv"
    write_splits(stratified_split(discover(root), seed=0), splits, root)
    model = build_model("mobilenet_v3_small", len(CLASSES), pretrained=False)
    ckpt = tmp_path / "best.pt"
    state = {"arch": "mobilenet_v3_small", "classes": CLASSES, "state_dict": model.state_dict()}
    torch.save(state, ckpt)
    common = ["--root", str(root), "--splits", str(splits)]
    export.main([str(ckpt), *common])

    quantize.main(
        [str(tmp_path / "best.onnx"), *common, "--calibration", "minmax",
         "--calibration-images", "4"]
    )  # fmt: skip

    assert (tmp_path / "best.int8.onnx").stat().st_size < (tmp_path / "best.onnx").stat().st_size
    assert json.loads((tmp_path / "best.int8.json").read_text())["classes"] == CLASSES
    result = json.loads((tmp_path / "quantize_report.json").read_text())
    assert result["calibration"] == "minmax"
    assert result["fp32"]["n"] == result["int8"]["n"] == 4
    assert 0.0 <= result["top1_agreement"] <= 1.0
    rows = (tmp_path / "test_predictions.int8.csv").read_text().splitlines()
    assert rows[0] == "path,label,predicted,confidence"
    assert len(rows) == 1 + 4
