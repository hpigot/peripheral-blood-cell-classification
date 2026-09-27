import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from bloodcell.export import main  # noqa: E402
from bloodcell.model import INPUT_SIZE, build_model  # noqa: E402

CLASSES = ["basophil", "neutrophil"]


def _checkpoint(tmp_path):
    model = build_model("mobilenet_v3_small", len(CLASSES), pretrained=False)
    ckpt = tmp_path / "best.pt"
    torch.save(
        {"arch": "mobilenet_v3_small", "classes": CLASSES, "state_dict": model.state_dict()}, ckpt
    )
    return ckpt


def _export(tmp_path, ckpt):
    # no dataset here, so the check falls back to random inputs
    missing = tmp_path / "missing"
    main([str(ckpt), "--root", str(missing), "--splits", str(missing / "splits.csv")])


def test_export_writes_onnx_sidecar_and_check(tmp_path):
    _export(tmp_path, _checkpoint(tmp_path))  # raises SystemExit if ONNX and torch disagree

    assert (tmp_path / "best.onnx").is_file()
    meta = json.loads((tmp_path / "best.json").read_text())
    assert meta["classes"] == CLASSES
    assert meta["input_size"] == INPUT_SIZE
    assert meta["temperature"] == 1.0  # no test_report.json yet
    check = json.loads((tmp_path / "export_check.json").read_text())
    assert check["source"] == "random inputs"
    assert check["top1_agreement"] == 1.0
    assert check["max_abs_logit_diff"] <= check["tolerance"]


def test_sidecar_carries_the_fitted_temperature(tmp_path):
    ckpt = _checkpoint(tmp_path)
    report = {"temperature_scaled": {"temperature": 0.61}}
    (tmp_path / "test_report.json").write_text(json.dumps(report))
    _export(tmp_path, ckpt)
    assert json.loads((tmp_path / "best.json").read_text())["temperature"] == 0.61
