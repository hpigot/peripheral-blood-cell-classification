import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from bloodcell.export import main  # noqa: E402
from bloodcell.model import INPUT_SIZE, build_model  # noqa: E402


def test_export_writes_onnx_and_preprocessing_sidecar(tmp_path):
    classes = ["basophil", "neutrophil"]
    model = build_model("mobilenet_v3_small", len(classes), pretrained=False)
    ckpt = tmp_path / "best.pt"
    torch.save(
        {"arch": "mobilenet_v3_small", "classes": classes, "state_dict": model.state_dict()}, ckpt
    )

    main([str(ckpt)])  # raises SystemExit if ONNX and torch outputs disagree

    assert (tmp_path / "best.onnx").is_file()
    meta = json.loads((tmp_path / "best.json").read_text())
    assert meta["classes"] == classes
    assert meta["input_size"] == INPUT_SIZE
