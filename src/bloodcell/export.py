"""Export a checkpoint to ONNX and verify it numerically against PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .model import INPUT_SIZE, MEAN, STD, build_model


def main(argv: list[str] | None = None) -> None:
    import onnxruntime as ort
    import torch

    ap = argparse.ArgumentParser(description="Export checkpoint to ONNX.")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--opset", type=int, default=18)
    a = ap.parse_args(argv)

    ckpt = torch.load(a.checkpoint, map_location="cpu")
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    onnx_path = a.checkpoint.with_suffix(".onnx")
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        opset_version=a.opset,
        dynamo=True,
        # its progress messages include emoji, which crash a redirected Windows console
        verbose=False,
    )
    # Sidecar metadata = the preprocessing contract the C++ side must follow.
    meta = {
        "classes": ckpt["classes"],
        "input_size": INPUT_SIZE,
        "mean": MEAN,
        "std": STD,
        "layout": "NCHW",
        "color": "RGB",
        "scale": "1/255",
    }
    onnx_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    x = torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"wrote {onnx_path} (+ .json); max |torch - onnx| = {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("ONNX output deviates from PyTorch; check opset/ops")


if __name__ == "__main__":
    main()
