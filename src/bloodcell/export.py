"""Export a checkpoint to ONNX and check it against PyTorch on real test images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .model import INPUT_SIZE, MEAN, STD, build_model
from .provenance import provenance

# Largest allowed |PyTorch logit - ONNX logit|. On the whole test split the
# baselines differ by at most 1.7e-5 (#34); this leaves room for other CPUs.
TOLERANCE = 1e-4


def compare(model, session, batches) -> dict:
    """Largest logit difference and top-class agreement between PyTorch and ONNX Runtime."""
    import torch

    diff, agree, n = 0.0, 0, 0
    with torch.no_grad():
        for x in batches:
            ref = model(x).numpy()
            got = session.run(None, {"input": x.numpy()})[0]
            diff = max(diff, float(np.abs(ref - got).max()))
            agree += int((ref.argmax(axis=1) == got.argmax(axis=1)).sum())
            n += len(x)
    return {"images": n, "max_abs_logit_diff": diff, "top1_agreement": agree / n}


def test_batches(splits: Path, root: Path, classes: list[str], limit: int, batch_size: int = 64):
    """Preprocessed test images, as the model sees them in evaluation."""
    from torch.utils.data import DataLoader

    from .data import read_splits
    from .dataset import CellDataset
    from .model import transforms

    test = read_splits(splits, root)["test"]
    if limit:
        test = test[:limit]
    ds = CellDataset(test, classes, transforms(train=False))
    return (x for x, _ in DataLoader(ds, batch_size))


def main(argv: list[str] | None = None) -> None:
    import onnxruntime as ort
    import torch

    ap = argparse.ArgumentParser(description="Export checkpoint to ONNX.")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--root", type=Path, default=Path("data/PBC_dataset_normal_DIB"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    ap.add_argument(
        "--check-images", type=int, default=0, help="test images to check on (0: all of them)"
    )
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
        # one file to copy to the device; the weights are far below ONNX's 2 GB limit
        external_data=False,
    )

    # The temperature belongs to the trained model (ADR 0005): without it the
    # edge would report uncalibrated confidence.
    report = a.checkpoint.with_name("test_report.json")
    if report.is_file():
        temperature = json.loads(report.read_text())["temperature_scaled"]["temperature"]
    else:
        temperature = 1.0
        print(f"warning: {report} not found, so the sidecar has temperature 1 (uncalibrated)")

    # Sidecar metadata = the preprocessing contract the C++ side must follow.
    meta = {
        "classes": ckpt["classes"],
        "input_size": INPUT_SIZE,
        "mean": MEAN,
        "std": STD,
        "layout": "NCHW",
        "color": "RGB",
        "scale": "1/255",
        "temperature": temperature,
    }
    onnx_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if a.splits.is_file() and a.root.is_dir():
        batches = test_batches(a.splits, a.root, ckpt["classes"], a.check_images)
        source = "test split"
    else:
        # no dataset on this machine (CI): random inputs still catch a broken graph
        batches = iter([torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE)])
        source = "random inputs"
    check = compare(model, sess, batches)
    check["source"] = source
    out = a.checkpoint.with_name("export_check.json")
    record = {**check, "tolerance": TOLERANCE, "opset": a.opset}
    record["provenance"] = provenance(None, "cpu")
    out.write_text(json.dumps(record, indent=2))

    print(
        f"wrote {onnx_path} (+ .json); {source}, {check['images']} images: "
        f"max |torch - onnx| = {check['max_abs_logit_diff']:.2e}, "
        f"top class agrees on {check['top1_agreement']:.2%}"
    )
    if check["max_abs_logit_diff"] > TOLERANCE or check["top1_agreement"] < 1.0:
        raise SystemExit("ONNX output deviates from PyTorch; check opset/ops")


if __name__ == "__main__":
    main()
