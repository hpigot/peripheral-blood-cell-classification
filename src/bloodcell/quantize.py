"""Quantize an exported ONNX model to INT8 and compare it with FP32 on the test split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .data import class_names, read_splits
from .metrics import report, softmax
from .provenance import provenance

CALIBRATION_IMAGES = 500
# Percentile clips each activation range at this percentile of the calibration
# values; 99.9 scored best on validation for MobileNet (ADR 0006).
PERCENTILE = 99.9


def batches(samples, classes: list[str], batch_size: int = 64):
    """(images, labels) as numpy, preprocessed as in evaluation."""
    from torch.utils.data import DataLoader

    from .dataset import CellDataset
    from .model import transforms

    ds = CellDataset(samples, classes, transforms(train=False))
    for x, y in DataLoader(ds, batch_size):
        yield x.numpy(), y.numpy()


class Calibration:
    """ONNX Runtime CalibrationDataReader over preprocessed training images."""

    def __init__(self, samples, classes: list[str]):
        # equal batches: the percentile calibrator can't stack a short last one
        self._it = (x for x, _ in batches(samples, classes, batch_size=25))

    def get_next(self) -> dict | None:
        x = next(self._it, None)
        return None if x is None else {"input": x}


def quantize(fp32: Path, int8: Path, calibration: Calibration, method: str) -> None:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if method == "percentile":
        options = {
            "calibrate_method": CalibrationMethod.Percentile,
            "extra_options": {"CalibPercentile": PERCENTILE},
        }
    else:
        options = {"calibrate_method": CalibrationMethod.MinMax}
    with tempfile.TemporaryDirectory() as tmp:
        # shape inference and graph fusions first, as ONNX Runtime recommends
        prep = Path(tmp) / "prep.onnx"
        quant_pre_process(str(fp32), str(prep))
        quantize_static(
            prep,
            int8,
            calibration,  # a duck-typed CalibrationDataReader
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            **options,
        )


def logits(model: Path, data) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    out, ys = [], []
    for x, y in data:
        out.append(sess.run(None, {"input": x})[0])
        ys.append(y)
    return np.concatenate(out), np.concatenate(ys)


def write_predictions(path: Path, samples, root: Path, classes, probs: np.ndarray) -> None:
    """Same format as bloodcell-eval's test_predictions.csv, for bloodcell-score --against."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "predicted", "confidence"])
        for s, row in zip(samples, probs, strict=True):
            top = int(row.argmax())
            w.writerow(
                [s.path.relative_to(root).as_posix(), s.label, classes[top], f"{row[top]:.4f}"]
            )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Quantize best.onnx to INT8 and compare with FP32.")
    ap.add_argument("onnx", type=Path, help="best.onnx from bloodcell-export")
    ap.add_argument("--root", type=Path, default=Path("data/PBC_dataset_normal_DIB"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--calibration-images", type=int, default=CALIBRATION_IMAGES)
    ap.add_argument(
        "--calibration",
        choices=["percentile", "minmax"],
        default="percentile",
        help="activation ranges: clip at a percentile, or the full min-max range "
        "(percentile keeps every activation in memory; too much for EfficientNet on 16 GB)",
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    splits = read_splits(a.splits, a.root)
    classes = class_names(splits)
    sidecar = json.loads(a.onnx.with_suffix(".json").read_text())
    if sidecar["classes"] != classes:
        raise SystemExit("the sidecar's classes don't match the split's")

    # a fixed random sample of training images; never the test split
    calib = random.Random(a.seed).sample(splits["train"], a.calibration_images)
    int8 = a.onnx.with_suffix(".int8.onnx")
    quantize(a.onnx, int8, Calibration(calib, classes), a.calibration)
    # the C++ program reads <model>.json: same contract, same temperature
    shutil.copyfile(a.onnx.with_suffix(".json"), int8.with_suffix(".json"))

    test = splits["test"]
    t = sidecar["temperature"]
    result: dict = {
        "calibration": a.calibration,
        "percentile": PERCENTILE if a.calibration == "percentile" else None,
        "calibration_images": len(calib),
        "seed": a.seed,
        "temperature": t,
    }
    preds = {}
    for name, path in (("fp32", a.onnx), ("int8", int8)):
        z, y = logits(path, batches(test, classes))
        result[name] = {"file_bytes": path.stat().st_size, **report(z, y, classes, t)}
        preds[name] = z.argmax(axis=1)
        if name == "int8":
            out = a.onnx.with_name("test_predictions.int8.csv")
            write_predictions(out, test, a.root, classes, softmax(z, t))
    result["top1_agreement"] = float((preds["fp32"] == preds["int8"]).mean())
    result["provenance"] = provenance(a.splits, "cpu")
    report_path = a.onnx.with_name("quantize_report.json")
    report_path.write_text(json.dumps(result, indent=2))

    for name in ("fp32", "int8"):
        r = result[name]
        print(
            f"{name}  {r['file_bytes'] / 1e6:5.1f} MB  bal-acc {r['balanced_accuracy']:.4f}  "
            f"ECE {r['ece']:.4f}"
        )
    print(f"top class agrees on {result['top1_agreement']:.2%}; full report -> {report_path}")


if __name__ == "__main__":
    main()
