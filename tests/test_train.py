import json
import random
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image  # noqa: E402

from bloodcell import evaluate, train  # noqa: E402
from bloodcell.data import discover, stratified_split, write_splits  # noqa: E402


def make_dataset(root: Path, per_class: int = 11) -> None:
    for cls in ("basophil", "neutrophil"):
        (root / cls).mkdir(parents=True)
        for i in range(per_class):
            noise = random.Random(f"{cls}/{i}").randbytes(32 * 32 * 3)
            Image.frombytes("RGB", (32, 32), noise).save(root / cls / f"{i}.png")


def test_train_then_evaluate_writes_run_artifacts(tmp_path):
    root = tmp_path / "images"
    make_dataset(root)
    splits_csv = tmp_path / "splits.csv"
    write_splits(stratified_split(discover(root), seed=0), splits_csv, root)
    common = ["--root", str(root), "--splits", str(splits_csv)]

    train.main(
        [*common, "--arch", "mobilenet_v3_small", "--no-pretrained", "--epochs", "1",
         "--batch-size", "8", "--workers", "0", "--out", str(tmp_path / "runs")]
    )  # fmt: skip
    (run_dir,) = (tmp_path / "runs").iterdir()
    history = json.loads((run_dir / "history.json").read_text())
    assert [h["epoch"] for h in history] == [1]
    assert (run_dir / "best.pt").is_file()
    config = json.loads((run_dir / "config.json").read_text())
    assert config["arch"] == "mobilenet_v3_small"
    assert config["provenance"]["device"].startswith(("cpu", "cuda", "mps"))

    evaluate.main([str(run_dir / "best.pt"), *common])
    report = json.loads((run_dir / "test_report.json").read_text())
    assert set(report) == {"uncalibrated", "temperature_scaled", "provenance"}
    assert report["uncalibrated"]["n"] == 4
    assert report["provenance"]["splits_sha256"] == config["provenance"]["splits_sha256"]
