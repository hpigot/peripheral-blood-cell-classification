"""Which part of the change of lab costs accuracy: colour, scale or neither?

Scores a checkpoint on Raabin-WBC Test-A under test-time corrections towards
PBC, without retraining: global colour (Reinhard), stain colours (Macenko),
cell size, and stains plus size. The colour corrections also run on the PBC
test split, as a control that they don't destroy what the model needs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .data import Sample, read_splits
from .external import read_raabin, score_external
from .metrics import report
from .provenance import device_name, file_sha256, provenance
from .stain import lab_stats, macenko, macenko_fit, nucleus_diameter, reinhard, rescale


class Correct:
    """Picklable PIL-to-PIL correction, so DataLoader workers can use it."""

    def __init__(
        self,
        lab: tuple[np.ndarray, np.ndarray] | None = None,
        stains: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        scale: float | None = None,
    ):
        self.lab, self.stains, self.scale = lab, stains, scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.scale is not None:
            img = rescale(img, self.scale)
        if self.lab is not None:
            img = reinhard(img, *self.lab)
        if self.stains is not None:
            img = macenko(img, *self.stains)
        return img


def _sample(samples: list[Sample], n: int, seed: int = 0) -> list[Sample]:
    return random.Random(seed).sample(samples, min(n, len(samples)))


def target_colour(samples: list[Sample], n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Mean of per-image CIELAB means and standard deviations over PBC training images."""
    stats = [lab_stats(Image.open(s.path)) for s in _sample(samples, n)]
    return np.mean([m for m, _ in stats], axis=0), np.mean([s for _, s in stats], axis=0)


def target_stains(
    samples: list[Sample], n: int = 200, pixels: int = 3000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Macenko stains, concentration range and background, pooled over PBC training images."""
    rng = np.random.default_rng(0)
    pooled = []
    for s in _sample(samples, n):
        rgb = np.asarray(Image.open(s.path).convert("RGB")).reshape(-1, 3)
        pooled.append(rgb[rng.choice(len(rgb), pixels, replace=False)])
    return macenko_fit(np.concatenate(pooled))


def median_nucleus(samples: list[Sample], n: int = 400) -> tuple[float, int]:
    """Median lymphocyte nucleus diameter and how many images it was measured on."""
    lymph = [s for s in samples if s.label == "lymphocyte"]
    d = np.array([nucleus_diameter(Image.open(s.path)) for s in _sample(lymph, n)])
    return float(np.nanmedian(d)), int((~np.isnan(d)).sum())


def main(argv: list[str] | None = None) -> None:
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    from .dataset import CellDataset
    from .model import load_checkpoint, pick_device, transforms
    from .train import predict_logits

    raabin = Path("data/raabin/Raabin-WBC")
    ap = argparse.ArgumentParser(description="Colour vs scale on Raabin-WBC Test-A.")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--root", type=Path, default=Path("data/PBC_dataset_normal_DIB"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--images", type=Path, default=raabin / "Test")
    ap.add_argument("--labels", type=Path, default=raabin / "Test.json")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    internal = a.checkpoint.with_name("test_report.json")
    if not internal.is_file():
        ap.error(f"{internal} not found: run bloodcell-eval first (it fits the temperature)")
    temperature = json.loads(internal.read_text())["temperature_scaled"]["temperature"]

    splits = read_splits(a.splits, a.root)
    ext = read_raabin(a.labels, a.images)
    lab = target_colour(splits["train"])
    stains = target_stains(splits["train"])
    pbc_d, pbc_n = median_nucleus(splits["train"])
    ext_d, ext_n = median_nucleus(ext)
    scale = pbc_d / ext_d
    print(f"nucleus diameter PBC {pbc_d:.1f} px, Raabin {ext_d:.1f} px -> scale {scale:.3f}")

    device = pick_device()
    model, classes = load_checkpoint(a.checkpoint, device)

    def logits(samples: list[Sample], fix: Correct) -> np.ndarray:
        tf = T.Compose([fix, transforms(train=False)])
        dl = DataLoader(CellDataset(samples, classes, tf), a.batch_size, num_workers=a.workers)
        return predict_logits(model, dl, device)[0]

    conditions = {
        "as_is": Correct(),
        "reinhard": Correct(lab=lab),
        "macenko": Correct(stains=stains),
        "scale": Correct(scale=scale),
        "macenko_and_scale": Correct(stains=stains, scale=scale),
    }
    ext_labels = [s.label for s in ext]
    result: dict = {
        "temperature": temperature,
        "reinhard_target": {"lab_mean": lab[0].tolist(), "lab_std": lab[1].tolist()},
        "macenko_target": {
            "stains": stains[0].tolist(),
            "max_concentration": stains[1].tolist(),
            "background": stains[2].tolist(),
        },
        "nucleus_diameter_px": {
            "pbc_train": pbc_d,
            "pbc_measured": pbc_n,
            "raabin": ext_d,
            "raabin_measured": ext_n,
        },
        "scale": scale,
        "raabin": {},
    }
    for name, fix in conditions.items():
        r = score_external(logits(ext, fix), ext_labels, classes, temperature)
        result["raabin"][name] = r
        print(
            f"raabin {name:17s} all-8 bal-acc {r['all_classes']['balanced_accuracy']:.4f}  "
            f"forced-5 {r['forced_choice']['temperature_scaled']['balanced_accuracy']:.4f}  "
            f"ECE {r['forced_choice']['temperature_scaled']['ece']:.4f}"
        )

    test = splits["test"]
    y = np.array([classes.index(s.label) for s in test])
    result["pbc_test"] = {}
    for name in ("as_is", "reinhard", "macenko"):
        r = report(logits(test, conditions[name]), y, classes, temperature)
        result["pbc_test"][name] = r
        print(f"pbc    {name:17s} bal-acc {r['balanced_accuracy']:.4f}  ECE {r['ece']:.4f}")

    result["dataset"] = {"labels_sha256": file_sha256(a.labels)}
    result["provenance"] = provenance(a.splits, device_name(device))
    out = a.checkpoint.with_name("shift_raabin.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"full report -> {out}")


if __name__ == "__main__":
    main()
