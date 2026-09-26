"""Fine-tune a pretrained CNN on the split CSV; keep the best-val-balanced-accuracy checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np

from .data import class_names, read_splits
from .dataset import CellDataset
from .metrics import report
from .model import ARCHS, build_model, pick_device, transforms
from .provenance import device_name, provenance


def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    out, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            out.append(model(x.to(device)).float().cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(out), np.concatenate(ys)


def main(argv: list[str] | None = None) -> None:
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description="Train a blood cell classifier.")
    ap.add_argument("--root", type=Path, default=Path("data/PBC_dataset_normal_DIB"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--arch", choices=ARCHS, default="mobilenet_v3_small")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="start from random weights instead of downloading ImageNet ones",
    )
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    device = pick_device()
    splits = read_splits(a.splits, a.root)
    classes = class_names(splits)
    train_ds = CellDataset(splits["train"], classes, transforms(train=True))
    val_ds = CellDataset(splits["val"], classes, transforms(train=False))
    # Keep workers alive between epochs: on Windows each new worker re-imports
    # torch, which left the GPU idle at the start of every pass.
    loader = partial(
        DataLoader,
        batch_size=a.batch_size,
        num_workers=a.workers,
        persistent_workers=a.workers > 0,
        pin_memory=device.type == "cuda",
    )
    # Shuffling and each worker's augmentation seed come from their own generator,
    # so the data a run sees doesn't depend on how model setup used the global RNG.
    train_dl = loader(train_ds, shuffle=True, generator=torch.Generator().manual_seed(a.seed))
    val_dl = loader(val_ds, shuffle=False)

    model = build_model(a.arch, len(classes), pretrained=a.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=len(train_dl)
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    run_dir = a.out / f"{a.arch}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config: dict = {k: str(v) for k, v in vars(a).items()}
    config["provenance"] = provenance(a.splits, device_name(device))
    if config["provenance"]["git_dirty"]:
        print("warning: uncommitted changes, so the git SHA doesn't fully describe this run")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    best, history = -1.0, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
        logits, ys = predict_logits(model, val_dl, device)
        r = report(logits, ys, classes)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(train_ds),
                **{k: r[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "ece")},
                "seconds": round(time.time() - t0, 1),  # training plus validation
            }
        )
        print(
            f"epoch {epoch:2d}  loss {history[-1]['train_loss']:.4f}  "
            f"val bal-acc {r['balanced_accuracy']:.4f}  ece {r['ece']:.4f}  "
            f"({history[-1]['seconds']:.0f}s)"
        )
        # rewritten every epoch so an interrupted run keeps its curve
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        if r["balanced_accuracy"] > best:
            best = r["balanced_accuracy"]
            torch.save(
                {"arch": a.arch, "classes": classes, "state_dict": model.state_dict()},
                run_dir / "best.pt",
            )

    print(f"best val balanced accuracy {best:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
