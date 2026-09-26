"""Where a result came from: code version, data split, environment (no torch needed)."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = ("torch", "torchvision", "numpy", "scikit-learn", "pillow")


def _git(*args: str) -> str | None:
    """Run git in the package's own checkout; None when it isn't one (e.g. a wheel)."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(splits: Path, device: str) -> dict:
    """Enough to rerun a result: commit, uncommitted changes, split file, versions."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        # tracked files changed but not committed: the SHA alone doesn't describe the code
        "git_dirty": None if status is None else bool(status),
        "splits_sha256": file_sha256(splits),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in PACKAGES},
        "device": device,
        "argv": sys.argv,
    }


def device_name(device) -> str:
    """Readable name of a torch device, for provenance."""
    import torch

    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(device)}"
    return f"{device.type}: {platform.processor() or platform.machine()}"
