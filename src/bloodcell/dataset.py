"""Torch Dataset over split samples.

Kept apart from ``data.py`` so splitting works without the torch extra.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from .data import Sample


class CellDataset(Dataset[tuple[Any, int]]):
    """Yields ``(transform(rgb_image), class_index)`` for each sample."""

    def __init__(
        self,
        samples: list[Sample],
        classes: list[str],
        transform: Callable[[Image.Image], Any] | None = None,
    ):
        self.samples = samples
        self.index = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        s = self.samples[i]
        img: Any = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.index[s.label]
