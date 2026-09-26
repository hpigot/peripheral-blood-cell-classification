"""Result figures in one house style (see docs/figures/STYLE.md).

Every figure is drawn twice, light and dark, so the README can swap them with
GitHub's ``<picture>`` element. The drawing functions take plain data (report
dicts, history lists) and never touch torch.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import rc_context
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

MODES = ("light", "dark")


@dataclass(frozen=True)
class Palette:
    surface: str
    ink: str  # titles, values
    ink2: str  # axis labels, legend text
    muted: str  # ticks, reference marks, the "before" condition
    grid: str
    axis: str
    series: tuple[str, str, str]  # categorical slots; validated for CVD as a set of 3
    ramp: tuple[str, ...]  # sequential blue, low -> high


# Reference palette from the data-viz method; slots 1-3 pass the colour-vision
# checks on all pairs in both modes (docs/figures/STYLE.md).
_BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
PALETTES = {
    "light": Palette(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series=("#2a78d6", "#eb6834", "#1baf7a"),
        ramp=("#fcfcfb", *_BLUE_RAMP),
    ),
    "dark": Palette(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series=("#3987e5", "#d95926", "#199e70"),
        ramp=("#1a1a19", *reversed(_BLUE_RAMP[:-1])),
    ),
}

# Colour follows the model, never its position in a particular chart.
MODEL_SLOTS = {"mobilenet_v3_small": 0, "efficientnet_b0": 1, "resnet50": 2}

CLASS_NAMES = {
    "basophil": "Basophil",
    "eosinophil": "Eosinophil",
    "erythroblast": "Erythroblast",
    "ig": "Immature granulocyte",
    "lymphocyte": "Lymphocyte",
    "monocyte": "Monocyte",
    "neutrophil": "Neutrophil",
    "platelet": "Platelet",
}

LINE_W = 2.0
DOT_SIZE = 64  # points^2, about 8 px across at 100 dpi
DPI = 200


def model_color(arch: str, mode: str) -> str:
    """The model's fixed slot; an unknown name falls back to slot 1."""
    return PALETTES[mode].series[MODEL_SLOTS.get(arch, 0)]


def _rc(mode: str) -> dict:
    p = PALETTES[mode]
    rc = {**sns.axes_style("ticks"), **sns.plotting_context("notebook", font_scale=0.9)}
    rc.update(
        {
            "figure.facecolor": p.surface,
            "axes.facecolor": p.surface,
            "savefig.facecolor": p.surface,
            "font.family": "sans-serif",
            # ships with matplotlib, so figures render identically on every machine
            "font.sans-serif": ["DejaVu Sans"],
            "text.color": p.ink,
            "axes.labelcolor": p.ink2,
            "axes.titlecolor": p.ink,
            "axes.titlelocation": "left",
            "axes.titleweight": "normal",
            "axes.titlesize": 12,
            "axes.titlepad": 10,
            "axes.edgecolor": p.axis,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": p.grid,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",
            "xtick.color": p.axis,
            "ytick.color": p.axis,
            "xtick.labelcolor": p.muted,
            "ytick.labelcolor": p.muted,
            "legend.frameon": False,
            "legend.labelcolor": p.ink2,
            "lines.linewidth": LINE_W,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )
    return rc


@contextmanager
def style(mode: str) -> Iterator[Palette]:
    """Apply the house style for one mode without touching global rcParams."""
    with rc_context(_rc(mode)):
        yield PALETTES[mode]


def _ink_on(fill: np.ndarray | tuple[float, ...]) -> str:
    """Black or white text, whichever reads on this fill."""
    r, g, b = fill[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if luminance > 0.5 else "#ffffff"


# --- figures -------------------------------------------------------------------------


def confusion_matrix(cm: list[list[int]], classes: list[str], mode: str) -> Figure:
    """Row-normalized confusion matrix: each row shows where one true class went."""
    counts = np.asarray(cm, dtype=float)
    share = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    names = [CLASS_NAMES.get(c, c) for c in classes]
    with style(mode) as p:
        fig = Figure(figsize=(7.2, 6.0), layout="constrained")
        ax = fig.subplots()
        cmap = LinearSegmentedColormap.from_list("house_seq", p.ramp)
        sns.heatmap(
            share,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            square=True,
            linewidths=2,
            linecolor=p.surface,  # the 2px surface gap between cells
            cbar_kws={"format": PercentFormatter(1, decimals=0), "shrink": 0.75},
        )
        for (i, j), v in np.ndenumerate(share):
            if v >= 0.005:  # label only what is visible; the table view has the rest
                fill = cmap(v)
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{v:.0%}" if v >= 0.1 else f"{v:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_ink_on(fill),
                )
        ax.set_xticklabels(names, rotation=40, ha="right")
        ax.set_yticklabels(names, rotation=0)
        ax.tick_params(length=0, labelcolor=p.ink2)  # class names carry identity
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Where each class goes (test set, share of true class)")
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(length=0, labelcolor=p.muted)
    return fig


def per_class_recall(recalls: dict[str, dict[str, float]], mode: str) -> Figure:
    """Dot plot of recall per class, one dot per model (at most 3 models)."""
    if len(recalls) > 3:
        raise ValueError("at most 3 models per figure; facet or split the comparison")
    first = next(iter(recalls.values()))
    classes = sorted(first, key=first.__getitem__)  # weakest class at the bottom
    with style(mode) as p:
        fig = Figure(figsize=(7.2, 4.2), layout="constrained")
        ax = fig.subplots()
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        for arch, rec in recalls.items():
            ax.scatter(
                [rec[c] for c in classes],
                range(len(classes)),
                s=DOT_SIZE,
                color=model_color(arch, mode),
                edgecolors=p.surface,
                linewidths=2,
                zorder=3,
                label=arch,
            )
        lo = min(min(r.values()) for r in recalls.values())
        ax.set_xlim(max(0.0, lo - 0.03), 1.005)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_yticks(range(len(classes)), [CLASS_NAMES.get(c, c) for c in classes])
        ax.tick_params(axis="y", length=0, labelcolor=p.ink2)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Recall (test set)")
        ax.set_title("Recall by class")
        if len(recalls) > 1:
            ax.legend(loc="lower right")
        else:
            ax.set_title(f"Recall by class · {next(iter(recalls))}")
    return fig


def reliability(conditions: dict[str, dict], mode: str) -> Figure:
    """Reliability diagram with bin counts below.

    ``conditions`` maps a label to a report dict (with ``reliability`` and ``ece``),
    in order: the first is drawn as the muted reference (e.g. uncalibrated), the
    second in the accent colour (e.g. temperature-scaled).
    """
    if len(conditions) > 2:
        raise ValueError("a reliability diagram compares at most 2 conditions")
    with style(mode) as p:
        fig = Figure(figsize=(6.4, 6.4), layout="constrained")
        top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1))
        top.plot([0, 1], [0, 1], color=p.axis, lw=1, zorder=1)
        # low confidence is where bins are sparse, so the label stays clear of the curves
        top.annotate(
            "perfectly calibrated",
            (0.2, 0.2),
            xytext=(6, -4),
            textcoords="offset points",
            color=p.muted,
            fontsize=8,
            va="top",
        )
        colors = (p.muted, p.series[0])
        for k, ((label, r), color) in enumerate(zip(conditions.items(), colors, strict=False)):
            bins = r["reliability"]
            top.plot(
                [b["confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                color=color,
                marker="o",
                markersize=7,
                markeredgecolor=p.surface,
                markeredgewidth=2,
                label=f"{label} (ECE {r['ece']:.1%})",
                zorder=3,
            )
            # count strip: side-by-side bars per bin, 2px apart via the surface edge
            width = (bins[0]["hi"] - bins[0]["lo"]) * 0.4 if bins else 0.0
            centers = [(b["lo"] + b["hi"]) / 2 + (k - 0.5) * width for b in bins]
            bottom.bar(
                centers,
                [b["count"] for b in bins],
                width=width,
                color=color,
                edgecolor=p.surface,
                linewidth=1,
            )
        top.set_xlim(0, 1.01)
        top.set_ylim(0, 1.01)
        for axis in (top.xaxis, top.yaxis, bottom.xaxis):
            axis.set_major_formatter(PercentFormatter(1, decimals=0))
        top.grid(True)
        top.set_axisbelow(True)
        top.set_ylabel("Accuracy in bin")
        top.set_title("Confidence vs accuracy (test set)")
        # lower right stays empty: high confidence with low accuracy is rare
        top.legend(loc="lower right")
        bottom.set_yscale("log")
        bottom.set_ylabel("Images")
        bottom.set_xlabel("Top-class confidence")
    return fig


def training_curves(histories: dict[str, list[dict]], mode: str) -> Figure:
    """Small multiples, one measure per panel: never two y-axes on one chart.

    ``histories`` maps a model name to its per-epoch history (at most 3 models).
    One model gets an endpoint label per panel; several get a legend instead.
    """
    if len(histories) > 3:
        raise ValueError("at most 3 models per figure; facet or split the comparison")
    panels = [
        ("train_loss", "Training loss", "{:.3f}"),
        ("balanced_accuracy", "Val balanced accuracy", "{:.1%}"),
        ("ece", "Val ECE", "{:.1%}"),
    ]
    with style(mode) as p:
        fig = Figure(figsize=(9.6, 3.2), layout="constrained")
        axes = fig.subplots(1, 3, sharex=True)
        for ax, (key, title, label) in zip(axes, panels, strict=True):
            for arch, history in histories.items():
                color = model_color(arch, mode)
                epochs = [h["epoch"] for h in history]
                ys = [h[key] for h in history]
                ax.plot(epochs, ys, color=color, label=arch)
                ax.scatter(
                    epochs[-1:],
                    ys[-1:],
                    s=DOT_SIZE,
                    color=color,
                    edgecolors=p.surface,
                    linewidths=2,
                    zorder=3,
                )
                if len(histories) == 1:
                    ax.annotate(
                        label.format(ys[-1]),  # label the endpoint only
                        (epochs[-1], ys[-1]),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        color=p.ink,
                        fontsize=9,
                    )
            if "%" in label:
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0 if key != "ece" else 1))
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.margins(y=0.2)
        if len(histories) > 1:
            axes[0].legend(loc="upper right")
    return fig


# --- rendering -----------------------------------------------------------------------


def render(name: str, draw: Callable[[str], Figure], out: Path) -> list[Path]:
    """Write ``<name>.png`` (light) and ``<name>-dark.png``."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for mode in MODES:
        path = out / (f"{name}.png" if mode == "light" else f"{name}-{mode}.png")
        draw(mode).savefig(path, dpi=DPI)
        paths.append(path)
    return paths


def plot_run(run_dir: Path, out: Path) -> list[Path]:
    """Standard figures for one training run (needs history.json and test_report.json)."""
    history = json.loads((run_dir / "history.json").read_text())
    reports = json.loads((run_dir / "test_report.json").read_text())
    arch = json.loads((run_dir / "config.json").read_text())["arch"]
    raw, scaled = reports["uncalibrated"], reports["temperature_scaled"]
    classes = list(raw["per_class_recall"])
    conditions = {
        "Uncalibrated": raw,
        f"Temperature-scaled, T = {scaled['temperature']:.2f}": scaled,
    }
    figures = {
        "training-curves": lambda m: training_curves({arch: history}, m),
        "confusion-matrix": lambda m: confusion_matrix(raw["confusion_matrix"], classes, m),
        "per-class-recall": lambda m: per_class_recall({arch: raw["per_class_recall"]}, m),
        "reliability": lambda m: reliability(conditions, m),
    }
    return [path for name, draw in figures.items() for path in render(name, draw, out)]


def plot_compare(run_dirs: list[Path], out: Path) -> list[Path]:
    """Side-by-side figures for up to 3 runs, plus each run's own figures in ``out/<arch>``."""
    runs = {}
    for run_dir in run_dirs:
        arch = json.loads((run_dir / "config.json").read_text())["arch"]
        if arch in runs:
            raise ValueError(f"two runs of {arch}; compare different models")
        runs[arch] = run_dir
    histories = {a: json.loads((d / "history.json").read_text()) for a, d in runs.items()}
    recalls = {
        a: json.loads((d / "test_report.json").read_text())["uncalibrated"]["per_class_recall"]
        for a, d in runs.items()
    }
    figures = {
        "training-curves": lambda m: training_curves(histories, m),
        "per-class-recall": lambda m: per_class_recall(recalls, m),
    }
    paths = [path for name, draw in figures.items() for path in render(name, draw, out)]
    for arch, run_dir in runs.items():
        paths += plot_run(run_dir, out / arch)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render the standard figures for one or more runs.")
    ap.add_argument("run_dirs", type=Path, nargs="+", metavar="run_dir")
    ap.add_argument("--out", type=Path, help="default: <run_dir>/figures (required for several)")
    a = ap.parse_args(argv)
    if len(a.run_dirs) == 1:
        paths = plot_run(a.run_dirs[0], a.out or a.run_dirs[0] / "figures")
    elif a.out is None:
        ap.error("--out is required when comparing several runs")
    else:
        paths = plot_compare(a.run_dirs, a.out)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
