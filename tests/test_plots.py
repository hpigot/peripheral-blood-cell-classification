import json

import numpy as np
import pytest

pytest.importorskip("seaborn")

from PIL import Image  # noqa: E402

from bloodcell import plots  # noqa: E402
from bloodcell.metrics import report  # noqa: E402

CLASSES = ["basophil", "ig", "neutrophil"]


def fake_report(temperature: float = 1.0) -> dict:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    logits = rng.normal(size=(300, 3))
    logits[np.arange(300), y] += 3.0
    return report(logits, y, CLASSES, temperature)


def make_run(run_dir, arch="mobilenet_v3_small"):
    run_dir.mkdir()
    history = [
        {"epoch": e, "train_loss": 1 / e, "balanced_accuracy": 0.9, "ece": 0.02} for e in (1, 2, 3)
    ]
    reports = {"uncalibrated": fake_report(), "temperature_scaled": fake_report(1.5)}
    (run_dir / "history.json").write_text(json.dumps(history))
    (run_dir / "test_report.json").write_text(json.dumps(reports))
    (run_dir / "config.json").write_text(json.dumps({"arch": arch}))


def test_plot_run_writes_light_and_dark_variants(tmp_path):
    make_run(tmp_path / "run")
    paths = plots.plot_run(tmp_path / "run", tmp_path / "figs")
    names = sorted(p.name for p in paths)
    assert names == sorted(
        f"{fig}{suffix}.png"
        for fig in ("training-curves", "confusion-matrix", "per-class-recall", "reliability")
        for suffix in ("", "-dark")
    )
    light = Image.open(tmp_path / "figs" / "reliability.png").convert("RGB")
    dark = Image.open(tmp_path / "figs" / "reliability-dark.png").convert("RGB")
    # the corner pixel is the chart surface of each mode
    assert light.getpixel((0, 0)) == (0xFC, 0xFC, 0xFB)
    assert dark.getpixel((0, 0)) == (0x1A, 0x1A, 0x19)


def test_plot_run_draws_mistakes_when_images_are_available(tmp_path):
    make_run(tmp_path / "run")
    root = tmp_path / "images"
    (root / "ig").mkdir(parents=True)
    rows = ["path,label,predicted,confidence"]
    for i in range(3):
        Image.new("RGB", (36, 36), (200, 120 + 40 * i, 200)).save(root / "ig" / f"{i}.png")
        rows.append(f"ig/{i}.png,ig,{'neutrophil' if i else 'ig'},0.9{i}")
    (tmp_path / "run" / "test_predictions.csv").write_text("\n".join(rows) + "\n")
    names = {p.name for p in plots.plot_run(tmp_path / "run", tmp_path / "figs", root)}
    assert {"misclassified.png", "misclassified-dark.png"} <= names
    # without the image folder the figure is skipped, not an error
    names = {p.name for p in plots.plot_run(tmp_path / "run", tmp_path / "figs2")}
    assert "misclassified.png" not in names


def test_plot_run_adds_external_figures_when_scored(tmp_path):
    from bloodcell.external import score_external

    make_run(tmp_path / "run")
    rng = np.random.default_rng(1)
    labels = ["basophil", "neutrophil"] * 20
    logits = rng.normal(size=(40, 3))
    ext = score_external(logits, labels, CLASSES, temperature=1.2)
    (tmp_path / "run" / "external_raabin.json").write_text(json.dumps(ext))
    names = {p.name for p in plots.plot_run(tmp_path / "run", tmp_path / "figs")}
    assert {"external-recall.png", "external-reliability-dark.png"} <= names


def test_mistakes_figure_handles_a_perfect_run(tmp_path):
    rows = [{"path": "x.png", "label": "ig", "predicted": "ig", "confidence": "0.99"}]
    plots.misclassified(rows, tmp_path, "light")


def test_plot_compare_writes_comparison_and_per_run_figures(tmp_path):
    make_run(tmp_path / "a")
    make_run(tmp_path / "b", arch="efficientnet_b0")
    paths = plots.plot_compare([tmp_path / "a", tmp_path / "b"], tmp_path / "figs")
    rel = {p.relative_to(tmp_path / "figs").as_posix() for p in paths}
    assert {"training-curves.png", "per-class-recall-dark.png"} <= rel
    assert {"mobilenet_v3_small/reliability.png", "efficientnet_b0/confusion-matrix.png"} <= rel
    assert len(rel) == 2 * 2 + 2 * 8


def test_plot_compare_rejects_two_runs_of_one_model(tmp_path):
    make_run(tmp_path / "a")
    make_run(tmp_path / "b")
    with pytest.raises(ValueError, match="two runs of mobilenet_v3_small"):
        plots.plot_compare([tmp_path / "a", tmp_path / "b"], tmp_path / "figs")


def test_style_does_not_leak_into_global_rcparams():
    import matplotlib

    before = matplotlib.rcParams["axes.facecolor"]
    plots.confusion_matrix([[5, 1], [0, 4]], ["a", "b"], "dark")
    assert matplotlib.rcParams["axes.facecolor"] == before


def test_known_models_use_exactly_the_validated_slots():
    for mode in plots.MODES:
        colors = [plots.model_color(arch, mode) for arch in plots.MODEL_SLOTS]
        assert sorted(colors) == sorted(plots.PALETTES[mode].series)


def test_at_most_three_series_per_figure():
    rec = {c: 0.9 for c in CLASSES}
    with pytest.raises(ValueError, match="at most 3"):
        plots.per_class_recall({f"m{i}": rec for i in range(4)}, "light")
    history = [{"epoch": 1, "train_loss": 1.0, "balanced_accuracy": 0.9, "ece": 0.02}]
    with pytest.raises(ValueError, match="exactly 2"):
        plots.recall_shift({f"m{i}": rec for i in range(3)}, "light")
    with pytest.raises(ValueError, match="at most 3"):
        plots.training_curves({f"m{i}": history for i in range(4)}, "light")
