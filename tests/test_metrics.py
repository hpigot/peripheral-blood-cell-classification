import numpy as np

from bloodcell.metrics import expected_calibration_error, fit_temperature, report, softmax


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_ece_is_zero_for_perfect_confident_predictions():
    probs = np.eye(3)[[0, 1, 2, 0]]
    assert expected_calibration_error(probs, np.array([0, 1, 2, 0])) == 0.0


def test_ece_detects_overconfidence():
    probs = np.tile([0.99, 0.01], (10, 1))
    y = np.array([0] * 5 + [1] * 5)  # 50% accurate at 99% confidence
    assert abs(expected_calibration_error(probs, y) - 0.49) < 1e-6


def test_temperature_scaling_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(size=(2000, 3))
    logits[np.arange(2000), y] += 1.0
    logits *= 5.0  # same ranking, far too confident
    t = fit_temperature(logits, y)
    assert t > 2.0
    assert expected_calibration_error(softmax(logits, t), y) < expected_calibration_error(
        softmax(logits), y
    )


def test_report_keys_and_confusion_shape():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    r = report(logits, np.array([0, 1, 1]), ["a", "b"])
    assert r["accuracy"] == 2 / 3
    assert r["confusion_matrix"] == [[1, 0], [1, 1]]
    assert set(r["per_class_recall"]) == {"a", "b"}
