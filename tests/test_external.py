import json

import numpy as np

from bloodcell.external import RAABIN_LABELS, read_raabin, score_external

CLASSES = ["a", "b", "c"]


def test_forced_choice_ignores_classes_the_external_set_lacks():
    # the model's top class for the third image is "c", which the external set doesn't have
    logits = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 0.0, 5.0], [0.0, 2.0, 0.0]])
    r = score_external(logits, ["a", "b", "a", "b"], CLASSES, temperature=1.0)
    assert r["forced_choice"]["temperature_scaled"]["accuracy"] == 1.0
    free = r["all_classes"]
    assert free["accuracy"] == 0.75
    assert free["per_class_recall"] == {"a": 0.5, "b": 1.0}
    assert free["predicted_absent_class"] == 0.25
    assert free["confusion_matrix"] == [[1, 0, 1], [0, 2, 0]]


def test_read_raabin_maps_numeric_labels(tmp_path):
    (tmp_path / "labels.json").write_text(json.dumps({"x.jpg": 4, "y.jpg": 1}))
    samples = read_raabin(tmp_path / "labels.json", tmp_path / "images")
    assert [(s.path.name, s.label) for s in samples] == [
        ("x.jpg", "eosinophil"),
        ("y.jpg", "neutrophil"),
    ]
    assert set(RAABIN_LABELS.values()) == {
        "neutrophil",
        "lymphocyte",
        "monocyte",
        "eosinophil",
        "basophil",
    }
