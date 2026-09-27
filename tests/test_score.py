import json

import pytest

from bloodcell.score import main

ROWS = [  # split, path, label
    ("train", "basophil/b0.jpg", "basophil"),
    ("train", "neutrophil/n0.jpg", "neutrophil"),
    ("test", "basophil/b1.jpg", "basophil"),
    ("test", "neutrophil/n1.jpg", "neutrophil"),
    ("test", "neutrophil/n2.jpg", "neutrophil"),
]


def _write(tmp_path, cpp_lines, python_rows):
    root = tmp_path / "images"
    splits = tmp_path / "splits.csv"
    splits.write_text("split,path,label\n" + "".join(f"{s},{p},{c}\n" for s, p, c in ROWS))
    preds = tmp_path / "preds.tsv"
    preds.write_text("".join(f"/home/pi/cells/{p}\t{c}\t{x}\n" for p, c, x in cpp_lines))
    python = tmp_path / "test_predictions.csv"
    python.write_text(
        "path,label,predicted,confidence\n" + "".join(f"{p},,{c},{x}\n" for p, c, x in python_rows)
    )
    return ["--root", str(root), "--splits", str(splits), "--against", str(python)], preds


CPP = [
    ("basophil/b1.jpg", "basophil", 0.91),
    ("neutrophil/n1.jpg", "neutrophil", 0.8),
    ("neutrophil/n2.jpg", "basophil", 0.6),
]


def test_scores_cpp_predictions_on_the_test_split(tmp_path):
    common, preds = _write(tmp_path, CPP, CPP)
    main([str(preds), *common])
    report = json.loads((tmp_path / "preds_report.json").read_text())
    assert report["n"] == 3
    assert report["per_class_recall"] == {"basophil": 1.0, "neutrophil": 0.5}
    assert report["parity"]["label_agreement"] == 1.0
    assert report["parity"]["max_abs_confidence_diff"] == 0.0


def test_fails_when_cpp_and_python_disagree(tmp_path):
    python = [*CPP[:2], ("neutrophil/n2.jpg", "neutrophil", 0.6)]
    common, preds = _write(tmp_path, CPP, python)
    with pytest.raises(SystemExit, match="disagree"):
        main([str(preds), *common])


def test_fails_when_a_test_image_has_no_prediction(tmp_path):
    common, preds = _write(tmp_path, CPP[:2], CPP)
    with pytest.raises(SystemExit, match="no prediction"):
        main([str(preds), *common])
