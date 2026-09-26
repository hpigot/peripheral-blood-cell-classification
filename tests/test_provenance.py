import re
import subprocess

from bloodcell import provenance as prov


def test_provenance_records_code_split_and_versions(tmp_path):
    splits = tmp_path / "splits.csv"
    splits.write_text("split,path,label\ntest,a/1.jpg,a\n")
    p = prov.provenance(splits, "cpu: test")
    assert re.fullmatch(r"[0-9a-f]{40}", p["git_sha"])  # the tests run from the checkout
    assert isinstance(p["git_dirty"], bool)
    assert p["splits_sha256"] == prov.file_sha256(splits)
    assert p["packages"]["numpy"]
    assert p["device"] == "cpu: test"


def test_split_hash_changes_with_content(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("split,path,label\ntest,x.jpg,x\n")
    b.write_text("split,path,label\ntrain,x.jpg,x\n")
    assert prov.file_sha256(a) != prov.file_sha256(b)


def test_missing_git_gives_none_instead_of_failing(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", no_git)
    splits = tmp_path / "splits.csv"
    splits.write_text("split,path,label\n")
    p = prov.provenance(splits, "cpu")
    assert p["git_sha"] is None
    assert p["git_dirty"] is None
