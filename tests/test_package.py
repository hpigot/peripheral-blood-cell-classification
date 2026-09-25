from importlib.metadata import version

import bloodcell


def test_version_matches_metadata():
    assert bloodcell.__version__ == version("bloodcell")
