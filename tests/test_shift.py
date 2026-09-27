import numpy as np
from PIL import Image

from bloodcell.shift import Correct


def test_correct_without_steps_is_identity():
    img = Image.fromarray(np.zeros((40, 30, 3), np.uint8))
    assert Correct()(img) is img
    assert Correct(scale=0.5)(img).size == img.size
