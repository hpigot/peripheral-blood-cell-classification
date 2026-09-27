import numpy as np
from PIL import Image, ImageDraw

from bloodcell.stain import (
    lab_stats,
    lab_to_rgb,
    macenko,
    macenko_fit,
    nucleus_diameter,
    reinhard,
    rescale,
    rgb_to_lab,
)


def _noise(seed=0, size=(40, 30)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8))


def test_lab_round_trip_and_white_point():
    rgb = np.asarray(_noise())
    assert np.abs(lab_to_rgb(rgb_to_lab(rgb)).astype(int) - rgb).max() <= 1
    assert np.allclose(rgb_to_lab(np.array([255, 255, 255], np.uint8)), [100, 0, 0], atol=0.01)


def test_reinhard_moves_lab_statistics_to_the_target():
    mean, std = np.array([70.0, 15.0, -5.0]), np.array([10.0, 5.0, 8.0])
    m, s = lab_stats(reinhard(_noise(), mean, std))
    assert np.allclose(m, mean, atol=1.0)
    assert np.allclose(s, std, atol=1.0)


def test_rescale_keeps_size_and_centres_the_content():
    img = Image.new("RGB", (60, 40), "white")
    ImageDraw.Draw(img).rectangle((20, 10, 39, 29), fill="black")  # 20 x 20, centred
    small = np.asarray(rescale(img, 0.5).convert("L"))
    big = np.asarray(rescale(img, 2.0).convert("L"))
    assert small.shape == big.shape == (40, 60)
    # the shrunk image sits in the middle; the border outside it is a reflection
    assert (small[10:30, 15:45] < 128).sum() == 10 * 10
    assert small[20, 30] < 128 and big[0, 20] < 128 and big[39, 39] < 128


def _two_stains(seed=0, n=64):
    """A synthetic smear: known stain vectors, random concentrations, background light."""
    stains = np.array([[0.75, 0.65, 0.12], [0.30, 0.95, 0.10]]).T
    stains /= np.linalg.norm(stains, axis=0)
    rng = np.random.default_rng(seed)
    conc = rng.uniform(0, 1.5, (2, n * n)) * (rng.uniform(size=n * n) > 0.3)
    light = np.array([250.0, 235.0, 210.0])
    rgb = light * np.exp(-(stains @ conc)).T - 1
    return stains, np.clip(rgb, 0, 255).astype(np.uint8).reshape(n, n, 3)


def test_macenko_recovers_the_stain_vectors():
    stains, rgb = _two_stains()
    fitted, max_conc, light = macenko_fit(rgb)
    assert (np.sum(fitted * stains, axis=0) > 0.99).all()  # cosine similarity
    assert np.allclose(light, [250, 235, 210], atol=2)
    assert (max_conc > 1.0).all()


def test_macenko_to_its_own_fit_changes_little():
    _, rgb = _two_stains(seed=1)
    out = np.asarray(macenko(Image.fromarray(rgb), *macenko_fit(rgb)))
    assert np.abs(out.astype(int) - rgb).mean() < 2


def test_nucleus_diameter_measures_a_central_disk():
    img = Image.new("RGB", (448, 448), (240, 220, 200))
    ImageDraw.Draw(img).ellipse((164, 164, 283, 283), fill=(60, 40, 120))  # diameter 120
    assert abs(nucleus_diameter(img) - 60) < 2  # measured at 224 x 224
    assert np.isnan(nucleus_diameter(Image.new("RGB", (100, 100), "white")))
