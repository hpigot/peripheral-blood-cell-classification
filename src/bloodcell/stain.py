"""Test-time corrections for a change of lab: colour and scale (numpy only, no torch).

Colour: Reinhard et al. (2001) colour transfer. Each image's CIELAB channels
are shifted and scaled to a target mean and standard deviation, here the
average over PBC training images. Scale: shrink or enlarge the image content
by a factor and keep the image size, so cells appear at the size the model
was trained on. The factor comes from ``nucleus_diameter``, not from labels.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# sRGB (D65) to CIE XYZ, and the D65 white point
_M = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_WHITE = _M.sum(axis=1)
_EPS = (6 / 29) ** 3


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (..., 3) to CIELAB (..., 3) float."""
    c = rgb.astype(np.float64) / 255
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = lin @ _M.T / _WHITE
    f = np.where(t > _EPS, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB (..., 3) to uint8 RGB, clipping colours outside the sRGB gamut."""
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    t = np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29))
    lin = np.clip(t * _WHITE @ np.linalg.inv(_M).T, 0, 1)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.round(c * 255).astype(np.uint8)


def lab_stats(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel CIELAB mean and standard deviation of an image."""
    lab = rgb_to_lab(np.asarray(img.convert("RGB"))).reshape(-1, 3)
    return lab.mean(axis=0), lab.std(axis=0)


def reinhard(img: Image.Image, mean: np.ndarray, std: np.ndarray) -> Image.Image:
    """Move the image's CIELAB mean and standard deviation to ``mean`` and ``std``."""
    lab = rgb_to_lab(np.asarray(img.convert("RGB")))
    m = lab.reshape(-1, 3).mean(axis=0)
    s = lab.reshape(-1, 3).std(axis=0)
    out = (lab - m) / np.maximum(s, 1e-6) * std + mean
    return Image.fromarray(lab_to_rgb(out))


def background(rgb: np.ndarray) -> np.ndarray:
    """Light through empty slide: the 95th percentile of each channel.

    Neither lab's background is white, so optical density is taken relative
    to this rather than to 255.
    """
    return np.percentile(rgb.reshape(-1, 3), 95, axis=0).astype(np.float64) + 1


def _od(rgb: np.ndarray, light: np.ndarray) -> np.ndarray:
    return np.maximum(-np.log((rgb.reshape(-1, 3).astype(np.float64) + 1) / light), 0)


def macenko_fit(
    rgb: np.ndarray, beta: float = 0.15, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two stain colours (3 x 2, unit optical-density vectors), their 99th-percentile
    concentrations and the background light, following Macenko et al. (2009).

    The stain whose vector has the larger red component comes first: in a
    May-Grünwald-Giemsa smear that's the blue dye of the nuclei. Pixels
    with optical density below ``beta`` in any channel are background.
    """
    light = background(rgb)
    od = _od(rgb, light)
    od = od[(od > beta).all(axis=1)]
    _, vecs = np.linalg.eigh(np.cov(od.T))
    plane = vecs[:, [2, 1]]  # the two largest eigenvectors
    # point both into positive optical density, so angles don't wrap around pi
    plane *= np.where(plane.sum(axis=0) < 0, -1, 1)
    proj = od @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [alpha, 100 - alpha])
    v = plane @ np.array([[np.cos(lo), np.cos(hi)], [np.sin(lo), np.sin(hi)]])
    v *= np.where(v[0] < 0, -1, 1)
    stains = v[:, np.argsort(-v[0])] / np.linalg.norm(v, axis=0)[np.argsort(-v[0])]
    conc = np.linalg.lstsq(stains, od.T, rcond=None)[0]
    return stains, np.percentile(conc, 99, axis=1), light


def macenko(
    img: Image.Image, stains: np.ndarray, max_conc: np.ndarray, light: np.ndarray
) -> Image.Image:
    """Re-render the image with the target stains, concentration range and background.

    Optical density outside the plane of the two stains (the orange of
    eosinophil granules and red cells, which a two-stain model can't hold)
    is carried over unchanged.
    """
    rgb = np.asarray(img.convert("RGB"))
    src, src_max, src_light = macenko_fit(rgb)
    od = _od(rgb, src_light).T
    conc = np.linalg.lstsq(src, od, rcond=None)[0]
    residual = od - src @ conc
    conc *= (max_conc / np.maximum(src_max, 1e-6))[:, None]
    out = light * np.exp(-(stains @ conc + residual)).T - 1
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8).reshape(rgb.shape))


def rescale(img: Image.Image, factor: float) -> Image.Image:
    """Scale the content by ``factor`` and keep the image size.

    Shrinking reflects the border into the gap; enlarging crops the centre.
    """
    w, h = img.size
    sw, sh = round(w * factor), round(h * factor)
    a = np.asarray(img.convert("RGB").resize((sw, sh), Image.Resampling.BILINEAR))
    if factor < 1:
        (top, left) = ((h - sh) // 2, (w - sw) // 2)
        a = np.pad(a, ((top, h - sh - top), (left, w - sw - left), (0, 0)), mode="reflect")
    else:
        top, left = (sh - h) // 2, (sw - w) // 2
        a = a[top : top + h, left : left + w]
    return Image.fromarray(a)


def _otsu(g: np.ndarray) -> float:
    hist, _ = np.histogram(g, 256, (0, 256))
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return float(between.argmax())


def nucleus_diameter(img: Image.Image, size: int = 224) -> float:
    """Equivalent diameter, in pixels at ``size`` x ``size``, of the central nucleus.

    The nucleus is the dark region (Otsu threshold on the green channel) that
    covers the image centre. Returns NaN when that region is missing, touches
    the border or isn't compact (it has merged with red cells), so a median
    over many images ignores the failures.
    """
    from scipy import ndimage as ndi

    g = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.float64)[..., 1]
    mask = ndi.binary_opening(g < _otsu(g), iterations=2)
    labels, _ = ndi.label(mask)
    centre = labels[size // 2, size // 2]
    if centre == 0:
        return float("nan")
    ys, xs = np.nonzero(labels == centre)
    area = len(ys)
    fill = area / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1))
    edge = min(ys.min(), xs.min()) == 0 or max(ys.max(), xs.max()) == size - 1
    if fill < 0.7 or edge:
        return float("nan")
    return float(2 * np.sqrt(area / np.pi))
