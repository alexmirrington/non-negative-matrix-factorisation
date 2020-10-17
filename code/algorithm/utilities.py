"""Miscellaneous utilities."""
import numpy as np


def rescale(img: np.ndarray, lower: float = 0, upper: float = 255):
    """Rescale an image's pixels to be between `lower` and `upper`."""
    img_min = np.min(img)
    img_max = np.max(img)
    return (upper - lower) * ((img - img_min) / (img_max - img_min)) + lower
