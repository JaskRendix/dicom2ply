import numpy as np
from skimage.draw import polygon2mask


def polygon_mask(
    row: np.ndarray, col: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Return binary mask for polygon in image space."""
    if len(row) < 3:
        return np.zeros(shape, np.int8)

    coords = np.column_stack([row, col])
    mask = polygon2mask(shape, coords)
    return mask.astype(np.int8)
