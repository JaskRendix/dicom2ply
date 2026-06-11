import numpy as np
from pydicom.dataset import Dataset


def _require_tag(ds: Dataset, name: str, length: int) -> np.ndarray:
    """Validate presence and shape of a required DICOM geometry tag."""
    if not hasattr(ds, name):
        raise ValueError(f"Missing required DICOM tag: {name}")

    value = getattr(ds, name)
    arr = np.asarray(value, float)

    if arr.size != length:
        raise ValueError(f"Invalid {name}: expected {length} values, got {arr.size}")

    return arr


def slice_orientation(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return orthonormal row, col, normal vectors for the CT slice.

    Ensures:
    - ImageOrientationPatient is valid
    - row/col vectors are normalized
    - col is re-orthogonalized against row
    - normal is normalized
    """
    iop = _require_tag(ds, "ImageOrientationPatient", 6)

    row = iop[:3]
    col = iop[3:]

    # Validate non-degenerate vectors
    nr = np.linalg.norm(row)
    if nr < 1e-6:
        raise ValueError("Invalid row direction vector (zero or near-zero length)")

    nc = np.linalg.norm(col)
    if nc < 1e-6:
        raise ValueError("Invalid column direction vector (zero or near-zero length)")

    # Normalize row
    row = row / nr

    # Re-orthogonalize col
    col = col - np.dot(col, row) * row
    nc2 = np.linalg.norm(col)
    if nc2 < 1e-6:
        raise ValueError("Row and column direction vectors are nearly parallel")
    col = col / nc2

    # Compute and normalize normal
    normal = np.cross(row, col)
    nn = np.linalg.norm(normal)
    if nn < 1e-6:
        raise ValueError("Failed to compute valid slice normal vector")
    normal = normal / nn

    return row, col, normal


def slice_position(ds: Dataset) -> float:
    """
    Return scalar slice position along the slice normal direction.

    Equivalent to projecting ImagePositionPatient onto the normal vector.
    """
    origin = _require_tag(ds, "ImagePositionPatient", 3)
    _, _, normal = slice_orientation(ds)
    return float(np.dot(normal, origin))


def check_planarity(points: np.ndarray, ds: Dataset, tol_mm: float = 1e-2) -> None:
    """
    Ensure contour points lie on the CT slice plane within tolerance.

    Raises:
        ValueError if any point deviates from the plane by more than tol_mm.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")

    origin = _require_tag(ds, "ImagePositionPatient", 3)
    _, _, normal = slice_orientation(ds)

    distances = np.abs((points - origin) @ normal)
    max_dist = float(distances.max())

    if max_dist > tol_mm:
        raise ValueError(
            f"Contour not coplanar: max distance {max_dist:.4f} mm > {tol_mm} mm"
        )


def patient_to_pixel(points: np.ndarray, ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Map patient-space (mm) points → pixel indices (row, col).

    Returns:
        row_indices, col_indices (both float arrays)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")

    origin = _require_tag(ds, "ImagePositionPatient", 3)
    spacing = _require_tag(ds, "PixelSpacing", 2)

    row_dir, col_dir, _ = slice_orientation(ds)

    v = points - origin

    # Precompute scale factors for speed
    row_scale = 1.0 / spacing[0]
    col_scale = 1.0 / spacing[1]

    row = (v @ row_dir) * row_scale
    col = (v @ col_dir) * col_scale

    return row, col
