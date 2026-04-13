import numpy as np
from pydicom.dataset import Dataset


def slice_orientation(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal row, col, normal vectors."""
    iop = np.asarray(ds.ImageOrientationPatient, float)
    row = iop[:3]
    col = iop[3:]

    # Re-orthogonalize for numerical stability
    row /= np.linalg.norm(row)
    col = col - np.dot(col, row) * row
    col /= np.linalg.norm(col)

    normal = np.cross(row, col)
    return row, col, normal


def slice_position(ds: Dataset) -> float:
    """Return scalar slice position along the normal direction."""
    row, col, normal = slice_orientation(ds)
    origin = np.asarray(ds.ImagePositionPatient, float)
    return float(np.dot(normal, origin))


def check_planarity(points: np.ndarray, ds: Dataset, tol_mm: float = 1e-2) -> None:
    """Ensure contour points lie on the CT slice plane."""
    _, _, normal = slice_orientation(ds)
    origin = np.asarray(ds.ImagePositionPatient, float)
    distances = np.abs((points - origin) @ normal)
    if distances.max() > tol_mm:
        raise ValueError(
            f"Contour not coplanar: max distance {distances.max():.4f} mm > {tol_mm} mm"
        )


def patient_to_pixel(points: np.ndarray, ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Map patient-space (mm) points → pixel indices (row, col)."""
    row_dir, col_dir, _ = slice_orientation(ds)
    origin = np.asarray(ds.ImagePositionPatient, float)
    spacing = np.asarray(ds.PixelSpacing, float)

    v = points - origin
    row = (v @ row_dir) / spacing[0]
    col = (v @ col_dir) / spacing[1]
    return row, col
