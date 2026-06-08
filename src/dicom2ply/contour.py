from dataclasses import dataclass, field

import numpy as np
from pydicom.dataset import Dataset

from dicom2ply.geometry import check_planarity, patient_to_pixel, slice_position
from dicom2ply.masking import polygon_mask


@dataclass
class ContourStats:
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    mode: float | None = None
    histogram: tuple[np.ndarray, np.ndarray] | None = None


@dataclass
class Contour:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    slice_uid: str
    bins: int

    ds: Dataset | None = None
    mask: np.ndarray | None = None
    masked_values: np.ndarray | None = None
    stats: ContourStats = field(default_factory=ContourStats)
    slice_pos: float | None = None

    @classmethod
    def from_rt(cls, contour_ds: Dataset, bins: int, cache) -> "Contour":
        coords = np.asarray(contour_ds.ContourData, float).reshape(-1, 3)
        x, y, z = coords.T
        uid = contour_ds.ContourImageSequence[0].ReferencedSOPInstanceUID

        obj = cls(x, y, z, uid, bins)
        obj.compute(cache)
        return obj

    def compute(self, cache) -> None:
        """
        Build mask, extract HU values, compute stats, and compute slice position.
        This version avoids unnecessary pixel_array decoding, fixes mask shape,
        and computes slice_pos from contour geometry rather than slice index.
        """
        ds = cache.load(self.slice_uid)
        self.ds = ds

        points = np.column_stack([self.x, self.y, self.z])
        check_planarity(points, ds)

        row, col = patient_to_pixel(points, ds)

        rows = int(ds.Rows)
        cols = int(ds.Columns)

        # Clip polygon coordinates to valid pixel bounds
        row = np.clip(row, 0, rows - 1)
        col = np.clip(col, 0, cols - 1)

        self.mask = polygon_mask(row, col, (rows, cols))

        pixel_array = ds.pixel_array.astype(float)

        # HU rescale
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixel_array = pixel_array * slope + intercept

        masked = pixel_array[self.mask.astype(bool)]
        self.masked_values = masked

        # Project contour centroid onto slice normal
        try:
            self.slice_pos = slice_position(ds)
        except Exception:
            # Fallback: mean z of contour points
            self.slice_pos = float(np.mean(self.z))

        if masked.size == 0:
            return

        counts, edges = np.histogram(masked, bins=self.bins)
        centers = (edges[:-1] + edges[1:]) / 2

        self.stats = ContourStats(
            histogram=(counts, edges),
            mode=float(centers[np.argmax(counts)]),
            mean=float(masked.mean()),
            std=float(masked.std()),
            median=float(np.median(masked)),
        )
