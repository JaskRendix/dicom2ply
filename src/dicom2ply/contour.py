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
        ds = cache.load(self.slice_uid)
        self.ds = ds

        points = np.column_stack([self.x, self.y, self.z])
        check_planarity(points, ds)

        row, col = patient_to_pixel(points, ds)
        self.mask = polygon_mask(row, col, ds.pixel_array.shape)

        pixel_array = ds.pixel_array.astype(float)

        if hasattr(ds, "RescaleSlope"):
            pixel_array *= float(ds.RescaleSlope)

        if hasattr(ds, "RescaleIntercept"):
            pixel_array += float(ds.RescaleIntercept)

        masked = pixel_array[self.mask.astype(bool)]
        self.masked_values = masked
        self.slice_pos = slice_position(ds)

        if masked.size == 0:
            return

        hist = np.histogram(masked, bins=self.bins)
        self.stats = ContourStats(
            histogram=hist,
            mode=float(hist[1][np.argmax(hist[0])]),
            mean=float(masked.mean()),
            std=float(masked.std()),
            median=float(np.median(masked)),
        )
