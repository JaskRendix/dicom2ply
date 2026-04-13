from dataclasses import dataclass

import numpy as np
from pydicom.dataset import Dataset

from dicom2ply.contour import Contour
from dicom2ply.ct_cache import CTSliceCache


@dataclass
class RegionOfInterest:
    name: str
    contours: list[Contour]
    bins: int

    histogram: tuple[np.ndarray, np.ndarray] | None = None
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    mode: float | None = None
    sum: float | None = None
    count: int | None = None
    extent: tuple[float, float, float, float, float, float] | None = None
    mask_stack: np.ndarray | None = None
    slice_positions: np.ndarray | None = None

    @classmethod
    def from_rt_roi(
        cls, roi_ds: Dataset, name: str, bins: int, ct_index: dict[str, str]
    ):
        """
        Build an ROI from an RTSTRUCT ROIContourSequence entry.
        Uses the modular Contour.from_rt + CTSliceCache pipeline.
        """
        cache = CTSliceCache(ct_index)

        seq = getattr(roi_ds, "ContourSequence", None)
        if not seq:
            return cls(name=name, contours=[], bins=bins)

        contours: list[Contour] = []
        for contour_ds in seq:
            c = Contour.from_rt(contour_ds, bins=bins, cache=cache)
            if c.stats.mean is not None:  # skip empty masks
                contours.append(c)

        # Sort contours deterministically by slice position
        contours.sort(key=lambda c: c.slice_pos)

        obj = cls(name=name, contours=contours, bins=bins)
        obj.compute_stats()
        obj.compute_extent()
        obj.compute_mask_stack()
        return obj

    def compute_stats(self):
        if not self.contours:
            self.count = 0
            return

        values = np.concatenate([c.masked_values for c in self.contours])
        counts, edges = np.histogram(values, bins=self.bins)

        centers = (edges[:-1] + edges[1:]) / 2

        self.histogram = (counts, edges)
        self.mode = float(centers[np.argmax(counts)])
        self.mean = float(values.mean())
        self.std = float(values.std())
        self.median = float(np.median(values))
        self.sum = float(values.sum())
        self.count = int(values.size)

    def compute_extent(self):
        if not self.contours:
            return

        xs = np.concatenate([c.x for c in self.contours])
        ys = np.concatenate([c.y for c in self.contours])
        zs = np.concatenate([c.z for c in self.contours])

        self.extent = (
            float(xs.min()),
            float(xs.max()),
            float(ys.min()),
            float(ys.max()),
            float(zs.min()),
            float(zs.max()),
        )

    def compute_mask_stack(self):
        if not self.contours:
            return

        # Avoid pixel_array decode: use metadata
        ds0 = self.contours[0].ds
        rows = int(ds0.Rows)
        cols = int(ds0.Columns)

        positions = np.array([c.slice_pos for c in self.contours])
        uniq = np.unique(positions)
        uniq.sort()
        self.slice_positions = uniq

        pos_to_idx = {p: i for i, p in enumerate(uniq)}
        volume = np.zeros((rows, cols, len(uniq)), np.int8)

        for c in self.contours:
            idx = pos_to_idx[c.slice_pos]

            if c.mask.shape != (rows, cols):
                continue

            volume[..., idx] |= c.mask

        self.mask_stack = volume
