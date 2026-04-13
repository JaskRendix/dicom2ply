from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset

from dicom2ply.roi import RegionOfInterest


@dataclass(frozen=True)
class CTSlice:
    sop_uid: str
    path: Path
    z: float


class Patient:

    def __init__(
        self,
        dicom_dir: str | Path,
        debug: bool = True,
        *,
        reader: Callable[..., FileDataset] = pydicom.dcmread,
        walker: Callable[..., Iterable] = os.walk,
    ) -> None:
        self.debug = debug
        self.dicom_dir = Path(dicom_dir)
        self._reader = reader
        self._walker = walker

        self._files = self._scan_files()

        # Tests expect this exact attribute name
        self.structure: FileDataset = self._load_rtstruct()

        # Deterministic CT slice index
        self._ct_slices = self._index_ct_slices()

        # Tests expect ct_slices and ct_index as SOPUID -> path strings
        self.ct_slices: dict[str, str] = {
            uid: s.path.as_posix() for uid, s in self._ct_slices.items()
        }
        self.ct_index: dict[str, str] = dict(self.ct_slices)

        # ROI metadata
        self.region_names = self._extract_roi_names()  # legacy name
        self._roi_cache: dict[str, RegionOfInterest] = {}

    @property
    def roi_names(self) -> list[str]:
        """List of ROI names available in the RTSTRUCT (for CLI validation)."""
        return list(self.region_names.values())

    @property
    def regions(self) -> dict[str, RegionOfInterest]:
        """
        Backwards-compatible mapping of ROI name -> RegionOfInterest.

        Semantics match the original implementation:
        - Iterate ROIContourSequence
        - Skip entries without a known name or without ContourSequence
        - If ROIContourSequence is empty, return {}
        - Never raise errors
        """
        if self._roi_cache:
            return dict(self._roi_cache)

        roi_contours = getattr(self.structure, "ROIContourSequence", None)
        if not roi_contours:
            return {}

        for roi in roi_contours:
            number = getattr(roi, "ReferencedROINumber", None)
            if number is None:
                continue

            name = self.region_names.get(int(number))
            if name is None:
                continue

            if not hasattr(roi, "ContourSequence"):
                continue

            region = RegionOfInterest.from_rt_roi(
                roi_ds=roi,
                name=name,
                bins=4096,
                ct_index=self.ct_index,
            )
            self._roi_cache[name] = region

        return dict(self._roi_cache)

    def get_roi(self, name: str) -> RegionOfInterest:
        """Return a single ROI by name, loading it on demand."""
        if name not in self.roi_names:
            raise KeyError(f"ROI '{name}' not found. Available: {self.roi_names}")

        if name in self._roi_cache:
            return self._roi_cache[name]

        roi = self._load_single_roi(name)
        self._roi_cache[name] = roi
        return roi

    def dump_ply(
        self,
        directory: str | Path = ".",
        names: Iterable[str] | None = None,
    ) -> None:
        """Export selected ROIs to PLY files."""
        from dicom2ply.ply_writer import write_roi_ply

        if names is None:
            names = self.roi_names

        selected = list(names)
        if not selected:
            raise ValueError("No ROIs found in RTSTRUCT or no names provided.")

        for name in selected:
            roi = self.get_roi(name)
            write_roi_ply(roi, directory)

    def _scan_files(self) -> list[Path]:
        try:
            _, _, files = next(self._walker(self.dicom_dir))
        except StopIteration:
            raise FileNotFoundError(f"No files found in directory: {self.dicom_dir}")

        return [self.dicom_dir / f for f in files]

    def _load_rtstruct(self) -> FileDataset:
        for path in self._files:
            try:
                ds = self._reader(path, stop_before_pixels=True)
            except Exception:
                continue

            if getattr(ds, "Modality", None) == "RTSTRUCT":
                return ds

        raise FileNotFoundError("No RTSTRUCT file found (Modality=RTSTRUCT).")

    def _index_ct_slices(self) -> dict[str, CTSlice]:
        slices: list[CTSlice] = []

        for path in self._files:
            try:
                ds = self._reader(path, stop_before_pixels=True)
            except Exception:
                continue

            if getattr(ds, "Modality", None) != "CT":
                continue

            sop = getattr(ds, "SOPInstanceUID", None)
            if sop is None:
                continue

            ipp = getattr(ds, "ImagePositionPatient", None)
            z = float(ipp[2]) if ipp and len(ipp) >= 3 else 0.0

            slices.append(CTSlice(sop_uid=sop, path=path, z=z))

        slices.sort(key=lambda s: s.z)
        return {s.sop_uid: s for s in slices}

    def _extract_roi_names(self) -> dict[int, str]:
        names: dict[int, str] = {}
        seq = getattr(self.structure, "RTROIObservationsSequence", [])
        for obs in seq:
            number = int(obs.ObservationNumber)
            label = str(obs.ROIObservationLabel)
            names[number] = label
        return names

    def _load_single_roi(self, name: str) -> RegionOfInterest:
        roi_number = None
        for number, label in self.region_names.items():
            if label == name:
                roi_number = number
                break

        if roi_number is None:
            raise KeyError(f"ROI '{name}' not found in RTROIObservationsSequence.")

        roi_contours = getattr(self.structure, "ROIContourSequence", [])
        for roi in roi_contours:
            if int(getattr(roi, "ReferencedROINumber", -1)) != roi_number:
                continue

            if not hasattr(roi, "ContourSequence"):
                raise KeyError(f"ROI '{name}' has no ContourSequence data.")

            return RegionOfInterest.from_rt_roi(
                roi_ds=roi,
                name=name,
                bins=4096,
                ct_index=self.ct_index,
            )

        raise KeyError(f"ROI '{name}' not found in ROIContourSequence.")
