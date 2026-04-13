from __future__ import annotations

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
        walker: Callable[[Path], Iterable[Path]] | None = None,
    ) -> None:
        self.debug = debug
        self.dicom_dir = Path(dicom_dir)
        self._reader = reader
        self._walker = walker or (lambda p: (f for f in p.iterdir() if f.is_file()))

        # All files in the directory (tests expect this attribute name)
        self._files: list[Path] = list(self._walker(self.dicom_dir))

        # Single-pass read cache: Path -> FileDataset
        self._datasets: dict[Path, FileDataset] = self._read_all_datasets()

        # Tests expect this exact attribute name
        self.structure: FileDataset = self._load_rtstruct()

        # Deterministic CT slice index
        self._ct_slices: dict[str, CTSlice] = self._index_ct_slices()

        # Tests expect ct_slices and ct_index as SOPUID -> path strings
        self.ct_slices: dict[str, str] = {
            uid: s.path.as_posix() for uid, s in self._ct_slices.items()
        }
        self.ct_index: dict[str, str] = dict(self.ct_slices)

        # ROI metadata
        self.region_names: dict[int, str] = self._extract_roi_names()  # legacy name

        # Pre-index ROIContourSequence by ReferencedROINumber for O(1) lookup
        self._roi_by_number: dict[int, FileDataset] = self._index_roi_contours()

        # Lazy cache of ROI name -> RegionOfInterest
        self._roi_cache: dict[str, RegionOfInterest] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

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

            # Avoid recomputing if already cached via get_roi
            if name in self._roi_cache:
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

        try:
            return self._roi_cache[name]
        except KeyError:
            pass

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
            try:
                roi = self.get_roi(name)
            except KeyError as e:
                raise ValueError(f"Cannot export ROI '{name}': {e}") from e

            write_roi_ply(roi, directory)

    # -------------------------------------------------------------------------
    # Internal helpers: DICOM loading and indexing
    # -------------------------------------------------------------------------

    def _read_all_datasets(self) -> dict[Path, FileDataset]:
        """
        Read all DICOM files once, without pixel data, and cache them.

        This avoids repeated I/O in _load_rtstruct and _index_ct_slices.
        """
        datasets: dict[Path, FileDataset] = {}
        for path in self._files:
            try:
                ds = self._reader(path, stop_before_pixels=True)
            except Exception:
                continue
            datasets[path] = ds
        return datasets

    def _load_rtstruct(self) -> FileDataset:
        for path in self._files:
            ds = self._datasets.get(path)
            if ds is None:
                continue

            if getattr(ds, "Modality", None) == "RTSTRUCT":
                return ds

        raise FileNotFoundError("No RTSTRUCT file found (Modality=RTSTRUCT).")

    def _index_ct_slices(self) -> dict[str, CTSlice]:
        slices: list[CTSlice] = []

        for path in self._files:
            ds = self._datasets.get(path)
            if ds is None:
                continue

            if getattr(ds, "Modality", None) != "CT":
                continue

            sop = getattr(ds, "SOPInstanceUID", None)
            if sop is None:
                continue

            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp and len(ipp) >= 3:
                try:
                    z = float(ipp[2])
                except (TypeError, ValueError):
                    z = float(getattr(ds, "InstanceNumber", 0))
            else:
                # safer fallback than collapsing everything to 0.0
                z = float(getattr(ds, "InstanceNumber", 0))

            slices.append(CTSlice(sop_uid=str(sop), path=path, z=z))

        slices.sort(key=lambda s: s.z)
        return {s.sop_uid: s for s in slices}

    def _extract_roi_names(self) -> dict[int, str]:
        """
        Build a mapping ObservationNumber/ROINumber -> ROI name.

        Primary source: RTROIObservationsSequence (ROIObservationLabel).
        Fallback: StructureSetROISequence (ROIName).
        """
        names: dict[int, str] = {}

        seq = getattr(self.structure, "RTROIObservationsSequence", [])
        for obs in seq:
            try:
                number = int(obs.ObservationNumber)
            except Exception:
                continue
            label = str(getattr(obs, "ROIObservationLabel", "")).strip()
            if not label:
                continue
            names[number] = label

        # Fallback: some RTSTRUCTs only populate StructureSetROISequence
        if not names:
            seq2 = getattr(self.structure, "StructureSetROISequence", [])
            for roi in seq2:
                try:
                    number = int(roi.ROINumber)
                except Exception:
                    continue
                label = str(getattr(roi, "ROIName", "")).strip()
                if not label:
                    continue
                names[number] = label

        return names

    def _index_roi_contours(self) -> dict[int, FileDataset]:
        """
        Pre-index ROIContourSequence by ReferencedROINumber.

        This is used by _load_single_roi for O(1) lookup.
        """
        index: dict[int, FileDataset] = {}
        seq = getattr(self.structure, "ROIContourSequence", [])
        for roi in seq:
            number = getattr(roi, "ReferencedROINumber", None)
            if number is None:
                continue
            try:
                num_int = int(number)
            except Exception:
                continue
            index[num_int] = roi
        return index

    def _load_single_roi(self, name: str) -> RegionOfInterest:
        """
        Load a single ROI by name from the RTSTRUCT.

        Semantics:
        - Find ROI number from region_names
        - Find corresponding ROIContourSequence entry
        - Require ContourSequence to be present
        - Raise KeyError with informative messages on failure
        """
        roi_number: int | None = None
        for number, label in self.region_names.items():
            if label == name:
                roi_number = number
                break

        if roi_number is None:
            raise KeyError(f"ROI '{name}' not found in RTROIObservationsSequence.")

        roi_ds = self._roi_by_number.get(roi_number)
        if roi_ds is None:
            raise KeyError(f"ROI '{name}' not found in ROIContourSequence.")

        if not hasattr(roi_ds, "ContourSequence"):
            raise KeyError(f"ROI '{name}' has no ContourSequence data.")

        return RegionOfInterest.from_rt_roi(
            roi_ds=roi_ds,
            name=name,
            bins=4096,
            ct_index=self.ct_index,
        )
