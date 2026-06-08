from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset

from dicom2ply.roi import RegionOfInterest

logger = logging.getLogger(__name__)


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
        # Configure logger level based on debug flag
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self.debug = debug
        self.dicom_dir = Path(dicom_dir)
        self._reader = reader
        self._walker = walker or (lambda p: (f for f in p.iterdir() if f.is_file()))

        # All files in the directory (tests expect this attribute name)
        self._files: list[Path] = list(self._walker(self.dicom_dir))
        logger.debug(f"Found {len(self._files)} files in {self.dicom_dir}")

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
        self.region_names: dict[int, str] = self._extract_roi_names()

        # Pre-index ROIContourSequence by ReferencedROINumber
        self._roi_by_number: dict[int, FileDataset] = self._index_roi_contours()

        # Lazy cache of ROI name -> RegionOfInterest
        self._roi_cache: dict[str, RegionOfInterest] = {}

    @property
    def roi_names(self) -> list[str]:
        return list(self.region_names.values())

    @property
    def regions(self) -> dict[str, RegionOfInterest]:
        if self._roi_cache:
            return dict(self._roi_cache)

        roi_contours = getattr(self.structure, "ROIContourSequence", None)
        if not roi_contours:
            logger.debug("No ROIContourSequence found in RTSTRUCT")
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

            if name in self._roi_cache:
                continue

            logger.debug(f"Loading ROI '{name}'")
            region = RegionOfInterest.from_rt_roi(
                roi_ds=roi,
                name=name,
                bins=4096,
                ct_index=self.ct_index,
            )
            self._roi_cache[name] = region

        return dict(self._roi_cache)

    def get_roi(self, name: str) -> RegionOfInterest:
        if name not in self.roi_names:
            raise KeyError(f"ROI '{name}' not found. Available: {self.roi_names}")

        if name in self._roi_cache:
            return self._roi_cache[name]

        logger.debug(f"Loading ROI '{name}' on demand")
        roi = self._load_single_roi(name)
        self._roi_cache[name] = roi
        return roi

    def dump_ply(
        self,
        directory: str | Path = ".",
        names: Iterable[str] | None = None,
        export_nifti: bool = False,
    ) -> None:
        from dicom2ply.ply_writer import write_roi_ply

        if names is None:
            names = self.roi_names

        selected = list(names)
        if not selected:
            raise ValueError("No ROIs found in RTSTRUCT or no names provided.")

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name in selected:
            logger.info(f"Exporting ROI '{name}' to PLY")
            roi = self.get_roi(name)
            write_roi_ply(roi, output_dir)

            if export_nifti:
                logger.info(f"Exporting NIfTI for ROI '{name}'")
                roi.export_nifti(output_dir / f"{name}.nii.gz")

    def _read_all_datasets(self) -> dict[Path, FileDataset]:
        datasets: dict[Path, FileDataset] = {}
        for path in self._files:
            try:
                ds = self._reader(path, stop_before_pixels=True)
            except Exception:
                logger.debug(f"Skipping unreadable DICOM file: {path}")
                continue
            datasets[path] = ds
        logger.debug(f"Loaded {len(datasets)} readable DICOM files")
        return datasets

    def _load_rtstruct(self) -> FileDataset:
        for path in self._files:
            ds = self._datasets.get(path)
            if ds is None:
                continue

            if getattr(ds, "Modality", None) == "RTSTRUCT":
                logger.debug(f"RTSTRUCT found: {path}")
                return ds

        raise FileNotFoundError("No RTSTRUCT file found (Modality=RTSTRUCT).")

    def _index_ct_slices(self) -> dict[str, CTSlice]:
        from dicom2ply.geometry import slice_position

        slices: list[CTSlice] = []

        rows = cols = None
        spacing = None
        orientation = None

        for path in self._files:
            ds = self._datasets.get(path)
            if ds is None:
                logger.debug(f"No dataset cached for: {path}")
                continue

            if getattr(ds, "Modality", None) != "CT":
                continue

            sop = getattr(ds, "SOPInstanceUID", None)
            if sop is None:
                logger.debug(f"CT slice without SOPInstanceUID: {path}")
                continue

            try:
                r = int(ds.Rows)
                c = int(ds.Columns)
                px = float(ds.PixelSpacing[0])
                py = float(ds.PixelSpacing[1])
                iop = tuple(float(v) for v in ds.ImageOrientationPatient)
            except Exception:
                logger.debug(f"Missing geometry metadata in: {path}")
                continue

            if rows is None:
                rows, cols = r, c
                spacing = (px, py)
                orientation = (tuple(iop[:3]), tuple(iop[3:]))
                logger.debug(
                    f"Reference CT geometry: {rows}x{cols}, spacing={spacing}, orientation={orientation}"
                )
            else:
                if (r, c) != (rows, cols):
                    logger.debug(
                        f"Inconsistent CT dimensions: {(r, c)} vs {(rows, cols)} in {path}"
                    )
                if spacing and (px, py) != spacing:
                    logger.debug(
                        f"Inconsistent CT pixel spacing: {(px, py)} vs {spacing} in {path}"
                    )
                if orientation and (tuple(iop[:3]), tuple(iop[3:])) != orientation:
                    logger.debug(f"Inconsistent CT orientation in: {path}")

            try:
                z = slice_position(ds)
            except Exception:
                z = float(getattr(ds, "InstanceNumber", 0))
                logger.debug(f"Falling back to InstanceNumber for slice: {path}")

            slices.append(CTSlice(sop_uid=str(sop), path=path, z=z))

        slices.sort(key=lambda s: s.z)
        logger.debug(f"Indexed {len(slices)} CT slices")

        return {s.sop_uid: s for s in slices}

    def _extract_roi_names(self) -> dict[int, str]:
        names: dict[int, str] = {}

        seq = getattr(self.structure, "RTROIObservationsSequence", [])
        for obs in seq:
            try:
                number = int(obs.ObservationNumber)
            except Exception:
                continue
            label = str(getattr(obs, "ROIObservationLabel", "")).strip()
            if label:
                names[number] = label

        if not names:
            seq2 = getattr(self.structure, "StructureSetROISequence", [])
            for roi in seq2:
                try:
                    number = int(roi.ROINumber)
                except Exception:
                    continue
                label = str(getattr(roi, "ROIName", "")).strip()
                if label:
                    names[number] = label

        logger.debug(f"Extracted ROI names: {names}")
        return names

    def _index_roi_contours(self) -> dict[int, FileDataset]:
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

        logger.debug(f"Indexed {len(index)} ROIContourSequence entries")
        return index

    def _load_single_roi(self, name: str) -> RegionOfInterest:
        roi_number = None
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

        logger.debug(f"Loading ROI '{name}' from RTSTRUCT")
        return RegionOfInterest.from_rt_roi(
            roi_ds=roi_ds,
            name=name,
            bins=4096,
            ct_index=self.ct_index,
        )
