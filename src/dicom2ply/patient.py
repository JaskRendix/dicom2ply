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
        bins: int = 4096,
        slice_tol: float = 0.01,
        planarity_tol: float = 1e-2,
        validate: bool = False,
        strict_geometry: bool = False,
        *,
        reader: Callable[..., FileDataset] = pydicom.dcmread,
        walker: Callable[[Path], Iterable[Path]] | None = None,
    ) -> None:

        self._configure_logging(debug)

        # Store parameters
        self.debug = debug
        self.bins = bins
        self.slice_tol = slice_tol
        self.planarity_tol = planarity_tol
        self.validate = validate
        self.strict_geometry = strict_geometry

        self.dicom_dir = Path(dicom_dir)
        self._reader = reader
        self._walker = walker or (lambda p: (f for f in p.iterdir() if f.is_file()))

        # Load files and datasets
        self._files = self._load_files()
        self._datasets = self._read_all_datasets()

        # Load RTSTRUCT
        self.structure = self._load_rtstruct()

        # CT slice indexing
        self._ct_slices = self._index_ct_slices()

        # Tests expect these exact attributes
        self.ct_slices = {uid: s.path.as_posix() for uid, s in self._ct_slices.items()}
        self.ct_index = dict(self.ct_slices)

        # ROI metadata
        self.region_names = self._extract_roi_names()
        self._roi_by_number = self._index_roi_contours()

        # Lazy ROI cache
        self._roi_cache: dict[str, RegionOfInterest] = {}

    def _configure_logging(self, debug: bool) -> None:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def _load_files(self) -> list[Path]:
        files = list(self._walker(self.dicom_dir))
        logger.debug(f"Found {len(files)} files in {self.dicom_dir}")
        return files

    def _read_all_datasets(self) -> dict[Path, FileDataset]:
        datasets = {}
        for path in self._files:
            try:
                ds = self._reader(path, stop_before_pixels=True)
                datasets[path] = ds
            except Exception:
                logger.debug(f"Skipping unreadable DICOM file: {path}")
        logger.debug(f"Loaded {len(datasets)} readable DICOM files")
        return datasets

    def _load_rtstruct(self) -> FileDataset:
        for path, ds in self._datasets.items():
            if getattr(ds, "Modality", None) == "RTSTRUCT":
                logger.debug(f"RTSTRUCT found: {path}")
                return ds
        raise FileNotFoundError("No RTSTRUCT file found (Modality=RTSTRUCT).")

    def _index_ct_slices(self) -> dict[str, CTSlice]:
        from dicom2ply.geometry import slice_position

        slices = []
        ref_dims = None
        ref_spacing = None
        ref_orientation = None

        inconsistent_dims = False
        inconsistent_spacing = False
        inconsistent_orientation = False

        for path, ds in self._iter_ct_datasets():
            sop = getattr(ds, "SOPInstanceUID", None)
            if sop is None:
                logger.debug(f"CT slice without SOPInstanceUID: {path}")
                continue

            geom = self._extract_ct_geometry(ds, path)
            if geom is None:
                continue

            dims, spacing, orientation = geom

            # First slice defines reference geometry
            if ref_dims is None:
                ref_dims = dims
                ref_spacing = spacing
                ref_orientation = orientation
                logger.debug(
                    f"Reference CT geometry: {dims[0]}x{dims[1]}, "
                    f"spacing={spacing}, orientation={orientation}"
                )
            else:
                inconsistent_dims |= dims != ref_dims
                inconsistent_spacing |= spacing != ref_spacing
                inconsistent_orientation |= orientation != ref_orientation

            # Compute Z position
            z = self._compute_slice_z(ds, path)
            slices.append(CTSlice(str(sop), path, z))

        # Sort slices by Z
        slices.sort(key=lambda s: s.z)
        logger.debug(f"Indexed {len(slices)} CT slices")

        # Log geometry summary
        self._log_ct_summary(
            slices,
            ref_dims,
            ref_spacing,
            ref_orientation,
            inconsistent_dims,
            inconsistent_spacing,
            inconsistent_orientation,
        )

        # Strict geometry enforcement
        if self.strict_geometry and (
            inconsistent_dims or inconsistent_spacing or inconsistent_orientation
        ):
            raise RuntimeError("CT geometry inconsistent across slices (strict mode)")

        return {s.sop_uid: s for s in slices}

    def _iter_ct_datasets(self):
        for path, ds in self._datasets.items():
            if getattr(ds, "Modality", None) == "CT":
                yield path, ds

    def _extract_ct_geometry(self, ds, path):
        try:
            dims = (int(ds.Rows), int(ds.Columns))
            spacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
            iop = tuple(float(v) for v in ds.ImageOrientationPatient)
            orientation = (tuple(iop[:3]), tuple(iop[3:]))
            return dims, spacing, orientation
        except Exception:
            logger.debug(f"Missing geometry metadata in: {path}")
            return None

    def _compute_slice_z(self, ds, path):
        from dicom2ply.geometry import slice_position

        try:
            return slice_position(ds)
        except Exception:
            logger.debug(f"Falling back to InstanceNumber for slice: {path}")
            return float(getattr(ds, "InstanceNumber", 0))

    def _log_ct_summary(
        self,
        slices,
        dims,
        spacing,
        orientation,
        inconsistent_dims,
        inconsistent_spacing,
        inconsistent_orientation,
    ):
        if not (self.debug or self.validate):
            return

        z_positions = [s.z for s in slices]
        diffs = (
            [
                abs(z_positions[i + 1] - z_positions[i])
                for i in range(len(z_positions) - 1)
            ]
            if len(z_positions) > 1
            else []
        )

        logger.info("CT Geometry Summary:")
        logger.info(f"  slices: {len(slices)}")
        logger.info(f"  dimensions: {dims}")
        logger.info(f"  pixel_spacing: {spacing}")

        if diffs:
            logger.info(
                f"  slice_spacing (min/mean/max): "
                f"{min(diffs):.4f}/{(sum(diffs)/len(diffs)):.4f}/{max(diffs):.4f}"
            )

        logger.info(f"  orientation_consistent: {not inconsistent_orientation}")

    def _extract_roi_names(self) -> dict[int, str]:
        names = {}

        # Primary: RTROIObservationsSequence
        for obs in getattr(self.structure, "RTROIObservationsSequence", []):
            try:
                number = int(obs.ObservationNumber)
                label = str(getattr(obs, "ROIObservationLabel", "")).strip()
                if label:
                    names[number] = label
            except Exception:
                continue

        # Fallback: StructureSetROISequence
        if not names:
            for roi in getattr(self.structure, "StructureSetROISequence", []):
                try:
                    number = int(roi.ROINumber)
                    label = str(getattr(roi, "ROIName", "")).strip()
                    if label:
                        names[number] = label
                except Exception:
                    continue

        logger.debug(f"Extracted ROI names: {names}")
        return names

    def _index_roi_contours(self) -> dict[int, FileDataset]:
        index = {}
        for roi in getattr(self.structure, "ROIContourSequence", []):
            number = getattr(roi, "ReferencedROINumber", None)
            try:
                index[int(number)] = roi
            except Exception:
                continue
        logger.debug(f"Indexed {len(index)} ROIContourSequence entries")
        return index

    @property
    def roi_names(self) -> list[str]:
        return list(self.region_names.values())

    @property
    def regions(self) -> dict[str, RegionOfInterest]:
        if self._roi_cache:
            return dict(self._roi_cache)

        for number, name in self.region_names.items():
            roi_ds = self._roi_by_number.get(number)
            if roi_ds and hasattr(roi_ds, "ContourSequence"):
                self._roi_cache[name] = RegionOfInterest.from_rt_roi(
                    roi_ds=roi_ds,
                    name=name,
                    bins=self.bins,
                    ct_index=self.ct_index,
                    slice_tol=self.slice_tol,
                    planarity_tol=self.planarity_tol,
                )

        return dict(self._roi_cache)

    def get_roi(self, name: str) -> RegionOfInterest:
        if name not in self.roi_names:
            raise KeyError(f"ROI '{name}' not found. Available: {self.roi_names}")

        if name in self._roi_cache:
            return self._roi_cache[name]

        roi = self._load_single_roi(name)
        self._roi_cache[name] = roi
        return roi

    def _load_single_roi(self, name: str) -> RegionOfInterest:
        # Resolve name -> number
        roi_number = next(
            (num for num, label in self.region_names.items() if label == name),
            None,
        )
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
            bins=self.bins,
            ct_index=self.ct_index,
            slice_tol=self.slice_tol,
            planarity_tol=self.planarity_tol,
        )

    def dump_ply(
        self,
        directory: str | Path = ".",
        names: Iterable[str] | None = None,
        export_nifti: bool = False,
    ) -> None:
        from dicom2ply.ply_writer import write_roi_ply

        names = list(names or self.roi_names)
        if not names:
            raise ValueError("No ROIs found in RTSTRUCT or no names provided.")

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, name in enumerate(names):
            logger.info(f"[{i+1}/{len(names)}] Exporting ROI '{name}' to PLY")
            roi = self.get_roi(name)
            write_roi_ply(roi, output_dir)

            if export_nifti:
                logger.info(f"Exporting NIfTI for ROI '{name}'")
                roi.export_nifti(output_dir / f"{name}.nii.gz")
