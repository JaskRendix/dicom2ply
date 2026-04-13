import os

import pydicom

from dicom2ply.ct_cache import CTSliceCache
from dicom2ply.roi import RegionOfInterest


class Patient:
    def __init__(self, dicom_dir: str, debug: bool = True) -> None:
        self.debug = debug
        self.dicom_dir = dicom_dir

        _, _, files = next(os.walk(self.dicom_dir))
        self.files = files

        rtstruct_path = self._find_rtstruct()
        self.structure = pydicom.dcmread(rtstruct_path)

        self.ct_slices = self._index_ct_slices()
        self.ct_index = self.ct_slices

        self.region_names = self._extract_roi_names()
        self.regions = self._load_rois()

    def _find_rtstruct(self) -> str:
        for f in self.files:
            if f.startswith("RS"):
                return os.path.join(self.dicom_dir, f)
        raise FileNotFoundError("No RTSTRUCT (RS*) file found in directory")

    def _index_ct_slices(self) -> dict[str, str]:
        index = {}
        for f in self.files:
            path = os.path.join(self.dicom_dir, f)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception:
                continue
            if getattr(ds, "Modality", None) == "CT":
                sop = getattr(ds, "SOPInstanceUID", None)
                if sop:
                    index[sop] = path
        return index

    def _extract_roi_names(self) -> dict[int, str]:
        names = {}
        for obs in self.structure.RTROIObservationsSequence:
            names[obs.ObservationNumber] = obs.ROIObservationLabel
        return names

    def _load_rois(self) -> dict[str, RegionOfInterest]:
        cache = CTSliceCache(self.ct_index)
        regions = {}

        for roi in self.structure.ROIContourSequence:
            number = roi.ReferencedROINumber
            name = self.region_names.get(number)
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
            regions[name] = region

        return regions

    def dump_ply(self, directory=".", names=None) -> None:
        if names is None:
            names = list(self.regions.keys())

        if not names:
            raise ValueError("No ROIs found in RTSTRUCT")

        from dicom2ply.ply_writer import write_roi_ply

        for name in names:
            write_roi_ply(self.regions[name], directory)
