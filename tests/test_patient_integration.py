import shutil
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

from dicom2ply.patient import Patient


def create_ct_slice(path: Path, z: float):
    """Create a synthetic CT slice at a given z-position."""
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = f"1.2.3.{z}"

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality = "CT"
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.ImagePositionPatient = [0, 0, float(z)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1, 1]
    ds.SliceThickness = 1
    ds.Rows = ds.Columns = 10
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = np.ones((10, 10), np.int16).tobytes()
    ds.save_as(path)
    return ds


def create_rtstruct(path: Path, roi_name="TestROI", ref_uid=None, with_contours=True):
    """Create a synthetic RTSTRUCT with optional contour data."""
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality = "RTSTRUCT"

    obs = Dataset()
    obs.ROIObservationLabel = roi_name
    obs.ObservationNumber = 1
    ds.RTROIObservationsSequence = [obs]

    if with_contours:
        roi = Dataset()
        roi.ReferencedROINumber = 1

        contour = Dataset()
        contour.ContourGeometricType = "POINT"
        contour.NumberOfContourPoints = 1
        contour.ContourData = [0.0, 0.0, 0.0]

        img_ref = Dataset()
        img_ref.ReferencedSOPInstanceUID = ref_uid or "1.2.3"
        contour.ContourImageSequence = [img_ref]

        roi.ContourSequence = [contour]
        ds.ROIContourSequence = [roi]
    else:
        ds.ROIContourSequence = []

    ds.save_as(path)
    return ds


def build_dicom_dir(tmp_path, ct_file, rt_file):
    """Copy synthetic CT + RTSTRUCT into a temporary DICOM directory."""
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    shutil.copy(ct_file, dicom_dir / Path(ct_file).name)
    shutil.copy(rt_file, dicom_dir / Path(rt_file).name)

    return dicom_dir


def test_patient_detects_rtstruct_and_ct(synthetic_ct, synthetic_rtstruct, tmp_path):
    """Patient should detect RTSTRUCT and index CT slices."""
    dicom_dir = build_dicom_dir(tmp_path, synthetic_ct, synthetic_rtstruct)

    p = Patient(str(dicom_dir), debug=False)

    # Behavior: RTSTRUCT is loaded
    assert p.structure.Modality == "RTSTRUCT"

    # Behavior: CT slices indexed
    assert len(p.ct_slices) == 1
    assert list(p.ct_slices.values())[0].endswith(".dcm")

    # Behavior: ROI names extracted
    assert "TestROI" in p.region_names.values()


def test_patient_builds_roi_objects(synthetic_ct, synthetic_rtstruct, tmp_path):
    """Patient should construct RegionOfInterest objects for valid ROIs."""
    dicom_dir = build_dicom_dir(tmp_path, synthetic_ct, synthetic_rtstruct)

    p = Patient(str(dicom_dir), debug=False)

    assert "TestROI" in p.regions
    roi = p.regions["TestROI"]

    assert roi.mean is not None
    assert roi.mask_stack is not None
    assert roi.mask_stack.sum() > 0


def test_patient_indexes_multiple_ct_slices_by_z(tmp_path, synthetic_rtstruct):
    """Patient should index multiple CT slices and sort them by z-position."""
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    # Create CT slices at z=0 and z=5
    ct0 = create_ct_slice(dicom_dir / "CT0.dcm", z=0)
    ct5 = create_ct_slice(dicom_dir / "CT5.dcm", z=5)

    # Patch RTSTRUCT to reference the first CT slice
    rt = pydicom.dcmread(synthetic_rtstruct)
    rt.ROIContourSequence[0].ContourSequence[0].ContourImageSequence[
        0
    ].ReferencedSOPInstanceUID = ct0.SOPInstanceUID
    rt.save_as(dicom_dir / "RS1.dcm")

    p = Patient(str(dicom_dir), debug=False)

    # Behavior: both slices indexed
    assert len(p.ct_slices) == 2

    # Behavior: ROI still loads
    assert "TestROI" in p.regions


def test_patient_skips_roi_without_contours(tmp_path, synthetic_ct):
    """Patient should not crash when ROI has no contours and should return empty regions."""
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    shutil.copy(synthetic_ct, dicom_dir / "CT1.dcm")

    # RTSTRUCT with no contours
    create_rtstruct(dicom_dir / "RS1.dcm", roi_name="EmptyROI", with_contours=False)

    p = Patient(str(dicom_dir), debug=False)

    # Behavior: no regions loaded
    assert p.regions == {}


def test_patient_dump_ply_invokes_writer(
    tmp_path, synthetic_ct, synthetic_rtstruct, monkeypatch
):
    """dump_ply should call write_roi_ply once per ROI."""
    dicom_dir = build_dicom_dir(tmp_path, synthetic_ct, synthetic_rtstruct)

    p = Patient(str(dicom_dir), debug=False)

    calls = []

    def fake_writer(roi, directory):
        calls.append(roi.name)

    monkeypatch.setattr("dicom2ply.ply_writer.write_roi_ply", fake_writer)

    p.dump_ply(directory=str(tmp_path))

    assert calls == ["TestROI"]


def test_patient_uses_structuresetroi_when_observations_missing(tmp_path, synthetic_ct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    shutil.copy(synthetic_ct, dicom_dir / "CT1.dcm")
    ct_ds = pydicom.dcmread(dicom_dir / "CT1.dcm")

    path = dicom_dir / "RS1.dcm"
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality = "RTSTRUCT"

    roi = Dataset()
    roi.ROINumber = 7
    roi.ROIName = "FallbackROI"
    ds.StructureSetROISequence = [roi]

    contour = Dataset()
    contour.ContourGeometricType = "POINT"
    contour.NumberOfContourPoints = 1
    contour.ContourData = [0.0, 0.0, 0.0]

    img_ref = Dataset()
    img_ref.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
    contour.ContourImageSequence = [img_ref]

    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = 7
    roi_contour.ContourSequence = [contour]
    ds.ROIContourSequence = [roi_contour]

    ds.save_as(path)

    p = Patient(str(dicom_dir), debug=False)

    assert "FallbackROI" in p.roi_names
    assert "FallbackROI" in p.regions


def test_patient_skips_ct_without_sop_uid(tmp_path):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    ct_path = dicom_dir / "CT_missing_uid.dcm"
    ds = create_ct_slice(ct_path, z=0)
    del ds.SOPInstanceUID
    ds.save_as(ct_path)

    create_rtstruct(dicom_dir / "RS1.dcm", ref_uid="1.2.3")

    p = Patient(str(dicom_dir), debug=False)

    assert p.ct_slices == {}


def test_patient_ct_missing_ipp_falls_back_to_instance_number(tmp_path):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    ct_path = dicom_dir / "CT1.dcm"
    ds = create_ct_slice(ct_path, z=0)
    del ds.ImagePositionPatient
    ds.InstanceNumber = 42
    ds.save_as(ct_path)

    create_rtstruct(dicom_dir / "RS1.dcm", ref_uid=ds.SOPInstanceUID)

    p = Patient(str(dicom_dir), debug=False)

    slice_obj = next(iter(p._ct_slices.values()))
    assert slice_obj.z == 42.0


def test_patient_roi_missing_contoursequence_skipped_in_regions(tmp_path, synthetic_ct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    shutil.copy(synthetic_ct, dicom_dir / "CT1.dcm")

    path = dicom_dir / "RS1.dcm"
    ds = create_rtstruct(path, roi_name="BadROI", with_contours=False)

    roi = Dataset()
    roi.ReferencedROINumber = 1
    ds.ROIContourSequence = [roi]
    ds.save_as(path)

    p = Patient(str(dicom_dir), debug=False)

    assert p.regions == {}
    with pytest.raises(KeyError):
        p.get_roi("BadROI")


def test_patient_mixed_valid_and_invalid_rois(tmp_path, synthetic_ct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    shutil.copy(synthetic_ct, dicom_dir / "CT1.dcm")
    ct_ds = pydicom.dcmread(dicom_dir / "CT1.dcm")

    path = dicom_dir / "RS1.dcm"
    ds = create_rtstruct(
        path,
        roi_name="ValidROI",
        ref_uid=ct_ds.SOPInstanceUID,
        with_contours=True,
    )

    obs2 = Dataset()
    obs2.ROIObservationLabel = "EmptyROI"
    obs2.ObservationNumber = 2
    ds.RTROIObservationsSequence.append(obs2)

    roi2 = Dataset()
    roi2.ReferencedROINumber = 2
    ds.ROIContourSequence.append(roi2)

    ds.save_as(path)

    p = Patient(str(dicom_dir), debug=False)

    assert "ValidROI" in p.regions
    assert "EmptyROI" not in p.regions


def test_patient_duplicate_roi_numbers_last_wins(tmp_path, synthetic_ct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    shutil.copy(synthetic_ct, dicom_dir / "CT1.dcm")
    ct_ds = pydicom.dcmread(dicom_dir / "CT1.dcm")

    path = dicom_dir / "RS1.dcm"
    ds = create_rtstruct(
        path,
        roi_name="FirstROI",
        ref_uid=ct_ds.SOPInstanceUID,
        with_contours=True,
    )

    obs2 = Dataset()
    obs2.ROIObservationLabel = "SecondROI"
    obs2.ObservationNumber = 1
    ds.RTROIObservationsSequence.append(obs2)

    roi2 = Dataset()
    roi2.ReferencedROINumber = 1
    roi2.ContourSequence = ds.ROIContourSequence[0].ContourSequence
    ds.ROIContourSequence.append(roi2)

    ds.save_as(path)

    p = Patient(str(dicom_dir), debug=False)

    assert p.roi_names == ["SecondROI"]
    assert "SecondROI" in p.regions
