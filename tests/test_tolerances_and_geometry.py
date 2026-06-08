import os
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

from dicom2ply.cli import validate_paths
from dicom2ply.patient import Patient


def create_ct(path: Path, uid: str, z: float):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = uid

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality = "CT"
    ds.SOPInstanceUID = uid
    ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.Rows = 10
    ds.Columns = 10
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    arr = np.ones((10, 10), dtype=np.int16)
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path))
    return str(path)


def create_rtstruct(path: Path, roi_name="TestROI", ref_uid=None):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality = "RTSTRUCT"

    obs = Dataset()
    obs.ROIObservationLabel = roi_name
    obs.ObservationNumber = 1
    ds.RTROIObservationsSequence = [obs]

    if ref_uid is not None:
        roi = Dataset()
        roi.ReferencedROINumber = 1

        contour = Dataset()
        contour.ContourData = [
            2.0,
            2.0,
            0.0,
            7.0,
            2.0,
            0.0,
            7.0,
            7.0,
            0.0,
            2.0,
            7.0,
            0.0,
        ]
        img = Dataset()
        img.ReferencedSOPInstanceUID = ref_uid
        contour.ContourImageSequence = [img]

        roi.ContourSequence = [contour]
        ds.ROIContourSequence = [roi]
    else:
        ds.ROIContourSequence = []

    ds.save_as(str(path))
    return str(path)


def test_duplicate_ct_slice_positions(tmp_path):
    # Create two CT slices with the same z
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    ct1 = create_ct(dicom_dir / "CT1.dcm", "1.2.3.4.1", 0.0)
    ct2 = create_ct(dicom_dir / "CT2.dcm", "1.2.3.4.2", 0.0)

    # RTSTRUCT referencing one of them
    rs = create_rtstruct(dicom_dir / "RS.dcm", ref_uid="1.2.3.4.1")

    p = Patient(str(dicom_dir), debug=True)
    assert len(p.ct_slices) == 2
    assert "1.2.3.4.1" in p.ct_slices and "1.2.3.4.2" in p.ct_slices


def test_clipped_contours_emit_warning(tmp_path, caplog):
    # Create CT and RT with contour outside bounds
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    ct = create_ct(dicom_dir / "CT.dcm", "1.2.3.4.5", 0.0)

    # RTSTRUCT with a contour partially outside the image
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = FileDataset(
        str(dicom_dir / "RS.dcm"), {}, file_meta=meta, preamble=b"\0" * 128
    )
    ds.Modality = "RTSTRUCT"
    obs = Dataset()
    obs.ROIObservationLabel = "ClipROI"
    obs.ObservationNumber = 1
    ds.RTROIObservationsSequence = [obs]

    roi = Dataset()
    roi.ReferencedROINumber = 1
    contour = Dataset()
    contour.ContourData = [-10.0, -10.0, 0.0, 50.0, 50.0, 0.0]
    img = Dataset()
    img.ReferencedSOPInstanceUID = "1.2.3.4.5"
    contour.ContourImageSequence = [img]
    roi.ContourSequence = [contour]
    ds.ROIContourSequence = [roi]
    ds.save_as(str(dicom_dir / "RS.dcm"))

    caplog.set_level("WARNING")

    p = Patient(str(dicom_dir), debug=False)
    # force load
    regions = p.regions
    found = False
    for r in regions.values():
        # Access contours to trigger computation
        for c in r.contours:
            pass
    # Look for clipping warning
    assert any("Clipping" in rec.message for rec in caplog.records)


def test_missing_structuresetroi_sequence(tmp_path):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    ct = create_ct(dicom_dir / "CT.dcm", "1.2.3.4.5", 0.0)

    # RTSTRUCT with RTROIObservationsSequence but no StructureSetROISequence
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = FileDataset(
        str(dicom_dir / "RS.dcm"), {}, file_meta=meta, preamble=b"\0" * 128
    )
    ds.Modality = "RTSTRUCT"
    obs = Dataset()
    obs.ROIObservationLabel = "OnlyObs"
    obs.ObservationNumber = 1
    ds.RTROIObservationsSequence = [obs]
    ds.ROIContourSequence = []
    ds.save_as(str(dicom_dir / "RS.dcm"))

    p = Patient(str(dicom_dir), debug=False)
    assert 1 in p.region_names
    assert "OnlyObs" in p.region_names.values()


def test_nifti_affine_matches(tmp_path):
    nib = pytest.importorskip("nibabel")

    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    ct = create_ct(dicom_dir / "CT.dcm", "1.2.3.4.5", 0.0)
    rs = create_rtstruct(dicom_dir / "RS.dcm", ref_uid="1.2.3.4.5")

    p = Patient(str(dicom_dir), debug=False)
    roi = p.get_roi("TestROI")

    out = tmp_path / "out.nii.gz"
    roi.export_nifti(out)

    img = nib.load(str(out))
    ds = roi.contours[0].ds
    spacing_z = roi._get_spacing_z(ds)
    expected = roi._get_affine(ds, spacing_z)

    assert np.allclose(img.affine, expected)


def test_validate_paths_invalid_output(tmp_path):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    # create input file
    (dicom_dir / "CT.dcm").write_text("x")

    # create a file where output dir should be, causing mkdir to fail
    output_file = tmp_path / "out"
    output_file.write_text("cannot be a dir")

    with pytest.raises(RuntimeError):
        validate_paths(dicom_dir, output_file)
