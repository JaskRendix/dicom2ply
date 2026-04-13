import shutil
from pathlib import Path

import pydicom

from dicom2ply.patient import Patient


def build_dicom_dir(tmp_path, ct_file, rt_file):
    """Copy synthetic CT + RTSTRUCT into a temporary DICOM directory."""
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    shutil.copy(ct_file, dicom_dir / Path(ct_file).name)
    shutil.copy(rt_file, dicom_dir / Path(rt_file).name)

    return dicom_dir


def test_patient_loads_rtstruct_and_ct(synthetic_ct, synthetic_rtstruct, tmp_path):
    dicom_dir = build_dicom_dir(tmp_path, synthetic_ct, synthetic_rtstruct)

    p = Patient(str(dicom_dir), debug=False)

    # RTSTRUCT loaded
    assert p.structure.Modality == "RTSTRUCT"

    # CT index built
    assert len(p.ct_slices) == 1
    assert list(p.ct_slices.values())[0].endswith(".dcm")

    # ROI names extracted
    assert "TestROI" in p.region_names.values()


def test_patient_builds_roi_objects(synthetic_ct, synthetic_rtstruct, tmp_path):
    dicom_dir = build_dicom_dir(tmp_path, synthetic_ct, synthetic_rtstruct)

    p = Patient(str(dicom_dir), debug=False)

    assert "TestROI" in p.regions
    roi = p.regions["TestROI"]

    assert roi.mean is not None
    assert roi.mask_stack is not None
    assert roi.mask_stack.sum() > 0


def test_patient_multiple_ct_slices(tmp_path, synthetic_rtstruct):
    import shutil

    import numpy as np
    from pydicom.dataset import FileDataset, FileMetaDataset

    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    ct_index = {}
    for z in [0, 5]:
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        meta.MediaStorageSOPInstanceUID = f"1.2.840.10008.5.1.4.1.1.2.{z}"

        ds = FileDataset(
            str(dicom_dir / f"CT{z}.dcm"),
            {},
            file_meta=meta,
            preamble=b"\0" * 128,
        )
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
        ds.save_as(ds.filename)

        ct_index[ds.SOPInstanceUID] = ds.filename

    rt = pydicom.dcmread(synthetic_rtstruct)

    first_uid = list(ct_index.keys())[0]
    rt.ROIContourSequence[0].ContourSequence[0].ContourImageSequence[
        0
    ].ReferencedSOPInstanceUID = first_uid

    rt_path = dicom_dir / "RS1.dcm"
    rt.save_as(rt_path)

    p = Patient(str(dicom_dir), debug=False)

    assert len(p.ct_slices) == 2
    assert "TestROI" in p.regions


def test_patient_skips_roi_without_contours(tmp_path, synthetic_ct):
    """RTSTRUCT with ROIObservation but no ROIContourSequence should not crash."""
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    shutil.copy(synthetic_ct, dicom_dir / "CT1.dcm")

    # Build minimal RTSTRUCT with no ROIContourSequence
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = FileDataset(
        str(dicom_dir / "RS1.dcm"),
        {},
        file_meta=meta,
        preamble=b"\0" * 128,
    )
    ds.Modality = "RTSTRUCT"

    obs = Dataset()
    obs.ROIObservationLabel = "EmptyROI"
    obs.ObservationNumber = 1
    ds.RTROIObservationsSequence = [obs]

    ds.ROIContourSequence = []  # no contours
    ds.save_as(ds.filename)

    p = Patient(str(dicom_dir), debug=False)

    # Should not crash, but ROI list should be empty
    assert p.regions == {}


def test_patient_dump_ply(tmp_path, synthetic_ct, synthetic_rtstruct, monkeypatch):
    """Ensure dump_ply calls write_roi_ply for each ROI."""
    dicom_dir = build_dicom_dir(tmp_path, synthetic_ct, synthetic_rtstruct)

    p = Patient(str(dicom_dir), debug=False)

    calls = []

    def fake_writer(roi, directory):
        calls.append(roi.name)

    monkeypatch.setattr("dicom2ply.ply_writer.write_roi_ply", fake_writer)

    p.dump_ply(directory=str(tmp_path))

    assert calls == ["TestROI"]
