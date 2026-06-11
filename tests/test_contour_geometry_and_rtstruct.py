from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

from dicom2ply.cli import validate_paths
from dicom2ply.patient import Patient


def _meta():
    m = FileMetaDataset()
    m.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    return m


def create_ct(path: Path, uid: str, pos, orient, spacing=(1.0, 1.0)):
    ds = FileDataset(str(path), {}, file_meta=_meta(), preamble=b"\0" * 128)
    ds.Modality = "CT"
    ds.SOPInstanceUID = uid
    ds.ImagePositionPatient = list(map(float, pos))
    ds.ImageOrientationPatient = list(map(float, orient))
    ds.PixelSpacing = list(map(float, spacing))
    ds.SliceThickness = 1.0
    ds.Rows = 10
    ds.Columns = 10
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = np.ones((10, 10), dtype=np.int16).tobytes()
    ds.save_as(str(path))
    return str(path)


def create_rtstruct(path: Path, roi_name, contours):
    ds = FileDataset(str(path), {}, file_meta=_meta(), preamble=b"\0" * 128)
    ds.Modality = "RTSTRUCT"

    obs = Dataset()
    obs.ROIObservationLabel = roi_name
    obs.ObservationNumber = 1
    ds.RTROIObservationsSequence = [obs]

    roi = Dataset()
    roi.ReferencedROINumber = 1
    roi.ContourSequence = []

    for uid, pts in contours:
        c = Dataset()
        c.ContourData = list(map(float, pts))
        img = Dataset()
        img.ReferencedSOPInstanceUID = uid
        c.ContourImageSequence = [img]
        roi.ContourSequence.append(c)

    ds.ROIContourSequence = [roi]
    ds.save_as(str(path))
    return str(path)


def test_non_coplanar_contour_raises(tmp_path):
    dicom = tmp_path / "d"
    dicom.mkdir()

    create_ct(
        dicom / "CT.dcm",
        "1.2.3.4",
        pos=(0, 0, 0),
        orient=(1, 0, 0, 0, 1, 0),
    )

    pts = [
        2,
        2,
        0,
        7,
        2,
        0,
        7,
        7,
        5,  # off-plane
    ]

    create_rtstruct(dicom / "RS.dcm", "NonPlanar", contours=[("1.2.3.4", pts)])

    with pytest.raises(ValueError):
        Patient(str(dicom), debug=False).get_roi("NonPlanar")


def test_small_z_drift_tolerated(tmp_path):
    dicom = tmp_path / "d"
    dicom.mkdir()

    create_ct(
        dicom / "CT.dcm",
        "1.2.3.4",
        pos=(0, 0, 0),
        orient=(1, 0, 0, 0, 1, 0),
    )

    pts = [
        2,
        2,
        0.005,
        7,
        2,
        0.005,
        7,
        7,
        0.005,
    ]

    create_rtstruct(dicom / "RS.dcm", "Drift", contours=[("1.2.3.4", pts)])

    p = Patient(str(dicom), debug=False)
    roi = p.get_roi("Drift")

    # Should not raise
    _ = roi.contours[0].voxel_points


def test_multi_slice_contour_rejected(tmp_path):
    dicom = tmp_path / "d"
    dicom.mkdir()

    create_ct(dicom / "CT1.dcm", "1.2.3.1", pos=(0, 0, 0), orient=(1, 0, 0, 0, 1, 0))
    create_ct(dicom / "CT2.dcm", "1.2.3.2", pos=(0, 0, 1), orient=(1, 0, 0, 0, 1, 0))

    create_rtstruct(
        dicom / "RS.dcm",
        "MultiSlice",
        contours=[
            ("1.2.3.1", [2, 2, 0, 7, 2, 0]),
            ("1.2.3.2", [2, 2, 1, 7, 2, 1]),
        ],
    )

    p = Patient(str(dicom), debug=False)
    roi = p.get_roi("MultiSlice")

    assert roi.contours == []  # all skipped


def test_oblique_slice_orientation(tmp_path):
    dicom = tmp_path / "d"
    dicom.mkdir()

    create_ct(
        dicom / "CT.dcm",
        "1.2.3.4",
        pos=(10, 20, 30),
        orient=(0.866, 0.5, 0, -0.5, 0.866, 0),
    )

    create_rtstruct(
        dicom / "RS.dcm", "Oblique", contours=[("1.2.3.4", [10, 20, 30, 12, 22, 30])]
    )

    p = Patient(str(dicom), debug=False)
    roi = p.get_roi("Oblique")

    # Either empty or valid — but must not crash
    if roi.contours:
        _ = roi.contours[0].voxel_points


def test_missing_contour_image_sequence_skipped(tmp_path):
    dicom = tmp_path / "d"
    dicom.mkdir()

    create_ct(dicom / "CT.dcm", "1.2.3.4", pos=(0, 0, 0), orient=(1, 0, 0, 0, 1, 0))

    ds = FileDataset(str(dicom / "RS.dcm"), {}, file_meta=_meta(), preamble=b"\0" * 128)
    ds.Modality = "RTSTRUCT"

    obs = Dataset()
    obs.ROIObservationLabel = "Bad"
    obs.ObservationNumber = 1
    ds.RTROIObservationsSequence = [obs]

    roi = Dataset()
    roi.ReferencedROINumber = 1

    c = Dataset()
    c.ContourData = [2, 2, 0, 7, 2, 0]
    roi.ContourSequence = [c]  # missing ContourImageSequence

    ds.ROIContourSequence = [roi]
    ds.save_as(str(dicom / "RS.dcm"))

    with pytest.raises(AttributeError):
        Patient(str(dicom), debug=False).regions


def test_validate_paths_missing_input(tmp_path):
    missing = tmp_path / "does_not_exist"
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        validate_paths(missing, out)
