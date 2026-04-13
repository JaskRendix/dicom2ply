import shutil
import subprocess

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset


def test_cli_runs(tmp_path, synthetic_ct, synthetic_rtstruct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    shutil.copy(synthetic_ct, dicom_dir / "CT.dcm")
    shutil.copy(synthetic_rtstruct, dicom_dir / "RS.dcm")

    output = tmp_path / "out"
    output.mkdir()

    result = subprocess.run(
        ["dicom2ply", str(dicom_dir), str(output)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert list(output.glob("*.ply")), "No PLY files were written"


def test_cli_roi_filter(tmp_path, synthetic_ct, synthetic_rtstruct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    shutil.copy(synthetic_ct, dicom_dir / "CT.dcm")
    shutil.copy(synthetic_rtstruct, dicom_dir / "RS.dcm")

    output = tmp_path / "out"
    output.mkdir()

    result = subprocess.run(
        ["dicom2ply", str(dicom_dir), str(output), "--names", "TestROI"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0

    files = list(output.glob("*.ply"))
    assert len(files) == 1
    assert "TestROI" in files[0].name


def test_cli_missing_rtstruct(tmp_path, synthetic_ct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    shutil.copy(synthetic_ct, dicom_dir / "CT.dcm")

    output = tmp_path / "out"
    output.mkdir()

    result = subprocess.run(
        ["dicom2ply", str(dicom_dir), str(output)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Error" in result.stderr or "Error" in result.stdout
    assert not list(output.glob("*.ply"))


def test_cli_default_exports_all_rois(tmp_path, synthetic_ct, synthetic_rtstruct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    shutil.copy(synthetic_ct, dicom_dir / "CT.dcm")
    shutil.copy(synthetic_rtstruct, dicom_dir / "RS.dcm")

    output = tmp_path / "out"
    output.mkdir()

    result = subprocess.run(
        ["dicom2ply", str(dicom_dir), str(output)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0

    files = list(output.glob("*.ply"))
    assert files, "Expected at least one PLY file when no --names is provided"

    assert len(files) == 1
    assert "TestROI" in files[0].name


def test_cli_invalid_roi_name(tmp_path, synthetic_ct, synthetic_rtstruct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    shutil.copy(synthetic_ct, dicom_dir / "CT.dcm")
    shutil.copy(synthetic_rtstruct, dicom_dir / "RS.dcm")

    output = tmp_path / "out"
    output.mkdir()

    result = subprocess.run(
        ["dicom2ply", str(dicom_dir), str(output), "--names", "DoesNotExist"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Error" in result.stdout or "Error" in result.stderr
    assert not list(output.glob("*.ply"))


def test_cli_missing_ct_slice(tmp_path, synthetic_rtstruct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    shutil.copy(synthetic_rtstruct, dicom_dir / "RS.dcm")

    output = tmp_path / "out"
    output.mkdir()

    result = subprocess.run(
        ["dicom2ply", str(dicom_dir), str(output)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Error" in result.stdout or "Error" in result.stderr
    assert not list(output.glob("*.ply"))


def test_cli_empty_rtstruct(tmp_path, synthetic_ct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    shutil.copy(synthetic_ct, dicom_dir / "CT.dcm")

    ds = FileDataset(
        filename_or_obj=str(tmp_path / "RS_empty.dcm"),
        dataset={},
        file_meta=FileMetaDataset(),
        preamble=b"\0" * 128,
    )
    ds.Modality = "RTSTRUCT"
    ds.RTROIObservationsSequence = []
    ds.ROIContourSequence = []
    ds.save_as(str(tmp_path / "RS_empty.dcm"))

    shutil.copy(tmp_path / "RS_empty.dcm", dicom_dir / "RS.dcm")

    output = tmp_path / "out"
    output.mkdir()

    result = subprocess.run(
        ["dicom2ply", str(dicom_dir), str(output)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Error" in result.stdout or "Error" in result.stderr
    assert not list(output.glob("*.ply"))


def test_cli_multiple_rois(tmp_path, synthetic_ct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    shutil.copy(synthetic_ct, dicom_dir / "CT.dcm")

    ds = FileDataset(
        filename_or_obj=str(tmp_path / "RS_multi.dcm"),
        dataset={},
        file_meta=FileMetaDataset(),
        preamble=b"\0" * 128,
    )
    ds.Modality = "RTSTRUCT"

    obs1 = Dataset()
    obs1.ROIObservationLabel = "ROI_A"
    obs1.ObservationNumber = 1

    obs2 = Dataset()
    obs2.ROIObservationLabel = "ROI_B"
    obs2.ObservationNumber = 2

    ds.RTROIObservationsSequence = [obs1, obs2]

    def make_contour():
        c = Dataset()
        c.ContourData = [2, 2, 0, 7, 2, 0, 7, 7, 0, 2, 7, 0]
        img = Dataset()
        img.ReferencedSOPInstanceUID = "1.2.3.4.5"
        c.ContourImageSequence = [img]
        return c

    roi1 = Dataset()
    roi1.ReferencedROINumber = 1
    roi1.ContourSequence = [make_contour()]

    roi2 = Dataset()
    roi2.ReferencedROINumber = 2
    roi2.ContourSequence = [make_contour()]

    ds.ROIContourSequence = [roi1, roi2]
    ds.save_as(str(tmp_path / "RS_multi.dcm"))

    shutil.copy(tmp_path / "RS_multi.dcm", dicom_dir / "RS.dcm")

    output = tmp_path / "out"
    output.mkdir()

    result = subprocess.run(
        ["dicom2ply", str(dicom_dir), str(output)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)

    assert result.returncode == 0

    files = list(output.glob("*.ply"))
    assert len(files) == 2
    assert any("ROI_A" in f.name for f in files)
    assert any("ROI_B" in f.name for f in files)


def test_cli_multiple_ct_slices(tmp_path, synthetic_rtstruct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    for i, z in enumerate([0.0, 1.0]):
        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage

        if i == 0:
            sop = "1.2.3.4.5"
        else:
            sop = f"1.2.3.4.5.{i}"

        file_meta.MediaStorageSOPInstanceUID = sop
        file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9"

        ds = FileDataset(
            filename_or_obj=str(tmp_path / f"CT{i}.dcm"),
            dataset={},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )
        ds.Modality = "CT"
        ds.SOPInstanceUID = sop
        ds.ImagePositionPatient = [0.0, 0.0, z]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.Rows = 10
        ds.Columns = 10
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SamplesPerPixel = 1
        ds.PixelData = np.arange(100, dtype=np.int16).reshape(10, 10).tobytes()

        ds.save_as(ds.filename)
        shutil.copy(ds.filename, dicom_dir / f"CT{i}.dcm")

    shutil.copy(synthetic_rtstruct, dicom_dir / "RS.dcm")

    output = tmp_path / "out"
    output.mkdir()

    result = subprocess.run(
        ["dicom2ply", str(dicom_dir), str(output)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)

    assert result.returncode == 0
    assert list(output.glob("*.ply"))
