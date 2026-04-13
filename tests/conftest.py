import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset


@pytest.fixture
def synthetic_ct(tmp_path):
    """Create a minimal synthetic CT slice with correct geometry."""
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
    file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9"

    ds = FileDataset(
        filename_or_obj=str(tmp_path / "CT1.dcm"),
        dataset={},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    ds.Modality = "CT"
    ds.SOPInstanceUID = "1.2.3.4.5"
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 10
    ds.Columns = 10
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1

    arr = np.arange(100, dtype=np.int16).reshape(10, 10)
    ds.PixelData = arr.tobytes()

    ds.save_as(ds.filename)
    return ds.filename


@pytest.fixture
def synthetic_rtstruct(tmp_path):
    """Create a minimal RTSTRUCT with one ROI and one contour."""
    ds = FileDataset(
        filename_or_obj=str(tmp_path / "RS1.dcm"),
        dataset={},
        file_meta=FileMetaDataset(),
        preamble=b"\0" * 128,
    )

    ds.Modality = "RTSTRUCT"

    obs = Dataset()
    obs.ROIObservationLabel = "TestROI"
    obs.ObservationNumber = 1
    ds.RTROIObservationsSequence = [obs]

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
    img.ReferencedSOPInstanceUID = "1.2.3.4.5"
    contour.ContourImageSequence = [img]

    roi = Dataset()
    roi.ReferencedROINumber = 1
    roi.ContourSequence = [contour]

    ds.ROIContourSequence = [roi]

    ds.save_as(ds.filename)
    return ds.filename
