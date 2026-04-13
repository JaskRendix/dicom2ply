import numpy as np
import pydicom
import pytest

from src.dicom2ply.contour import Contour
from src.dicom2ply.ct_cache import CTSliceCache
from src.dicom2ply.roi import RegionOfInterest


def test_oblique_slice_orientation(tmp_path):
    from pydicom.dataset import FileDataset, FileMetaDataset

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.840.10008.5.1.4.1.1.2.1"

    ds = FileDataset(
        str(tmp_path / "CT_oblique.dcm"),
        {},
        file_meta=meta,
        preamble=b"\0" * 128,
    )
    ds.Modality = "CT"
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.ImageOrientationPatient = [0.7071, 0.7071, 0, -0.7071, 0.7071, 0]
    ds.ImagePositionPatient = [0, 0, 0]
    ds.PixelSpacing = [1, 1]
    ds.SliceThickness = 1
    ds.Rows = ds.Columns = 10
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = np.arange(100, dtype=np.int16).reshape(10, 10).tobytes()
    ds.save_as(ds.filename)

    cache = CTSliceCache({ds.SOPInstanceUID: ds.filename})

    contour = pydicom.dataset.Dataset()
    contour.ContourData = [2, 2, 0, 7, 2, 0, 7, 7, 0, 2, 7, 0]
    img = pydicom.dataset.Dataset()
    img.ReferencedSOPInstanceUID = ds.SOPInstanceUID
    contour.ContourImageSequence = [img]

    c = Contour.from_rt(contour, bins=16, cache=cache)
    assert c.mask.sum() > 0


def test_rescale_slope_intercept(tmp_path):
    from pydicom.dataset import FileDataset, FileMetaDataset

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.840.10008.5.1.4.1.1.2.2"

    ds = FileDataset(
        str(tmp_path / "CT_hu.dcm"),
        {},
        file_meta=meta,
        preamble=b"\0" * 128,
    )
    ds.Modality = "CT"
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.ImagePositionPatient = [0, 0, 0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1, 1]
    ds.SliceThickness = 1
    ds.Rows = ds.Columns = 10
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1

    raw = np.arange(100, dtype=np.int16).reshape(10, 10)
    ds.PixelData = raw.tobytes()
    ds.RescaleSlope = 2
    ds.RescaleIntercept = 100
    ds.save_as(ds.filename)

    cache = CTSliceCache({ds.SOPInstanceUID: ds.filename})

    contour = pydicom.dataset.Dataset()
    contour.ContourData = [2, 2, 0, 7, 2, 0, 7, 7, 0, 2, 7, 0]
    img = pydicom.dataset.Dataset()
    img.ReferencedSOPInstanceUID = ds.SOPInstanceUID
    contour.ContourImageSequence = [img]

    c = Contour.from_rt(contour, bins=16, cache=cache)
    assert c.stats.mean > raw.mean()


def test_roi_with_slice_gaps(tmp_path, synthetic_rtstruct):
    from pydicom.dataset import FileDataset, FileMetaDataset

    ct_index = {}

    for z in [0, 2]:
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        meta.MediaStorageSOPInstanceUID = f"1.2.840.10008.5.1.4.1.1.2.{z}"

        ds = FileDataset(
            str(tmp_path / f"CT_gap_{z}.dcm"),
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

    def make_contour(z, uid):
        c = pydicom.dataset.Dataset()
        c.ContourData = [2, 2, z, 7, 2, z, 7, 7, z, 2, 7, z]
        img = pydicom.dataset.Dataset()
        img.ReferencedSOPInstanceUID = uid
        c.ContourImageSequence = [img]
        return c

    roi_ds = pydicom.dataset.Dataset()
    roi_ds.ContourSequence = [
        make_contour(0, list(ct_index.keys())[0]),
        make_contour(2, list(ct_index.keys())[1]),
    ]

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=roi_ds,
        name="GapROI",
        bins=16,
        ct_index=ct_index,
    )

    assert roi.mask_stack.shape[-1] == 2
    assert list(roi.slice_positions) == [0, 2]


def test_nested_contours_hole(synthetic_ct):
    ct_ds = pydicom.dcmread(synthetic_ct)

    def make_contour(coords):
        c = pydicom.dataset.Dataset()
        c.ContourData = coords
        img = pydicom.dataset.Dataset()
        img.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
        c.ContourImageSequence = [img]
        return c

    outer = make_contour([2, 2, 0, 8, 2, 0, 8, 8, 0, 2, 8, 0])
    inner = make_contour([4, 4, 0, 6, 4, 0, 6, 6, 0, 4, 6, 0])

    roi_ds = pydicom.dataset.Dataset()
    roi_ds.ContourSequence = [outer, inner]

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=roi_ds,
        name="HoleROI",
        bins=16,
        ct_index={ct_ds.SOPInstanceUID: synthetic_ct},
    )

    assert roi.mask_stack.sum() > (6 * 6)


def test_mixed_slice_thickness(tmp_path, synthetic_rtstruct):
    from pydicom.dataset import FileDataset, FileMetaDataset

    ct_index = {}

    for z, thick in [(0, 1), (1, 5)]:
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        meta.MediaStorageSOPInstanceUID = f"1.2.840.10008.5.1.4.1.1.2.{z}"

        ds = FileDataset(
            str(tmp_path / f"CT_thick_{z}.dcm"),
            {},
            file_meta=meta,
            preamble=b"\0" * 128,
        )
        ds.Modality = "CT"
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.ImagePositionPatient = [0, 0, float(z)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1, 1]
        ds.SliceThickness = thick
        ds.Rows = ds.Columns = 10
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelData = np.ones((10, 10), np.int16).tobytes()
        ds.save_as(ds.filename)
        ct_index[ds.SOPInstanceUID] = ds.filename

    def make_contour(z, uid):
        c = pydicom.dataset.Dataset()
        c.ContourData = [2, 2, z, 7, 2, z, 7, 7, z, 2, 7, z]
        img = pydicom.dataset.Dataset()
        img.ReferencedSOPInstanceUID = uid
        c.ContourImageSequence = [img]
        return c

    roi_ds = pydicom.dataset.Dataset()
    roi_ds.ContourSequence = [
        make_contour(0, list(ct_index.keys())[0]),
        make_contour(1, list(ct_index.keys())[1]),
    ]

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=roi_ds,
        name="ThickROI",
        bins=16,
        ct_index=ct_index,
    )

    assert list(roi.slice_positions) == [0, 1]


def test_duplicate_sopuid_different_z(synthetic_ct):
    ct_ds = pydicom.dcmread(synthetic_ct)

    def make_contour(z):
        c = pydicom.dataset.Dataset()
        c.ContourData = [2, 2, z, 7, 2, z, 7, 7, z, 2, 7, z]
        img = pydicom.dataset.Dataset()
        img.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
        c.ContourImageSequence = [img]
        return c

    contour1 = make_contour(0)
    contour2 = make_contour(5)

    roi_ds = pydicom.dataset.Dataset()
    roi_ds.ContourSequence = [contour1, contour2]

    with pytest.raises(ValueError):
        RegionOfInterest.from_rt_roi(
            roi_ds=roi_ds,
            name="BadROI",
            bins=16,
            ct_index={ct_ds.SOPInstanceUID: synthetic_ct},
        )


def test_missing_ct_slice_raises(synthetic_rtstruct):
    rt = pydicom.dcmread(synthetic_rtstruct)
    roi_ds = rt.ROIContourSequence[0]
    roi_ds.ContourSequence[0].ContourImageSequence[
        0
    ].ReferencedSOPInstanceUID = "9.9.9.9.9.9.9.9.9"

    with pytest.raises(FileNotFoundError):
        RegionOfInterest.from_rt_roi(
            roi_ds=roi_ds,
            name="MissingCT",
            bins=16,
            ct_index={},
        )
