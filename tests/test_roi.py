import pydicom
import pytest

from src.dicom2ply.contour import Contour
from src.dicom2ply.ct_cache import CTSliceCache
from src.dicom2ply.roi import RegionOfInterest


def test_contour_mask(synthetic_ct, synthetic_rtstruct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    rt = pydicom.dcmread(synthetic_rtstruct)

    cache = CTSliceCache({ct_ds.SOPInstanceUID: synthetic_ct})

    contour_ds = rt.ROIContourSequence[0].ContourSequence[0]
    c = Contour.from_rt(contour_ds, bins=16, cache=cache)

    assert c.mask is not None
    assert c.mask.sum() > 0
    assert c.stats.mean is not None


def test_roi_stats(synthetic_ct, synthetic_rtstruct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    rt = pydicom.dcmread(synthetic_rtstruct)

    ct_index = {ct_ds.SOPInstanceUID: synthetic_ct}

    roi_ds = rt.ROIContourSequence[0]
    roi = RegionOfInterest.from_rt_roi(
        roi_ds=roi_ds,
        name="TestROI",
        bins=16,
        ct_index=ct_index,
    )

    assert roi.mean is not None
    assert roi.mask_stack is not None
    assert roi.mask_stack.sum() > 0


def test_non_planar_contour_raises(synthetic_ct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    cache = CTSliceCache({ct_ds.SOPInstanceUID: synthetic_ct})

    # Contour points deliberately off-plane (z != 0)
    contour = pydicom.dataset.Dataset()
    contour.ContourData = [
        2,
        2,
        0.0,
        7,
        2,
        1.0,  # <-- off-plane
        7,
        7,
        0.0,
    ]
    img = pydicom.dataset.Dataset()
    img.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
    contour.ContourImageSequence = [img]

    with pytest.raises(ValueError):
        Contour.from_rt(contour, bins=16, cache=cache)


def test_empty_contour_mask(synthetic_ct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    cache = CTSliceCache({ct_ds.SOPInstanceUID: synthetic_ct})

    contour = pydicom.dataset.Dataset()
    contour.ContourData = [2, 2, 0.0]  # only 1 point
    img = pydicom.dataset.Dataset()
    img.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
    contour.ContourImageSequence = [img]

    c = Contour.from_rt(contour, bins=16, cache=cache)
    assert c.mask.sum() == 0
    assert c.stats.mean is None


def test_roi_multiple_contours_same_slice(synthetic_ct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    cache = CTSliceCache({ct_ds.SOPInstanceUID: synthetic_ct})

    # Two squares on the same slice
    def make_contour(coords):
        c = pydicom.dataset.Dataset()
        c.ContourData = coords
        img = pydicom.dataset.Dataset()
        img.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
        c.ContourImageSequence = [img]
        return c

    contour1 = make_contour([2, 2, 0, 4, 2, 0, 4, 4, 0, 2, 4, 0])
    contour2 = make_contour([6, 6, 0, 8, 6, 0, 8, 8, 0, 6, 8, 0])

    roi_ds = pydicom.dataset.Dataset()
    roi_ds.ContourSequence = [contour1, contour2]

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=roi_ds,
        name="TestROI",
        bins=16,
        ct_index={ct_ds.SOPInstanceUID: synthetic_ct},
    )

    assert roi.mask_stack.shape[-1] == 1
    assert roi.mask_stack.sum() > 0


def test_roi_extent(synthetic_ct, synthetic_rtstruct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    rt = pydicom.dcmread(synthetic_rtstruct)

    roi_ds = rt.ROIContourSequence[0]
    roi = RegionOfInterest.from_rt_roi(
        roi_ds=roi_ds,
        name="TestROI",
        bins=16,
        ct_index={ct_ds.SOPInstanceUID: synthetic_ct},
    )

    xmin, xmax, ymin, ymax, zmin, zmax = roi.extent
    assert xmin == 2.0
    assert xmax == 7.0
    assert ymin == 2.0
    assert ymax == 7.0
    assert zmin == 0.0
    assert zmax == 0.0


def test_mask_stack_sorted_by_slice_position(synthetic_ct):
    ct_ds = pydicom.dcmread(synthetic_ct)

    # Create two slices with different Z positions
    ds2 = pydicom.dcmread(synthetic_ct)
    ds2.SOPInstanceUID = "1.2.3.4.6"
    ds2.ImagePositionPatient = [0, 0, 5]
    ds2.save_as(str(synthetic_ct) + "_2")

    ct_index = {
        ct_ds.SOPInstanceUID: synthetic_ct,
        ds2.SOPInstanceUID: str(synthetic_ct) + "_2",
    }

    def make_contour(z, uid):
        c = pydicom.dataset.Dataset()
        c.ContourData = [2, 2, z, 7, 2, z, 7, 7, z, 2, 7, z]
        img = pydicom.dataset.Dataset()
        img.ReferencedSOPInstanceUID = uid
        c.ContourImageSequence = [img]
        return c

    contour1 = make_contour(0, ct_ds.SOPInstanceUID)
    contour2 = make_contour(5, ds2.SOPInstanceUID)

    roi_ds = pydicom.dataset.Dataset()
    roi_ds.ContourSequence = [contour1, contour2]

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=roi_ds,
        name="TestROI",
        bins=16,
        ct_index=ct_index,
    )

    assert list(roi.slice_positions) == sorted(roi.slice_positions)
    assert roi.mask_stack.shape[-1] == 2


def test_pixel_spacing_affects_mask(synthetic_ct):
    ds = pydicom.dcmread(synthetic_ct)
    ds.PixelSpacing = [2.0, 2.0]  # double spacing
    ds.save_as(synthetic_ct)

    cache = CTSliceCache({ds.SOPInstanceUID: synthetic_ct})

    contour = pydicom.dataset.Dataset()
    contour.ContourData = [0, 0, 0, 9, 0, 0, 9, 9, 0]
    img = pydicom.dataset.Dataset()
    img.ReferencedSOPInstanceUID = ds.SOPInstanceUID
    contour.ContourImageSequence = [img]

    c = Contour.from_rt(contour, bins=16, cache=cache)

    # With spacing=2mm, the polygon covers fewer pixels
    assert c.mask.sum() < 100


def test_histogram_bins(synthetic_ct, synthetic_rtstruct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    rt = pydicom.dcmread(synthetic_rtstruct)

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=rt.ROIContourSequence[0],
        name="TestROI",
        bins=32,
        ct_index={ct_ds.SOPInstanceUID: synthetic_ct},
    )

    hist, edges = roi.histogram
    assert len(hist) == 32
    assert len(edges) == 33


def test_roi_export_json(tmp_path, synthetic_ct, synthetic_rtstruct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    rt = pydicom.dcmread(synthetic_rtstruct)

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=rt.ROIContourSequence[0],
        name="TestROI",
        bins=16,
        ct_index={ct_ds.SOPInstanceUID: synthetic_ct},
    )

    data = roi.export_json()
    assert isinstance(data, dict)
    assert data["name"] == "TestROI"
    assert "stats" in data
    assert "volume" in data


def test_roi_export_mask_slice_png(tmp_path, synthetic_ct, synthetic_rtstruct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    rt = pydicom.dcmread(synthetic_rtstruct)

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=rt.ROIContourSequence[0],
        name="TestROI",
        bins=16,
        ct_index={ct_ds.SOPInstanceUID: synthetic_ct},
    )

    out = tmp_path / "slice.png"
    roi.export_mask_slice_png(0, out)
    assert out.exists()


def test_roi_export_all_slices_png(tmp_path, synthetic_ct, synthetic_rtstruct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    rt = pydicom.dcmread(synthetic_rtstruct)

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=rt.ROIContourSequence[0],
        name="TestROI",
        bins=16,
        ct_index={ct_ds.SOPInstanceUID: synthetic_ct},
    )

    out = tmp_path / "slices"
    roi.export_all_slices_png(out)
    assert out.exists()
    assert list(out.glob("*.png"))


def test_roi_export_mesh_ply(tmp_path, synthetic_ct, synthetic_rtstruct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    rt = pydicom.dcmread(synthetic_rtstruct)

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=rt.ROIContourSequence[0],
        name="TestROI",
        bins=16,
        ct_index={ct_ds.SOPInstanceUID: synthetic_ct},
    )

    out = tmp_path / "mesh.ply"
    roi.export_mesh_ply(out)
    assert out.exists()


def test_roi_export_mask_nifti_float(tmp_path, synthetic_ct, synthetic_rtstruct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    rt = pydicom.dcmread(synthetic_rtstruct)

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=rt.ROIContourSequence[0],
        name="TestROI",
        bins=16,
        ct_index={ct_ds.SOPInstanceUID: synthetic_ct},
    )

    out = tmp_path / "mask_float.nii.gz"
    roi.export_mask_nifti_float(out)
    assert out.exists()


def test_roi_get_voxel_coordinates(synthetic_ct, synthetic_rtstruct):
    ct_ds = pydicom.dcmread(synthetic_ct)
    rt = pydicom.dcmread(synthetic_rtstruct)

    roi = RegionOfInterest.from_rt_roi(
        roi_ds=rt.ROIContourSequence[0],
        name="TestROI",
        bins=16,
        ct_index={ct_ds.SOPInstanceUID: synthetic_ct},
    )

    coords = roi.get_voxel_coordinates()
    assert coords.ndim == 2
    assert coords.shape[1] == 3
    assert coords.shape[0] > 0
