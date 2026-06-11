import numpy as np
import pytest

from dicom2ply.exporters import (
    hu_to_rgb,
    triangulate_polygon,
    write_roi_las,
    write_roi_ply,
    write_roi_ply_mesh,
    write_roi_ply_points,
)


class DummyContour:
    def __init__(self, points, hu=None, slice_pos=None):
        self.points_patient = np.asarray(points, dtype=np.float32)
        self.masked_values = None if hu is None else np.asarray(hu, dtype=np.float32)
        self.slice_pos = slice_pos


class DummyROI:
    def __init__(self, name, contours, stats=None):
        self.name = name
        self.contours = contours
        self.mean = stats.get("mean") if stats else None
        self.std = stats.get("std") if stats else None
        self.median = stats.get("median") if stats else None
        self.mode = stats.get("mode") if stats else None
        self.sum = stats.get("sum") if stats else None


@pytest.mark.parametrize(
    "hu, expected",
    [
        (np.array([-2000], dtype=np.float32), np.array([[0, 0, 0]], dtype=np.uint8)),
        (np.array([0], dtype=np.float32), np.array([[63, 63, 63]], dtype=np.uint8)),
        (
            np.array([3000], dtype=np.float32),
            np.array([[255, 255, 255]], dtype=np.uint8),
        ),
    ],
)
def test_hu_to_rgb(hu, expected):
    rgb = hu_to_rgb(hu)
    assert rgb.shape == (1, 3)
    assert np.all(rgb == expected)


def test_triangulate_polygon_triangle():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    tri = triangulate_polygon(pts)
    assert tri.shape == (1, 3)
    assert np.all(tri[0] == [0, 1, 2])


def test_triangulate_polygon_quad():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    tri = triangulate_polygon(pts)
    assert tri.shape == (2, 3)
    assert np.all(tri[0] == [0, 1, 2])
    assert np.all(tri[1] == [0, 2, 3])


def test_triangulate_polygon_degenerate():
    pts = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
    tri = triangulate_polygon(pts)
    assert tri.shape == (0, 3)


def test_write_roi_ply(tmp_path):
    roi = DummyROI(
        "Test",
        contours=[
            DummyContour(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0]], hu=[10, 20, 30], slice_pos=5.0
            )
        ],
        stats={"mean": 10, "std": 2, "median": 9, "mode": 8, "sum": 100},
    )

    outdir = tmp_path / "ply"
    write_roi_ply(roi, outdir)

    outfile = outdir / "roi_Test.ply"
    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_write_roi_ply_points(tmp_path):
    roi = DummyROI(
        "RGB",
        contours=[
            DummyContour(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0]], hu=[100, 200, 300], slice_pos=10.0
            )
        ],
    )

    outdir = tmp_path / "ply_rgb"
    write_roi_ply_points(roi, outdir)

    outfile = outdir / "roi_RGB_points.ply"
    assert outfile.exists()
    assert outfile.stat().st_size > 0


@pytest.mark.parametrize(
    "hu_values",
    [
        [10, 20, 30],
        [np.nan, np.nan, np.nan],
        None,
    ],
)
def test_write_roi_las(tmp_path, hu_values):
    if hu_values is None:
        contour = DummyContour([[0, 0, 0], [1, 0, 0], [1, 1, 0]], hu=None)
    else:
        contour = DummyContour([[0, 0, 0], [1, 0, 0], [1, 1, 0]], hu=hu_values)

    roi = DummyROI("LAS", contours=[contour])

    outdir = tmp_path / "las"
    write_roi_las(roi, outdir)

    outfile = outdir / "roi_LAS.las"
    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_empty_roi(tmp_path, caplog):
    roi = DummyROI("Empty", contours=[])
    outdir = tmp_path / "empty"

    write_roi_ply(roi, outdir)

    assert "No points for ROI" in caplog.text
    assert not (outdir / "roi_Empty.ply").exists()


def test_nan_hu(tmp_path):
    roi = DummyROI(
        "NaN",
        contours=[
            DummyContour([[0, 0, 0], [1, 0, 0], [1, 1, 0]], hu=[np.nan, np.nan, np.nan])
        ],
    )

    outdir = tmp_path / "nan"
    write_roi_ply_points(roi, outdir)

    outfile = outdir / "roi_NaN_points.ply"
    assert outfile.exists()


def test_multiple_contours(tmp_path):
    roi = DummyROI(
        "Multi",
        contours=[
            DummyContour([[0, 0, 0], [1, 0, 0]], hu=[10, 20]),
            DummyContour([[1, 1, 0], [2, 1, 0]], hu=[30, 40]),
        ],
    )

    outdir = tmp_path / "multi"
    write_roi_ply(roi, outdir)

    outfile = outdir / "roi_Multi.ply"
    assert outfile.exists()
    assert outfile.stat().st_size > 0


@pytest.mark.parametrize(
    "num_contours, pts_per_contour",
    [
        (1, 3),
        (5, 10),
        (10, 50),
    ],
)
def test_large_roi(tmp_path, num_contours, pts_per_contour):
    contours = []
    for i in range(num_contours):
        pts = np.random.rand(pts_per_contour, 3) * 100
        hu = np.random.randn(pts_per_contour) * 50
        contours.append(DummyContour(pts, hu=hu, slice_pos=float(i)))

    roi = DummyROI("Large", contours=contours)

    outdir = tmp_path / "large"
    write_roi_ply_points(roi, outdir)

    outfile = outdir / "roi_Large_points.ply"
    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_write_roi_ply_mesh_triangle(tmp_path):
    roi = DummyROI(
        "MeshTri",
        contours=[
            DummyContour(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                hu=[10, 20, 30],
                slice_pos=0.0,
            )
        ],
    )

    outdir = tmp_path / "mesh_tri"
    write_roi_ply_mesh(roi, outdir)

    outfile = outdir / "roi_MeshTri_mesh.ply"
    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_write_roi_ply_mesh_quad(tmp_path):
    roi = DummyROI(
        "MeshQuad",
        contours=[
            DummyContour(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                hu=[10, 20, 30, 40],
                slice_pos=0.0,
            )
        ],
    )

    outdir = tmp_path / "mesh_quad"
    write_roi_ply_mesh(roi, outdir)

    outfile = outdir / "roi_MeshQuad_mesh.ply"
    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_write_roi_ply_mesh_multiple_contours(tmp_path):
    roi = DummyROI(
        "MeshMulti",
        contours=[
            DummyContour([[0, 0, 0], [1, 0, 0], [0, 1, 0]], hu=[1, 2, 3]),
            DummyContour([[2, 2, 0], [3, 2, 0], [2, 3, 0]], hu=[4, 5, 6]),
        ],
    )

    outdir = tmp_path / "mesh_multi"
    write_roi_ply_mesh(roi, outdir)

    outfile = outdir / "roi_MeshMulti_mesh.ply"
    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_write_roi_ply_mesh_degenerate(tmp_path):
    roi = DummyROI(
        "MeshDegenerate",
        contours=[
            DummyContour([[0, 0, 0], [0, 0, 0]], hu=[1, 1]),  # < 3 points
        ],
    )

    outdir = tmp_path / "mesh_degenerate"
    write_roi_ply_mesh(roi, outdir)

    outfile = outdir / "roi_MeshDegenerate_mesh.ply"
    # File should still be created, but contain only vertices
    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_write_roi_ply_mesh_empty_roi(tmp_path, caplog):
    roi = DummyROI("MeshEmpty", contours=[])

    outdir = tmp_path / "mesh_empty"
    write_roi_ply_mesh(roi, outdir)

    assert "No points for ROI" in caplog.text
    assert not (outdir / "roi_MeshEmpty_mesh.ply").exists()
