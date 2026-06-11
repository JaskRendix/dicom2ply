from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from plyfile import PlyData, PlyElement

if TYPE_CHECKING:
    from .roi import RegionOfInterest

logger = logging.getLogger(__name__)


def hu_to_rgb(hu: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Map HU values to RGB using a simple windowed linear mapping.
    Window: [-1000, 3000] → [0, 255]
    """
    hu_clipped = np.clip(hu, -1000, 3000)
    norm = (hu_clipped + 1000) / 4000
    norm = np.nan_to_num(norm, nan=0.0)
    rgb = (norm * 255).astype(np.uint8)
    return np.stack([rgb, rgb, rgb], axis=1)


def write_roi_ply(roi: RegionOfInterest, directory: str) -> None:
    """
    Export a RegionOfInterest as a PLY point cloud.
    """

    os.makedirs(directory, exist_ok=True)

    patient_points: list[NDArray[np.float32]] = []
    hu_values: list[NDArray[np.float32]] = []
    slice_positions: list[NDArray[np.float32]] = []
    normals: list[NDArray[np.float32]] = []

    for contour in roi.contours:
        pts: NDArray[np.float32] = contour.points_patient.astype(np.float32)
        patient_points.append(pts)

        # HU values
        if contour.masked_values is not None and contour.masked_values.size > 0:
            hu = np.full(
                len(pts), float(contour.masked_values.mean()), dtype=np.float32
            )
        else:
            hu = np.full(len(pts), np.nan, dtype=np.float32)
        hu_values.append(hu)

        # Slice position
        if contour.slice_pos is not None:
            sp = np.full(len(pts), float(contour.slice_pos), dtype=np.float32)
        else:
            sp = np.full(len(pts), np.nan, dtype=np.float32)
        slice_positions.append(sp)

        # Flat normal
        if len(pts) >= 3:
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            if norm > 1e-6:
                n = n / norm
            else:
                n = np.zeros(3, dtype=np.float32)
        else:
            n = np.zeros(3, dtype=np.float32)

        normals.append(np.tile(n, (len(pts), 1)).astype(np.float32))

    if not patient_points:
        logger.warning(f"No points for ROI '{roi.name}', skipping PLY export")
        return

    pts = np.vstack(patient_points).astype(np.float32)
    hu = np.hstack(hu_values).astype(np.float32)
    sp = np.hstack(slice_positions).astype(np.float32)
    nrm = np.vstack(normals).astype(np.float32)

    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("hu", "f4"),
        ("slice_pos", "f4"),
    ]

    vertex_data = np.zeros(len(pts), dtype=vertex_dtype)
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = pts.T
    vertex_data["nx"], vertex_data["ny"], vertex_data["nz"] = nrm.T
    vertex_data["hu"] = hu
    vertex_data["slice_pos"] = sp

    ply = PlyData([PlyElement.describe(vertex_data, "vertex")], text=False)

    ply.comments.append(f"roi_name {roi.name}")
    ply.comments.append(f"num_contours {len(roi.contours)}")
    ply.comments.append(f"num_points {len(pts)}")

    if roi.mean is not None:
        ply.comments.append(f"roi_mean {roi.mean}")
        ply.comments.append(f"roi_std {roi.std}")
        ply.comments.append(f"roi_median {roi.median}")
        ply.comments.append(f"roi_mode {roi.mode}")
        ply.comments.append(f"roi_sum {roi.sum}")

    file_name = os.path.join(directory, f"roi_{roi.name}.ply")
    logger.info(f"Writing PLY for ROI '{roi.name}' → {file_name}")
    ply.write(file_name)


def write_roi_ply_points(roi: RegionOfInterest, directory: str) -> None:
    """
    Export ROI as a PLY point cloud with:
    - XYZ
    - Normals
    - HU values
    - Slice position
    - RGB color (from HU)
    """

    os.makedirs(directory, exist_ok=True)

    pts_all: list[NDArray[np.float32]] = []
    hu_all: list[NDArray[np.float32]] = []
    slice_pos_all: list[NDArray[np.float32]] = []
    normals_all: list[NDArray[np.float32]] = []

    for contour in roi.contours:
        pts = contour.points_patient.astype(np.float32)
        pts_all.append(pts)

        if contour.masked_values is not None and contour.masked_values.size > 0:
            hu = np.full(
                len(pts), float(contour.masked_values.mean()), dtype=np.float32
            )
        else:
            hu = np.full(len(pts), np.nan, dtype=np.float32)
        hu_all.append(hu)

        if contour.slice_pos is not None:
            sp = np.full(len(pts), float(contour.slice_pos), dtype=np.float32)
        else:
            sp = np.full(len(pts), np.nan, dtype=np.float32)
        slice_pos_all.append(sp)

        if len(pts) >= 3:
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            if norm > 1e-6:
                n = n / norm
            else:
                n = np.zeros(3, dtype=np.float32)
        else:
            n = np.zeros(3, dtype=np.float32)

        normals_all.append(np.tile(n, (len(pts), 1)).astype(np.float32))

    if not pts_all:
        logger.warning(f"No points for ROI '{roi.name}', skipping PLY export")
        return

    pts = np.vstack(pts_all).astype(np.float32)
    hu = np.hstack(hu_all).astype(np.float32)
    slice_pos = np.hstack(slice_pos_all).astype(np.float32)
    normals = np.vstack(normals_all).astype(np.float32)
    rgb = hu_to_rgb(hu)

    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("hu", "f4"),
        ("slice_pos", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    vertex_data = np.zeros(len(pts), dtype=vertex_dtype)
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = pts.T
    vertex_data["nx"], vertex_data["ny"], vertex_data["nz"] = normals.T
    vertex_data["hu"] = hu
    vertex_data["slice_pos"] = slice_pos
    vertex_data["red"], vertex_data["green"], vertex_data["blue"] = rgb.T

    ply = PlyData([PlyElement.describe(vertex_data, "vertex")], text=False)

    file_name = os.path.join(directory, f"roi_{roi.name}_points.ply")
    logger.info(f"Writing PLY point cloud for ROI '{roi.name}' → {file_name}")
    ply.write(file_name)


def triangulate_polygon(pts: NDArray[np.float32]) -> NDArray[np.int32]:
    """
    Fan triangulation: (0, i, i+1)
    """
    if len(pts) < 3:
        return np.zeros((0, 3), dtype=np.int32)
    return np.column_stack(
        [
            np.zeros(len(pts) - 2, dtype=np.int32),
            np.arange(1, len(pts) - 1, dtype=np.int32),
            np.arange(2, len(pts), dtype=np.int32),
        ]
    )


def write_roi_las(roi: RegionOfInterest, directory: str) -> None:
    """
    Export ROI as LAS/LAZ point cloud.
    Includes XYZ, intensity (HU), and RGB.
    """

    try:
        import laspy
    except ImportError as e:
        raise ImportError(
            "LAS export requires the optional dependency 'laspy'. "
            "Install it via: pip install dicom2ply[laspy]"
        ) from e

    os.makedirs(directory, exist_ok=True)

    pts_list: list[NDArray[np.float32]] = []
    hu_list: list[NDArray[np.float32]] = []

    for contour in roi.contours:
        pts = contour.points_patient.astype(np.float32)
        pts_list.append(pts)

        if contour.masked_values is not None and contour.masked_values.size > 0:
            hu = np.full(
                len(pts), float(contour.masked_values.mean()), dtype=np.float32
            )
        else:
            hu = np.full(len(pts), np.nan, dtype=np.float32)
        hu_list.append(hu)

    if not pts_list:
        logger.warning(f"No points for ROI '{roi.name}', skipping LAS export")
        return

    pts = np.vstack(pts_list).astype(np.float64)
    hu = np.hstack(hu_list).astype(np.float32)

    # Avoid warnings: replace NaN before color mapping
    hu_safe = np.nan_to_num(hu, nan=0.0)
    rgb = hu_to_rgb(hu_safe)

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)

    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]
    las.intensity = hu_safe.astype(np.uint16)
    las.red = rgb[:, 0].astype(np.uint16) * 256
    las.green = rgb[:, 1].astype(np.uint16) * 256
    las.blue = rgb[:, 2].astype(np.uint16) * 256

    file_name = os.path.join(directory, f"roi_{roi.name}.las")
    logger.info(f"Writing LAS for ROI '{roi.name}' → {file_name}")
    las.write(file_name)


def write_roi_ply_mesh(roi: RegionOfInterest, directory: str) -> None:
    """
    Export ROI as a triangulated PLY mesh.
    Each contour is triangulated independently using fan triangulation.
    """

    os.makedirs(directory, exist_ok=True)

    from dicom2ply.exporters import triangulate_polygon

    vertices: list[NDArray[np.float32]] = []
    faces: list[NDArray[np.int32]] = []
    offset = 0

    for contour in roi.contours:
        pts = contour.points_patient.astype(np.float32)

        # Always append vertices
        vertices.append(pts)

        # Triangulate only if enough points
        if pts.shape[0] >= 3:
            tri = triangulate_polygon(pts)
            if tri.size > 0:
                faces.append(tri + offset)

        offset += len(pts)

    if not vertices:
        logger.warning(f"No points for ROI '{roi.name}', skipping mesh export")
        return

    verts = np.vstack(vertices).astype(np.float32)
    face_arr = (
        np.vstack(faces).astype(np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    )

    vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    vertex_data = np.zeros(len(verts), dtype=vertex_dtype)
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = verts.T

    face_dtype = [("vertex_indices", "i4", (3,))]
    face_data = np.zeros(len(face_arr), dtype=face_dtype)
    if len(face_arr) > 0:
        face_data["vertex_indices"] = face_arr

    ply = PlyData(
        [
            PlyElement.describe(vertex_data, "vertex"),
            PlyElement.describe(face_data, "face"),
        ],
        text=False,
    )

    file_name = os.path.join(directory, f"roi_{roi.name}_mesh.ply")
    logger.info(f"Writing PLY mesh for ROI '{roi.name}' → {file_name}")
    ply.write(file_name)
