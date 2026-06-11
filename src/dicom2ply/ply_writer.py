import logging
import os

import numpy as np
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)


def write_roi_ply(roi, directory) -> None:
    """
    Export a RegionOfInterest as a PLY point cloud.
    """

    os.makedirs(directory, exist_ok=True)

    # Collect all vertices from all contours
    patient_points = []
    hu_values = []
    slice_positions = []
    normals = []

    for contour in roi.contours:
        pts = contour.points_patient
        patient_points.append(pts)

        # HU values (masked_values) may be empty for clipped/empty contours
        if contour.masked_values is not None and contour.masked_values.size > 0:
            # Repeat HU values to match number of vertices
            hu = np.full(
                len(pts), float(contour.masked_values.mean()), dtype=np.float32
            )
        else:
            hu = np.full(len(pts), np.nan, dtype=np.float32)
        hu_values.append(hu)

        # Slice position (float)
        if contour.slice_pos is not None:
            sp = np.full(len(pts), float(contour.slice_pos), dtype=np.float32)
        else:
            sp = np.full(len(pts), np.nan, dtype=np.float32)
        slice_positions.append(sp)

        # Flat normal per contour (optional but useful)
        # Compute normal from polygon plane if possible
        if len(pts) >= 3:
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            if norm > 1e-6:
                n = n / norm
            else:
                n = np.array([0.0, 0.0, 0.0])
        else:
            n = np.array([0.0, 0.0, 0.0])

        normals.append(np.tile(n, (len(pts), 1)).astype(np.float32))

    if not patient_points:
        logger.warning(f"No points for ROI '{roi.name}', skipping PLY export")
        return

    # Stack all data
    pts = np.vstack(patient_points).astype(np.float32)
    hu = np.hstack(hu_values).astype(np.float32)
    sp = np.hstack(slice_positions).astype(np.float32)
    nrm = np.vstack(normals).astype(np.float32)

    # Build structured array for plyfile
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
    vertex_data["x"] = pts[:, 0]
    vertex_data["y"] = pts[:, 1]
    vertex_data["z"] = pts[:, 2]
    vertex_data["nx"] = nrm[:, 0]
    vertex_data["ny"] = nrm[:, 1]
    vertex_data["nz"] = nrm[:, 2]
    vertex_data["hu"] = hu
    vertex_data["slice_pos"] = sp

    vertex_element = PlyElement.describe(vertex_data, "vertex")
    ply = PlyData([vertex_element], text=False)

    # Metadata
    ply.comments.append(f"roi_name {roi.name}")
    ply.comments.append(f"num_contours {len(roi.contours)}")
    ply.comments.append(f"num_points {len(pts)}")

    if roi.mean is not None:
        ply.comments.append(f"roi_mean {roi.mean}")
        ply.comments.append(f"roi_std {roi.std}")
        ply.comments.append(f"roi_median {roi.median}")
        ply.comments.append(f"roi_mode {roi.mode}")
        ply.comments.append(f"roi_sum {roi.sum}")

    # Output path
    file_name = os.path.join(directory, f"roi_{roi.name}.ply")
    logger.info(f"Writing PLY for ROI '{roi.name}' → {file_name}")

    ply.write(file_name)
