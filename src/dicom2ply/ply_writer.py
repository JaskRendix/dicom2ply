import os

import numpy as np
from plyfile import PlyData, PlyElement


def write_roi_ply(roi, directory) -> None:
    """
    Write a RegionOfInterest to a PLY point cloud using plyfile.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Collect all vertices from all contours
    points = []
    for contour in roi.contours:
        pts = np.column_stack([contour.x, contour.y, contour.z])
        points.append(pts)

    if not points:
        print(f" - No points for ROI {roi.name}")
        return

    points = np.vstack(points).astype(np.float32)

    # Build structured array for plyfile
    vertex_data = np.zeros(
        points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]

    vertex_element = PlyElement.describe(vertex_data, "vertex")

    # Build PLY object
    ply = PlyData([vertex_element], text=False)  # binary_little_endian

    # Add metadata as comments
    ply.comments.append(f"name roi_{roi.name}")
    if roi.mean is not None:
        ply.comments.append(f"mean {roi.mean}")
        ply.comments.append(f"std {roi.std}")
        ply.comments.append(f"median {roi.median}")
        ply.comments.append(f"mode {roi.mode}")
        ply.comments.append(f"sum {roi.sum}")
        ply.comments.append(f"len {len(points)}")

    # Output path
    file_name = os.path.join(directory, f"roi_{roi.name}.ply")
    print(file_name)

    # Write file
    ply.write(file_name)
