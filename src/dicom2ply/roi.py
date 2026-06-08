from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydicom.dataset import Dataset

from dicom2ply.contour import Contour
from dicom2ply.ct_cache import CTSliceCache


@dataclass
class RegionOfInterest:
    name: str
    contours: list[Contour]
    bins: int

    histogram: tuple[np.ndarray, np.ndarray] | None = None
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    mode: float | None = None
    sum: float | None = None
    count: int | None = None
    extent: tuple[float, float, float, float, float, float] | None = None
    mask_stack: np.ndarray | None = None
    slice_positions: np.ndarray | None = None

    # Cached mesh (verts, faces) for all mesh exports
    _mesh_verts: np.ndarray | None = None
    _mesh_faces: np.ndarray | None = None

    @classmethod
    def from_rt_roi(
        cls, roi_ds: Dataset, name: str, bins: int, ct_index: dict[str, str]
    ):
        """
        Build an ROI from an RTSTRUCT ROIContourSequence entry.
        Uses the modular Contour.from_rt + CTSliceCache pipeline.
        """
        cache = CTSliceCache(ct_index)

        seq = getattr(roi_ds, "ContourSequence", None)
        if not seq:
            return cls(name=name, contours=[], bins=bins)

        contours: list[Contour] = []
        for contour_ds in seq:
            c = Contour.from_rt(contour_ds, bins=bins, cache=cache)
            if c.stats.mean is not None:
                contours.append(c)

        slice_map = {}
        for c in contours:
            if c.slice_uid not in slice_map:
                slice_map[c.slice_uid] = c.slice_pos
            else:
                if abs(slice_map[c.slice_uid] - c.slice_pos) > 1e-3:
                    raise ValueError(
                        f"Contours for SOPInstanceUID {c.slice_uid} have inconsistent "
                        f"slice positions: {slice_map[c.slice_uid]} vs {c.slice_pos}"
                    )

        # Sort contours deterministically by slice position
        contours.sort(key=lambda c: c.slice_pos)

        obj = cls(name=name, contours=contours, bins=bins)

        obj.compute_extent()  # geometry only
        obj.compute_mask_stack()  # build 3D mask
        obj.compute_stats()  # HU stats
        obj.compute_volume_stats()  # physical volume stats

        return obj

    def compute_stats(self):
        """
        Compute HU statistics and histogram for the ROI.
        Uses concatenation (simple, but can be refactored to incremental if needed).
        """
        if not self.contours:
            self.count = 0
            return

        values_list = [
            c.masked_values for c in self.contours if c.masked_values is not None
        ]
        if not values_list:
            self.count = 0
            return

        values = np.concatenate(values_list)
        counts, edges = np.histogram(values, bins=self.bins)
        centers = (edges[:-1] + edges[1:]) / 2

        self.histogram = (counts, edges)
        self.mode = float(centers[np.argmax(counts)])
        self.mean = float(values.mean())
        self.std = float(values.std())
        self.median = float(np.median(values))
        self.sum = float(values.sum())
        self.count = int(values.size)

    def compute_extent(self):
        """
        Compute bounding-box extent in patient space from contour vertices.
        """
        if not self.contours:
            return

        xs = np.concatenate([c.x for c in self.contours])
        ys = np.concatenate([c.y for c in self.contours])
        zs = np.concatenate([c.z for c in self.contours])

        self.extent = (
            float(xs.min()),
            float(xs.max()),
            float(ys.min()),
            float(ys.max()),
            float(zs.min()),
            float(zs.max()),
        )

    def compute_mask_stack(self):
        """
        Build a 3D mask volume using true geometric slice positions.
        Ensures consistent geometry and avoids silent mismatches.
        """
        if not self.contours:
            return

        ds0 = self.contours[0].ds
        rows = int(ds0.Rows)
        cols = int(ds0.Columns)

        # Collect true geometric slice positions
        positions = np.array([float(c.slice_pos) for c in self.contours])

        # Floating‑point safe unique positions
        uniq = np.unique(np.round(positions, decimals=5))
        uniq.sort()
        self.slice_positions = uniq

        # Map slice_pos → index
        pos_to_idx = {p: i for i, p in enumerate(uniq)}

        # Allocate mask volume
        volume = np.zeros((rows, cols, len(uniq)), np.int8)

        for c in self.contours:
            if c.slice_pos is None:
                continue

            # Round slice_pos to match uniq keys
            p = float(np.round(c.slice_pos, 5))
            if p not in pos_to_idx:
                continue
            idx = pos_to_idx[p]

            # Validate mask geometry
            if c.mask is None or c.mask.shape != (rows, cols):
                # Geometry mismatch → skip
                continue

            volume[..., idx] |= c.mask

        self.mask_stack = volume

    def _get_spacing_z(self, ds: Dataset) -> float:
        """
        Compute slice spacing (mm) from geometric slice positions if available,
        otherwise fall back to SliceThickness.
        """
        if self.slice_positions is not None and len(self.slice_positions) > 1:
            diffs = np.diff(self.slice_positions)
            return float(np.mean(np.abs(diffs)))
        return float(getattr(ds, "SliceThickness", 1.0))

    def _get_affine(self, ds: Dataset, spacing_z: float) -> np.ndarray:
        """
        Construct a 4x4 affine matrix from DICOM orientation and spacing.
        """
        row_dir = np.array(ds.ImageOrientationPatient[:3], float)
        col_dir = np.array(ds.ImageOrientationPatient[3:], float)
        normal = np.cross(row_dir, col_dir)

        spacing_x = float(ds.PixelSpacing[0])
        spacing_y = float(ds.PixelSpacing[1])

        affine = np.eye(4)
        affine[:3, 0] = -row_dir * spacing_x
        affine[:3, 1] = -col_dir * spacing_y
        affine[:3, 2] = normal * spacing_z

        origin = np.array(ds.ImagePositionPatient, float)
        affine[:3, 3] = origin

        return affine

    def export_nifti(self, output_path: str | Path) -> None:
        """
        Export the ROI mask stack as a NIfTI volume using true geometric
        slice positions and correct DICOM orientation.
        """
        try:
            import nibabel as nib
        except ImportError:
            print("Cannot export NIfTI: nibabel not installed.")
            return

        if self.mask_stack is None or not self.contours:
            print("No mask data available for NIfTI export.")
            return

        ds = self.contours[0].ds

        spacing_z = self._get_spacing_z(ds)
        affine = self._get_affine(ds, spacing_z)

        # NIfTI expects (X,Y,Z) = (cols, rows, slices)
        data = np.swapaxes(self.mask_stack, 0, 1).astype(np.uint8)

        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        img.header.set_qform(affine, code=1)

        nib.save(img, str(output_path))

    def export_mask_nifti_float(self, output_path: str | Path) -> None:
        """
        Export the ROI mask as a float32 NIfTI volume using true geometric
        slice positions and correct DICOM orientation.
        """
        try:
            import nibabel as nib
        except ImportError:
            print("Cannot export NIfTI: nibabel not installed.")
            return

        if self.mask_stack is None or self.slice_positions is None:
            print("No mask data available.")
            return

        ds = self.contours[0].ds

        spacing_z = self._get_spacing_z(ds)
        affine = self._get_affine(ds, spacing_z)

        data = np.swapaxes(self.mask_stack, 0, 1).astype(np.float32)

        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        img.header.set_qform(affine, code=1)

        nib.save(img, str(output_path))

    def compute_volume_stats(self) -> None:
        """
        Compute physical volume statistics for the ROI using mask_stack and DICOM metadata.
        Produces:
            - volume_cc: total segmented volume
            - voxel_volume_mm3 / voxel_volume_cc
            - slice_thickness
            - pixel_spacing
            - num_voxels
            - num_slices
            - extent_volume_cc: bounding-box physical volume
        """
        if self.mask_stack is None or not self.contours:
            return

        ds = self.contours[0].ds

        # Pixel spacing (mm)
        spacing_x = float(ds.PixelSpacing[0])
        spacing_y = float(ds.PixelSpacing[1])
        pixel_spacing = (spacing_x, spacing_y)

        # Slice thickness (mm)
        if self.slice_positions is not None and len(self.slice_positions) > 1:
            slice_thickness = float(np.abs(np.diff(self.slice_positions)).mean())
        else:
            slice_thickness = float(getattr(ds, "SliceThickness", 1.0))

        # Voxel volume
        voxel_volume_mm3 = spacing_x * spacing_y * slice_thickness
        voxel_volume_cc = voxel_volume_mm3 / 1000.0

        # Segmented voxel count
        num_voxels = int(np.sum(self.mask_stack))
        volume_cc = num_voxels * voxel_volume_cc

        # Bounding-box geometric volume
        if self.extent is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = self.extent
            dx = xmax - xmin
            dy = ymax - ymin
            dz = zmax - zmin
            extent_volume_cc = (dx * dy * dz) / 1000.0
        else:
            extent_volume_cc = None

        # Store results
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness
        self.voxel_volume_mm3 = voxel_volume_mm3
        self.voxel_volume_cc = voxel_volume_cc
        self.num_voxels = num_voxels
        self.num_slices = self.mask_stack.shape[2]
        self.volume_cc = volume_cc
        self.extent_volume_cc = extent_volume_cc

    def export_json(self) -> dict[str, Any]:
        """
        Return a JSON‑serializable dictionary containing all ROI statistics,
        geometry, and volume information.
        """
        return {
            "name": self.name,
            "num_contours": len(self.contours),
            "bins": self.bins,
            "stats": {
                "mean": self.mean,
                "std": self.std,
                "median": self.median,
                "mode": self.mode,
                "sum": self.sum,
                "count": self.count,
            },
            "extent": {
                "xmin": self.extent[0] if self.extent else None,
                "xmax": self.extent[1] if self.extent else None,
                "ymin": self.extent[2] if self.extent else None,
                "ymax": self.extent[3] if self.extent else None,
                "zmin": self.extent[4] if self.extent else None,
                "zmax": self.extent[5] if self.extent else None,
            },
            "volume": {
                "volume_cc": getattr(self, "volume_cc", None),
                "voxel_volume_mm3": getattr(self, "voxel_volume_mm3", None),
                "voxel_volume_cc": getattr(self, "voxel_volume_cc", None),
                "slice_thickness": getattr(self, "slice_thickness", None),
                "pixel_spacing": getattr(self, "pixel_spacing", None),
                "num_voxels": getattr(self, "num_voxels", None),
                "num_slices": getattr(self, "num_slices", None),
                "extent_volume_cc": getattr(self, "extent_volume_cc", None),
            },
        }

    def export_mask_slice_png(self, slice_index: int, output_path: str | Path) -> None:
        """
        Export a single mask slice as a PNG image.
        """
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("Cannot export PNG: imageio not installed.")
            return

        if self.mask_stack is None:
            print("No mask stack available.")
            return

        if slice_index < 0 or slice_index >= self.mask_stack.shape[2]:
            print("Slice index out of range.")
            return

        mask = (self.mask_stack[..., slice_index] > 0).astype(np.uint8)
        img = mask * 255

        imageio.imwrite(output_path, img)

    def export_all_slices_png(self, directory: str | Path) -> None:
        """
        Export all mask slices as PNG images into a directory.
        """
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("Cannot export PNG: imageio not installed.")
            return

        if self.mask_stack is None:
            print("No mask stack available.")
            return

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        for i in range(self.mask_stack.shape[2]):
            mask = (self.mask_stack[..., i] > 0).astype(np.uint8)
            img = mask * 255
            imageio.imwrite(out / f"{self.name}_slice_{i:04d}.png", img)

    def _build_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (and cache) a surface mesh (verts, faces) from the mask_stack
        using marching cubes. Volume is in (Z, X, Y) order for skimage.
        """
        if self._mesh_verts is not None and self._mesh_faces is not None:
            return self._mesh_verts, self._mesh_faces

        try:
            from skimage import measure
        except ImportError:
            raise ImportError("Cannot build mesh: skimage not installed.")

        if self.mask_stack is None:
            raise ValueError("No mask stack available for mesh export.")

        # skimage expects (Z, Y, X); we have (rows, cols, slices) = (Y, X, Z)
        vol = np.swapaxes(self.mask_stack, 0, 2)  # (Z, X, Y)
        vol = np.swapaxes(vol, 1, 2)  # (Z, Y, X)

        if vol.shape[0] < 2 or vol.shape[1] < 2 or vol.shape[2] < 2:
            self._mesh_verts = np.zeros((0, 3), dtype=np.float32)
            self._mesh_faces = np.zeros((0, 3), dtype=np.int32)
            return self._mesh_verts, self._mesh_faces

        verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)

        self._mesh_verts = verts.astype(np.float32)
        self._mesh_faces = faces.astype(np.int32)
        return self._mesh_verts, self._mesh_faces

    def export_mesh_ply(self, output_path: str | Path) -> None:
        """
        Export a surface mesh (marching cubes) of the ROI mask as a PLY file.
        """
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            print("Cannot export mesh: plyfile not installed.")
            return

        if self.mask_stack is None:
            print("No mask stack available.")
            return

        try:
            verts, faces = self._build_mesh()
        except ImportError as e:
            print(str(e))
            return
        except ValueError as e:
            print(str(e))
            return

        if verts.size == 0 or faces.size == 0:
            vertex_data = np.zeros(0, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
            face_data = np.zeros(0, dtype=[("vertex_indices", "i4", (3,))])
            ply = PlyData(
                [
                    PlyElement.describe(vertex_data, "vertex"),
                    PlyElement.describe(face_data, "face"),
                ],
                text=False,
            )
            ply.write(str(output_path))
            return

        vertex_data = np.zeros(
            len(verts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
        )
        vertex_data["x"], vertex_data["y"], vertex_data["z"] = verts.T

        face_data = np.zeros(len(faces), dtype=[("vertex_indices", "i4", (3,))])
        face_data["vertex_indices"] = faces

        ply = PlyData(
            [
                PlyElement.describe(vertex_data, "vertex"),
                PlyElement.describe(face_data, "face"),
            ],
            text=False,
        )
        ply.write(str(output_path))

    def export_mesh_stl(self, output_path: str | Path) -> None:
        """
        Export a surface mesh (marching cubes) of the ROI mask as a binary STL file.
        """
        if self.mask_stack is None:
            print("No mask stack available.")
            return

        try:
            verts, faces = self._build_mesh()
        except ImportError as e:
            print(str(e))
            return
        except ValueError as e:
            print(str(e))
            return

        if verts.size == 0 or faces.size == 0:
            print("Cannot export STL: volume too small for marching cubes.")
            with open(output_path, "wb") as f:
                f.write(b"dicom2ply STL export".ljust(80, b" "))
                f.write((0).to_bytes(4, byteorder="little", signed=False))
            return

        # Binary STL: 80‑byte header, uint32 triangle count, then triangles
        with open(output_path, "wb") as f:
            f.write(b"dicom2ply STL export".ljust(80, b" "))
            f.write(len(faces).to_bytes(4, byteorder="little", signed=False))

            for tri in faces:
                v0, v1, v2 = verts[tri]

                # Compute normal
                n = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(n)
                if norm > 0:
                    n /= norm
                else:
                    n = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                f.write(np.asarray(n, dtype=np.float32).tobytes())
                f.write(np.asarray(v0, dtype=np.float32).tobytes())
                f.write(np.asarray(v1, dtype=np.float32).tobytes())
                f.write(np.asarray(v2, dtype=np.float32).tobytes())
                f.write((0).to_bytes(2, byteorder="little", signed=False))

    def export_mesh_obj(self, output_path: str | Path) -> None:
        """
        Export a surface mesh (marching cubes) of the ROI mask as a Wavefront OBJ file.
        """
        if self.mask_stack is None:
            print("No mask stack available.")
            return

        try:
            verts, faces = self._build_mesh()
        except ImportError as e:
            print(str(e))
            return
        except ValueError as e:
            print(str(e))
            return

        if verts.size == 0 or faces.size == 0:
            print("Cannot export OBJ: volume too small for marching cubes.")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# dicom2ply OBJ export (empty mesh)\n")
            return

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# dicom2ply OBJ export\n")

            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # OBJ is 1‑indexed
            for tri in faces:
                i, j, k = tri + 1
                f.write(f"f {i} {j} {k}\n")

    def get_voxel_coordinates(self) -> np.ndarray:
        """
        Return Nx3 array of voxel center coordinates in patient space.
        Uses true geometric slice positions instead of slice indices.
        """
        if self.mask_stack is None or self.slice_positions is None or not self.contours:
            return np.zeros((0, 3), float)

        ds = self.contours[0].ds

        # Orientation vectors
        row_dir = np.array(ds.ImageOrientationPatient[:3], float)
        col_dir = np.array(ds.ImageOrientationPatient[3:], float)
        normal = np.cross(row_dir, col_dir)

        # Pixel spacing
        spacing_x = float(ds.PixelSpacing[0])
        spacing_y = float(ds.PixelSpacing[1])

        # True slice positions (already sorted)
        slice_positions = np.asarray(self.slice_positions, float)

        # Origin of the first slice
        origin = np.array(ds.ImagePositionPatient, float)
        origin_proj = origin.dot(normal)

        # Mask voxel indices
        rr, cc, zz = np.where(self.mask_stack > 0)

        # Map slice index -> true Z position along normal
        z_phys = slice_positions[zz]

        # Compute coordinates (optionally could add 0.5 for voxel centers)
        coords = (
            origin
            + rr[:, None] * row_dir * spacing_x
            + cc[:, None] * col_dir * spacing_y
            + (z_phys[:, None] - origin_proj) * normal
        )

        return coords
