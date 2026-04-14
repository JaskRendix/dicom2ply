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
            if c.stats.mean is not None:  # skip empty masks
                contours.append(c)

        # Sort contours deterministically by slice position
        contours.sort(key=lambda c: c.slice_pos)

        obj = cls(name=name, contours=contours, bins=bins)

        obj.compute_extent()  # geometry only
        obj.compute_mask_stack()  # build 3D mask
        obj.compute_stats()  # HU stats
        obj.compute_volume_stats()  # physical volume stats

        return obj

    def compute_stats(self):
        if not self.contours:
            self.count = 0
            return

        values = np.concatenate([c.masked_values for c in self.contours])
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
        if not self.contours:
            return

        # Avoid pixel_array decode: use metadata
        ds0 = self.contours[0].ds
        rows = int(ds0.Rows)
        cols = int(ds0.Columns)

        positions = np.array([c.slice_pos for c in self.contours])
        uniq = np.unique(positions)
        uniq.sort()
        self.slice_positions = uniq

        pos_to_idx = {p: i for i, p in enumerate(uniq)}
        volume = np.zeros((rows, cols, len(uniq)), np.int8)

        for c in self.contours:
            idx = pos_to_idx[c.slice_pos]

            if c.mask.shape != (rows, cols):
                continue

            volume[..., idx] |= c.mask

        self.mask_stack = volume

    def export_nifti(self, output_path: str | Path) -> None:
        """
        Export the ROI mask stack as a NIfTI volume.
        Uses DICOM orientation + spacing from the first contour's dataset.
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

        # Orientation vectors
        row_dir = np.array(ds.ImageOrientationPatient[:3], float)
        col_dir = np.array(ds.ImageOrientationPatient[3:], float)
        normal = np.cross(row_dir, col_dir)

        # Spacing
        spacing_x = float(ds.PixelSpacing[0])
        spacing_y = float(ds.PixelSpacing[1])

        if self.slice_positions is not None and len(self.slice_positions) > 1:
            spacing_z = float(np.abs(np.diff(self.slice_positions)).mean())
        else:
            spacing_z = float(getattr(ds, "SliceThickness", 1.0))

        # Affine construction
        affine = np.eye(4)
        affine[:3, 0] = -row_dir * spacing_x
        affine[:3, 1] = -col_dir * spacing_y
        affine[:3, 2] = normal * spacing_z

        origin = np.array(ds.ImagePositionPatient, float)
        affine[:3, 3] = origin

        # NIfTI expects (X, Y, Z) = (cols, rows, slices)
        # Your mask_stack is (rows, cols, slices)
        data = np.swapaxes(self.mask_stack, 0, 1).astype(np.uint8)

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

        # FIX: binarize before scaling to avoid int8 overflow
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
            # FIX: binarize before scaling
            mask = (self.mask_stack[..., i] > 0).astype(np.uint8)
            img = mask * 255

            imageio.imwrite(out / f"{self.name}_slice_{i:04d}.png", img)

    def export_mesh_ply(self, output_path: str | Path) -> None:
        """
        Export a surface mesh (marching cubes) of the ROI mask as a PLY file.
        """
        try:
            from plyfile import PlyData, PlyElement
            from skimage import measure
        except ImportError:
            print("Cannot export mesh: skimage or plyfile not installed.")
            return

        if self.mask_stack is None:
            print("No mask stack available.")
            return

        vol = np.swapaxes(self.mask_stack, 0, 2)

        if vol.shape[0] < 2 or vol.shape[1] < 2 or vol.shape[2] < 2:
            from plyfile import PlyData, PlyElement

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

        verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)

        verts = verts.astype(np.float32)
        faces = faces.astype(np.int32)

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

    def export_mask_nifti_float(self, output_path: str | Path) -> None:
        """
        Export the ROI mask as a float32 NIfTI volume.
        """
        try:
            import nibabel as nib
        except ImportError:
            print("Cannot export NIfTI: nibabel not installed.")
            return

        if self.mask_stack is None:
            print("No mask data available.")
            return

        ds = self.contours[0].ds

        row_dir = np.array(ds.ImageOrientationPatient[:3], float)
        col_dir = np.array(ds.ImageOrientationPatient[3:], float)
        normal = np.cross(row_dir, col_dir)

        spacing_x = float(ds.PixelSpacing[0])
        spacing_y = float(ds.PixelSpacing[1])
        spacing_z = (
            float(np.abs(np.diff(self.slice_positions)).mean())
            if self.slice_positions is not None and len(self.slice_positions) > 1
            else float(getattr(ds, "SliceThickness", 1.0))
        )

        affine = np.eye(4)
        affine[:3, 0] = -row_dir * spacing_x
        affine[:3, 1] = -col_dir * spacing_y
        affine[:3, 2] = normal * spacing_z
        affine[:3, 3] = np.array(ds.ImagePositionPatient, float)

        data = np.swapaxes(self.mask_stack, 0, 1).astype(np.float32)

        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        img.header.set_qform(affine, code=1)
        nib.save(img, str(output_path))

    def get_voxel_coordinates(self) -> np.ndarray:
        """
        Return Nx3 array of voxel center coordinates in patient space.
        """
        if self.mask_stack is None:
            return np.zeros((0, 3), float)

        ds = self.contours[0].ds

        row_dir = np.array(ds.ImageOrientationPatient[:3], float)
        col_dir = np.array(ds.ImageOrientationPatient[3:], float)
        normal = np.cross(row_dir, col_dir)

        spacing_x = float(ds.PixelSpacing[0])
        spacing_y = float(ds.PixelSpacing[1])
        spacing_z = (
            float(np.abs(np.diff(self.slice_positions)).mean())
            if len(self.slice_positions) > 1
            else float(getattr(ds, "SliceThickness", 1.0))
        )

        origin = np.array(ds.ImagePositionPatient, float)

        rows, cols, slices = self.mask_stack.shape
        rr, cc, zz = np.where(self.mask_stack > 0)

        coords = (
            origin
            + rr[:, None] * row_dir * spacing_x
            + cc[:, None] * col_dir * spacing_y
            + zz[:, None] * normal * spacing_z
        )

        return coords
