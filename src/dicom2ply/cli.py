from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from dicom2ply.patient import Patient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DICOM RTSTRUCT to PLY point clouds and optional ROI exports"
    )
    parser.add_argument("dicom_dir", type=Path, help="Directory containing DICOM files")
    parser.add_argument("output_dir", type=Path, help="Directory to write output files")

    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional list of ROI names to export",
    )

    # New export options
    parser.add_argument("--nifti", action="store_true", help="Export binary mask NIfTI")
    parser.add_argument(
        "--float-nifti", action="store_true", help="Export float32 mask NIfTI"
    )
    parser.add_argument(
        "--json", action="store_true", help="Export ROI statistics as JSON"
    )
    parser.add_argument(
        "--png-slices", action="store_true", help="Export mask slices as PNG images"
    )
    parser.add_argument(
        "--mesh", action="store_true", help="Export marching-cubes mesh as PLY"
    )
    parser.add_argument(
        "--stl", action="store_true", help="Export marching-cubes mesh as STL"
    )
    parser.add_argument(
        "--obj", action="store_true", help="Export marching-cubes mesh as OBJ"
    )
    parser.add_argument(
        "--coords", action="store_true", help="Export voxel coordinates as .npy"
    )

    return parser.parse_args()


def validate_paths(dicom_dir: Path, output_dir: Path) -> None:
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {dicom_dir}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Failed to create output directory {output_dir}: {e}"
        ) from e


def safe_validate_roi_names(
    patient: Patient,
    requested: Iterable[str] | None,
) -> list[str] | None:
    if requested is None:
        return None

    roi_names_attr = getattr(patient, "roi_names", None)

    if not isinstance(roi_names_attr, Iterable):
        return [*requested]

    available = {str(name) for name in roi_names_attr}
    missing = [name for name in requested if name not in available]

    if missing:
        raise ValueError(
            f"ROI names not found: {missing}\nAvailable ROIs: {sorted(available)}"
        )

    return [*requested]


def run_conversion(
    dicom_dir: Path,
    output_dir: Path,
    names: Iterable[str] | None,
    args: argparse.Namespace,
) -> None:
    patient = Patient(str(dicom_dir))
    validated = safe_validate_roi_names(patient, names)

    patient.dump_ply(
        directory=str(output_dir), names=validated, export_nifti=args.nifti
    )

    for name in validated or patient.roi_names:
        roi = patient.get_roi(name)

        if args.float_nifti:
            roi.export_mask_nifti_float(output_dir / f"{name}_float.nii.gz")

        if args.json:
            import json

            with open(output_dir / f"{name}.json", "w") as f:
                json.dump(roi.export_json(), f, indent=2)

        if args.png_slices:
            roi.export_all_slices_png(output_dir / f"{name}_slices")

        if args.mesh:
            roi.export_mesh_ply(output_dir / f"{name}_mesh.ply")

        if args.stl:
            roi.export_mesh_stl(output_dir / f"{name}_mesh.stl")

        if args.obj:
            roi.export_mesh_obj(output_dir / f"{name}_mesh.obj")

        if args.coords:
            coords = roi.get_voxel_coordinates()
            npy_path = output_dir / f"{name}_coords.npy"
            import numpy as np

            np.save(npy_path, coords)


def main() -> int:
    args = parse_args()

    try:
        validate_paths(args.dicom_dir, args.output_dir)
        run_conversion(args.dicom_dir, args.output_dir, args.names, args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
