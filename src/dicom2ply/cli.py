from __future__ import annotations

import argparse
import fnmatch
import json
import logging
from collections.abc import Iterable
from pathlib import Path

from dicom2ply.patient import Patient

logger = logging.getLogger("dicom2ply.cli")


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

    # Export options
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

    # Debug flag
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output during DICOM parsing",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Print geometry validation summary (implies verbose output)",
    )

    parser.add_argument(
        "--strict-geometry",
        action="store_true",
        help="Treat geometry mismatches as errors instead of warnings",
    )

    parser.add_argument(
        "--slice-tolerance",
        type=float,
        default=0.01,
        help="Slice position matching tolerance in mm (default: 0.01)",
    )

    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=4096,
        help="Number of histogram bins used for HU statistics (default: 4096)",
    )

    parser.add_argument(
        "--planarity-tolerance",
        type=float,
        default=1e-2,
        help="Planarity tolerance in mm for contour planarity checks (default: 1e-2)",
    )

    parser.add_argument(
        "--list-rois",
        action="store_true",
        help="List available ROI names and basic stats, then exit",
    )

    parser.add_argument(
        "--filter-name",
        action="append",
        default=None,
        help="Filter ROIs by exact name; can be passed multiple times or comma-separated",
    )

    parser.add_argument(
        "--filter-pattern",
        action="append",
        default=None,
        help="Filter ROIs by glob-like pattern (e.g., '*GTV*'); can be passed multiple times",
    )

    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save minimal ROI metadata JSON alongside exports",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file describing one or more runs",
    )

    return parser.parse_args()


def configure_logging(debug: bool) -> None:
    """Configure global logging based on --debug flag."""
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

    logging.addLevelName(logging.ERROR, "Error")


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
    patient = Patient(
        str(dicom_dir),
        debug=args.debug,
        bins=args.histogram_bins,
        slice_tol=args.slice_tolerance,
        planarity_tol=args.planarity_tolerance,
        validate=args.validate,
        strict_geometry=args.strict_geometry,
    )
    # Validate requested names
    validated = safe_validate_roi_names(patient, names)

    # Apply filter-name and filter-pattern to determine final ROI list
    available = list(patient.roi_names)
    if validated is not None:
        candidate_names = list(validated)
    else:
        candidate_names = available

    # Expand --filter-name (allow comma-separated values)
    filter_names = set()
    if getattr(args, "filter_name", None):
        for entry in args.filter_name:
            parts = [p.strip() for p in entry.split(",") if p.strip()]
            filter_names.update(parts)

    # Combine pattern filters
    filter_patterns = []
    if getattr(args, "filter_pattern", None):
        for pat in args.filter_pattern:
            filter_patterns.append(pat)

    def name_allowed(n: str) -> bool:
        if filter_names:
            if n not in filter_names:
                return False
        if filter_patterns:
            ok = any(fnmatch.fnmatchcase(n, p) for p in filter_patterns)
            if not ok:
                return False
        return True

    final_names = [n for n in candidate_names if name_allowed(n)]

    # If list-rois requested, print and return
    if getattr(args, "list_rois", False):
        logger.info("Available ROIs:")
        for n in final_names:
            try:
                roi = patient.get_roi(n)
                voxels = getattr(roi, "num_voxels", None)
                vol = getattr(roi, "volume_cc", None)
            except Exception:
                voxels = None
                vol = None
            logger.info(f"  {n}: voxels={voxels}, volume_cc={vol}")
        return

    logger.info("Exporting PLY files...")
    patient.dump_ply(
        directory=str(output_dir), names=final_names, export_nifti=args.nifti
    )

    # Per-ROI exports with progress reporting
    total = len(final_names)
    for idx, name in enumerate(final_names):
        logger.info(f"[{idx+1}/{total}] Exporting ROI '{name}'")
        roi = patient.get_roi(name)

        if args.float_nifti:
            logger.info(f"Exporting float NIfTI for ROI '{name}'")
            roi.export_mask_nifti_float(output_dir / f"{name}_float.nii.gz")

        if args.json:
            logger.info(f"Exporting JSON stats for ROI '{name}'")

            with open(output_dir / f"{name}.json", "w") as f:
                json.dump(roi.export_json(), f, indent=2)

        if args.png_slices:
            logger.info(f"Exporting PNG slices for ROI '{name}'")
            roi.export_all_slices_png(output_dir / f"{name}_slices")

        if args.mesh:
            logger.info(f"Exporting PLY mesh for ROI '{name}'")
            roi.export_mesh_ply(output_dir / f"{name}_mesh.ply")

        if args.stl:
            logger.info(f"Exporting STL mesh for ROI '{name}'")
            roi.export_mesh_stl(output_dir / f"{name}_mesh.stl")

        if args.obj:
            logger.info(f"Exporting OBJ mesh for ROI '{name}'")
            roi.export_mesh_obj(output_dir / f"{name}_mesh.obj")

        if args.coords:
            logger.info(f"Exporting voxel coordinates for ROI '{name}'")
            coords = roi.get_voxel_coordinates()
            npy_path = output_dir / f"{name}_coords.npy"
            import numpy as np

            np.save(npy_path, coords)

        # Minimal metadata export
        if getattr(args, "save_metadata", False):
            meta = {
                "roi_name": name,
                "patient_id": getattr(patient.structure, "PatientID", None),
                "study_date": getattr(patient.structure, "StudyDate", None),
                "num_voxels": getattr(roi, "num_voxels", None),
                "volume_cc": getattr(roi, "volume_cc", None),
            }
            meta_path = output_dir / f"{name}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)


def main() -> int:
    args = parse_args()
    configure_logging(args.debug)

    try:
        # If a batch config is provided, parse and run each entry
        if getattr(args, "config", None):
            try:
                import yaml
            except Exception:
                raise RuntimeError(
                    "YAML config support requires PyYAML (install optional dependency)"
                )

            cfg = yaml.safe_load(args.config.read_text())
            if not isinstance(cfg, list):
                raise RuntimeError("Config file must be a YAML list of run definitions")

            for run in cfg:
                # Each run should specify dicom_dir and output_dir at minimum
                run_dic = Path(run.get("dicom_dir"))
                run_out = Path(run.get("output_dir"))
                # Merge run-specific options into a namespace-like object
                run_args = argparse.Namespace(**vars(args))
                # Override with values from config
                for k, v in run.items():
                    setattr(run_args, k, v)

                validate_paths(run_dic, run_out)
                run_conversion(
                    run_dic, run_out, getattr(run_args, "names", None), run_args
                )
        else:
            validate_paths(args.dicom_dir, args.output_dir)
            run_conversion(args.dicom_dir, args.output_dir, args.names, args)
    except Exception as e:
        logger.error(f"{e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
