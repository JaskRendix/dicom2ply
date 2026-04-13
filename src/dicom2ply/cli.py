from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from dicom2ply.patient import Patient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DICOM RTSTRUCT to PLY point clouds"
    )
    parser.add_argument("dicom_dir", type=Path, help="Directory containing DICOM files")
    parser.add_argument("output_dir", type=Path, help="Directory to write PLY files")
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional list of ROI names to export",
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

    # Only validate if roi_names is iterable
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
) -> None:
    patient = Patient(str(dicom_dir))
    validated = safe_validate_roi_names(patient, names)
    patient.dump_ply(directory=str(output_dir), names=validated)


def main() -> int:
    args = parse_args()

    try:
        validate_paths(args.dicom_dir, args.output_dir)
        run_conversion(args.dicom_dir, args.output_dir, args.names)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
