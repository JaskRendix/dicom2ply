import argparse
from pathlib import Path

from dicom2ply.patient import Patient


def main():
    parser = argparse.ArgumentParser(
        description="Convert DICOM RTSTRUCT to PLY point clouds"
    )
    parser.add_argument("dicom_dir", help="Directory containing DICOM files")
    parser.add_argument("output_dir", help="Directory to write PLY files")
    parser.add_argument(
        "--names", nargs="*", help="Optional list of ROI names to export"
    )
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    output_dir = Path(args.output_dir)

    if not dicom_dir.is_dir():
        raise SystemExit(f"Input directory not found: {dicom_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        patient = Patient(str(dicom_dir))
        patient.dump_ply(directory=str(output_dir), names=args.names)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0
