import argparse
from dicom2ply.patient import Patient

def main():
    parser = argparse.ArgumentParser(description="Convert DICOM RTSTRUCT to PLY point clouds")
    parser.add_argument("dicom_dir", help="Directory containing DICOM files")
    parser.add_argument("output_dir", help="Directory to write PLY files")
    args = parser.parse_args()

    patient = Patient(args.dicom_dir)
    patient.dump_ply(directory=args.output_dir)
