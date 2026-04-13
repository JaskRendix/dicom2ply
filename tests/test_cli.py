import shutil
import subprocess


def test_cli_runs(tmp_path, synthetic_ct, synthetic_rtstruct):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    shutil.copy(synthetic_ct, dicom_dir / "CT.dcm")
    shutil.copy(synthetic_rtstruct, dicom_dir / "RS.dcm")

    output = tmp_path / "out"
    output.mkdir()

    result = subprocess.run(
        ["dicom2ply", str(dicom_dir), str(output)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
