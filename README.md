# dicom2ply

`dicom2ply` converts DICOM RT Structure Set (RTSTRUCT) contours into PLY point clouds.  
Each ROI is exported as a separate binary PLY file for downstream meshing or geometric processing.

This project is a modern, modular rewrite of the original implementation by Christopher M. Poole  
(<https://github.com/christopherpoole/dicom2ply>), updated for Python 3 and current tooling.

---

## Features

- Extracts ROI contours from DICOM RTSTRUCT files  
- Reconstructs 3D coordinates using CT slice metadata  
- Outputs one binary PLY file per ROI  
- Optional ROI filtering via `--names` or `names=[...]`  
- Command‑line interface (`dicom2ply`)  
- Modular architecture (`contour`, `roi`, `geometry`, `ct_cache`, `masking`, `ply_writer`)  
- Full test suite and GitHub Actions CI  
- Modern `src/` layout and Python packaging

---

## Installation

```
pip install dicom2ply
```

Or from source:

```
pip install -e .
```

Requires Python ≥ 3.10.

---

## Command‑Line Usage

```
dicom2ply <dicom_dir> <output_dir> [--names ROI1 ROI2 ...]
```

- `dicom_dir` must contain CT slices and one RTSTRUCT file  
- `output_dir` receives one `roi_<name>.ply` file per ROI  
- `--names` restricts export to specific ROI names (optional)

Example:

```
dicom2ply ./dicom ./ply --names GTV CTV PTV
```

If any requested ROI name is missing, the tool reports the missing names and lists available ROIs.

---

## Python API

```python
from dicom2ply.patient import Patient

p = Patient("/path/to/dicom")

# Export all ROIs
p.dump_ply(directory="/path/to/output")

# Export only selected ROIs
p.dump_ply(directory="/path/to/output", names=["GTV", "PTV"])
```

---

## Project Structure

```
src/dicom2ply/
    cli.py
    contour.py
    ct_cache.py
    geometry.py
    masking.py
    patient.py
    ply_writer.py
    roi.py
tests/
.github/workflows/tests.yml
pyproject.toml
README.md
```

---

## Tests

Run the full suite:

```
pytest
```

Tests cover:

- ROI geometry reconstruction  
- Contour parsing and validation  
- CT slice lookup and metadata handling  
- PLY writing  
- CLI execution and ROI filtering  

---

## Supported Assumptions

- Axial CT slices with consistent spacing  
- Standard RTSTRUCT contour encoding  
- No gantry tilt  
- No interpolation between missing slices  

---

## Relation to the Original Project

This repository preserves the purpose and name of the original `dicom2ply` project by Christopher M. Poole.  
The implementation has been fully rewritten to:

- support Python 3  
- remove legacy dependencies  
- adopt a modular architecture  
- provide a CLI entry point  
- include automated tests  
- follow modern packaging and `src/` layout conventions  

The original repository remains available for reference.
