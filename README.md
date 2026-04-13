# dicom2ply

`dicom2ply` converts DICOM RT Structure Set (RTSTRUCT) contours into PLY point clouds. Each ROI is exported as a separate PLY file for downstream meshing or geometric processing.

This repository is a modernized and modular rewrite of the original project by Christopher M. Poole:  
https://github.com/christopherpoole/dicom2ply

The current implementation uses Python 3, pydicom, NumPy, scikit‑image, and `plyfile`, and follows a `src/` layout with full test coverage.

---

## Features

- Extracts ROI contours from DICOM RTSTRUCT files  
- Reconstructs 3D coordinates using CT metadata  
- Outputs one binary PLY file per ROI  
- Provides a command‑line interface (`dicom2ply`)  
- Includes a modular codebase (`contour`, `roi`, `geometry`, `ct_cache`, `masking`, `ply_writer`)  
- Includes a full test suite and GitHub Actions workflow  

---

## Installation

```
pip install dicom2ply
```

Or from source:

```
pip install -e .
```

Requires Python 3.10 or newer.

---

## Command‑Line Usage

```
dicom2ply <dicom_dir> <output_dir>
```

`dicom_dir` must contain CT slices and one RTSTRUCT file.  
`output_dir` will receive one `roi_<name>.ply` file per ROI.

---

## Python API

```python
from dicom2ply.patient import Patient

p = Patient("/path/to/dicom")
p.dump_ply(directory="/path/to/output")
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

Tests cover ROI geometry, contour handling, CT slice lookup, PLY writing, and CLI execution.

---

## Relation to the Original Project

This repository maintains the purpose and name of the original `dicom2ply` project by Christopher M. Poole.  
The implementation has been rewritten to:

- support Python 3  
- remove legacy dependencies  
- use a modular architecture  
- provide a CLI entry point  
- include automated tests  
- follow modern packaging standards  

The original repository remains available for reference.
