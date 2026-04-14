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
- Optional ROI filtering via `--names`  
- Additional ROI exports:  
  - Binary mask NIfTI (`--nifti`)  
  - Float32 mask NIfTI (`--float-nifti`)  
  - ROI statistics as JSON (`--json`)  
  - Mask slices as PNG images (`--png-slices`)  
  - Marching‑cubes mesh as PLY (`--mesh`)  
  - Voxel coordinates as `.npy` (`--coords`)  
- Command‑line interface (`dicom2ply`)  
- Modular architecture (`contour`, `roi`, `ct_cache`, `patient`, `ply_writer`)  
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
dicom2ply <dicom_dir> <output_dir> [options]
```

### Required arguments

- `dicom_dir` — directory containing CT slices and one RTSTRUCT  
- `output_dir` — directory where output files are written  

### Optional ROI selection

```
--names ROI1 ROI2 ...
```

Exports only the listed ROIs.  
If omitted, all ROIs in the RTSTRUCT are processed.

### Export options

| Flag | Output |
|------|--------|
| `--nifti` | Binary mask NIfTI (`<name>.nii.gz`) |
| `--float-nifti` | Float32 mask NIfTI (`<name>_float.nii.gz`) |
| `--json` | ROI statistics (`<name>.json`) |
| `--png-slices` | PNG mask slices (`<name>_slices/…`) |
| `--mesh` | Marching‑cubes mesh (`<name>_mesh.ply`) |
| `--coords` | Voxel coordinates (`<name>_coords.npy`) |

### Example

```
dicom2ply ./dicom ./out \
    --names GTV CTV \
    --json --mesh --png-slices
```

This writes:

- `roi_GTV.ply`, `roi_CTV.ply`  
- `GTV.json`, `CTV.json`  
- `GTV_mesh.ply`, `CTV_mesh.ply`  
- `GTV_slices/…`, `CTV_slices/…`

---

## Python API

```python
from dicom2ply.patient import Patient

p = Patient("/path/to/dicom")

# Export PLY only
p.dump_ply(directory="/path/to/out")

# Export PLY + binary NIfTI
p.dump_ply(directory="/path/to/out", export_nifti=True)

# Access ROI objects for additional exports
roi = p.get_roi("GTV")
roi.export_mask_nifti_float("GTV_float.nii.gz")
roi.export_all_slices_png("GTV_slices")
roi.export_mesh_ply("GTV_mesh.ply")
roi.export_json()
roi.get_voxel_coordinates()
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
- Mesh export requires at least a 2×2×2 mask volume; thin ROIs produce an empty mesh file.

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
