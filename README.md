# dicom2ply

`dicom2ply` converts DICOM RT Structure Set (RTSTRUCT) contours into 3D geometric data.  
Each ROI can be exported as a PLY point cloud, a NIfTI mask, a mesh, PNG slices, or voxel coordinates.

This is a modern rewrite of the original project by Christopher M. Poole  
(<https://github.com/christopherpoole/dicom2ply>), updated for Python 3 and a modular, tested architecture.

---

## Features

- Extracts ROI contours from DICOM RTSTRUCT files  
- Reconstructs 3D coordinates using CT slice geometry  
- Exports one binary PLY file per ROI  
- Optional ROI filtering (`--names`)  
- Additional export formats:
  - Binary mask NIfTI (`--nifti`)
  - Float32 mask NIfTI (`--float-nifti`)
  - ROI statistics as JSON (`--json`)
  - Mask slices as PNG (`--png-slices`)
  - Marching‑cubes meshes: PLY, STL, OBJ (`--mesh`, `--stl`, `--obj`)
  - Voxel coordinates as `.npy` (`--coords`)
- Structured logging with optional verbose mode (`--debug`)
- Modular codebase (`contour`, `roi`, `ct_cache`, `patient`, `ply_writer`)
- Full test suite and GitHub Actions CI
- Modern `src/` layout and Python packaging

---

## Installation

```
pip install dicom2ply   # not yet on PyPI
```

From source:

```
pip install -e .
```

Requires Python 3.10 or newer.

---

## Command-line usage

```
dicom2ply <dicom_dir> <output_dir> [options]
```

### Required arguments

- `dicom_dir`: directory containing CT slices and one RTSTRUCT  
- `output_dir`: directory where output files are written  

### ROI selection

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
| `--png-slices` | PNG mask slices (`<name>_slices/...`) |
| `--mesh` | Marching‑cubes mesh (`<name>_mesh.ply`) |
| `--stl` | STL mesh (`<name>_mesh.stl`) |
| `--obj` | OBJ mesh (`<name>_mesh.obj`) |
| `--coords` | Voxel coordinates (`<name>_coords.npy`) |

### Logging

By default, the tool prints only high‑level progress and errors.

Verbose diagnostic logging:

```
--debug
```

This prints details about CT slice indexing, geometry checks, planarity validation, skipped contours, missing slices, and mesh generation.

Example:

```
dicom2ply ./dicom ./out --debug
```

---

## Example

```
dicom2ply ./dicom ./out \
    --names GTV CTV \
    --json --mesh --png-slices
```

Produces:

- `roi_GTV.ply`, `roi_CTV.ply`
- `GTV.json`, `CTV.json`
- `GTV_mesh.ply`, `CTV_mesh.ply`
- `GTV_slices/...`, `CTV_slices/...`

---

## Python API

```python
from dicom2ply.patient import Patient

p = Patient("/path/to/dicom")

# Export PLY only
p.dump_ply(directory="/path/to/out")

# Export PLY + binary NIfTI
p.dump_ply(directory="/path/to/out", export_nifti=True)

# Access ROI objects
roi = p.get_roi("GTV")
roi.export_mask_nifti_float("GTV_float.nii.gz")
roi.export_all_slices_png("GTV_slices")
roi.export_mesh_ply("GTV_mesh.ply")
roi.export_json()
roi.get_voxel_coordinates()
```

---

## Project structure

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

Run:

```
pytest
```

Tests cover:

- ROI geometry reconstruction  
- Contour parsing and validation  
- CT slice lookup and metadata handling  
- PLY writing  
- CLI execution and ROI filtering  
- Logging and error propagation  

---

## Assumptions

- Axial CT slices with consistent spacing  
- Standard RTSTRUCT contour encoding  
- No gantry tilt  
- No interpolation between missing slices  
- Mesh export requires at least a 2×2×2 mask volume  
  (thin ROIs produce an empty mesh)

---

## Relation to the original project

This project keeps the purpose and name of the original `dicom2ply` by Christopher M. Poole.  
The implementation has been rewritten to support Python 3, remove legacy dependencies, adopt a modular architecture, provide a CLI entry point, include automated tests, and follow modern packaging conventions.
