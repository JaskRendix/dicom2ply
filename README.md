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
- ROI selection:
  - `--names` for explicit ROI lists  
  - `--filter-name` for exact matches  
  - `--filter-pattern` for glob patterns  
- ROI inspection:
  - `--list-rois` prints ROI names and basic stats  
- Additional export formats:
  - Binary mask NIfTI (`--nifti`)
  - Float32 mask NIfTI (`--float-nifti`)
  - ROI statistics JSON (`--json`)
  - Minimal metadata JSON (`--save-metadata`)
  - PNG mask slices (`--png-slices`)
  - Marching‑cubes meshes: PLY, STL, OBJ (`--mesh`, `--stl`, `--obj`)
  - Voxel coordinates (`--coords`)
- Batch processing via YAML config (`--config`)
- Structured logging with progress indicators
- Modular codebase (`contour`, `roi`, `ct_cache`, `geometry`, `masking`, `patient`, `ply_writer`)
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

Optional dependencies:

```
pip install dicom2ply[imageio]
pip install dicom2ply[nifti]
pip install dicom2ply[yaml]
pip install dicom2ply[test]
```

---

## Command-line usage

```
dicom2ply <dicom_dir> <output_dir> [options]
```

### Required arguments

- `dicom_dir`: directory containing CT slices and one RTSTRUCT  
- `output_dir`: directory where output files are written  

---

## ROI selection

```
--names ROI1 ROI2 ...
--filter-name ROI1 --filter-name ROI2
--filter-pattern "*GTV*"
```

If no selection flags are provided, all ROIs are exported.

---

## ROI inspection

```
--list-rois
```

Prints ROI names and basic statistics, then exits.

---

## Export options

| Flag | Output |
|------|--------|
| `--nifti` | Binary mask NIfTI |
| `--float-nifti` | Float32 mask NIfTI |
| `--json` | ROI statistics JSON |
| `--save-metadata` | Minimal metadata JSON |
| `--png-slices` | PNG mask slices |
| `--mesh` | PLY mesh |
| `--stl` | STL mesh |
| `--obj` | OBJ mesh |
| `--coords` | Voxel coordinates `.npy` |

---

## Batch processing

```
--config config.yaml
```

YAML file must contain a list of runs:

```yaml
- dicom_dir: ./dicom
  output_dir: ./out
  filter_name: ["GTV"]
  mesh: true
```

---

## Logging

Default logging prints high‑level progress.

```
--debug
```

Enables detailed logging for CT indexing, geometry checks, contour validation, and mesh generation.

Progress indicators show ROI export position:

```
[1/3] Exporting ROI 'GTV'
```

---

## Example

```
dicom2ply ./dicom ./out \
    --filter-pattern "*TV" \
    --json --mesh --png-slices --save-metadata
```

Produces:

- `roi_GTV.ply`, `roi_CTV.ply`
- `GTV.json`, `CTV.json`
- `GTV_meta.json`, `CTV_meta.json`
- `GTV_mesh.ply`, `CTV_mesh.ply`
- `GTV_slices/...`, `CTV_slices/...`

---

## Python API

```python
from dicom2ply.patient import Patient

p = Patient("/path/to/dicom")

p.dump_ply(directory="/path/to/out")

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

```
pytest
```

Covers:

- ROI geometry reconstruction  
- Contour parsing  
- CT slice lookup  
- PLY writing  
- CLI execution  
- ROI filtering and listing  
- Metadata export  
- YAML config handling  
- Logging and error propagation  

---

## Assumptions

- Axial CT slices with consistent spacing  
- Standard RTSTRUCT contour encoding  
- No gantry tilt  
- No interpolation between missing slices  
- Mesh export requires a non‑empty mask volume  
