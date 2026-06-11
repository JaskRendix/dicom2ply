# dicom2ply

`dicom2ply` converts DICOM RT Structure Set (RTSTRUCT) contours into 3D geometric data.  
Each ROI can be exported as point clouds, meshes, masks, slices, or voxel coordinates.

Modern rewrite of the original project by Christopher M. Poole (<https://github.com/christopherpoole/dicom2ply>).

---

## Features

- Extracts ROI contours from RTSTRUCT  
- Reconstructs 3D coordinates using CT slice geometry  
- Exports one binary PLY point cloud per ROI  
- ROI selection:
  - `--names`
  - `--filter-name`
  - `--filter-pattern`
- ROI inspection:
  - `--list-rois`
- Export formats:
  - Binary mask NIfTI (`--nifti`)
  - Float32 mask NIfTI (`--float-nifti`)
  - ROI statistics JSON (`--json`)
  - Minimal metadata JSON (`--save-metadata`)
  - PNG mask slices (`--png-slices`)
  - Marching‑cubes meshes: PLY, STL, OBJ (`--mesh`, `--stl`, `--obj`)
  - Voxel coordinates (`--coords`)
  - Extended exporters:
    - HU→RGB PLY (`--ply-rgb`)
    - LAS/LAZ point clouds (`--las`)
    - Triangulated contour meshes (`--tri-mesh`)
- Batch processing via YAML (`--config`)
- Structured logging with progress indicators
- Modular codebase (`contour`, `roi`, `ct_cache`, `geometry`, `masking`, `patient`, `exporters`)
- Full test suite and CI
- Modern `src/` layout and packaging

---

## Installation

```
pip install -e .
```

Requires Python 3.10+.

Optional dependencies:

```
pip install dicom2ply[imageio]
pip install dicom2ply[nifti]
pip install dicom2ply[yaml]
pip install dicom2ply[laspy]
pip install dicom2ply[test]
```

---

## Command-line usage

```
dicom2ply <dicom_dir> <output_dir> [options]
```

### Required arguments

- `dicom_dir`: directory containing CT slices and one RTSTRUCT  
- `output_dir`: directory for output files  

---

## ROI selection

```
--names ROI1 ROI2
--filter-name ROI1
--filter-pattern "*GTV*"
```

If no selection flags are provided, all ROIs are exported.

---

## ROI inspection

```
--list-rois
```

Prints ROI names and basic statistics.

---

## Export options

| Flag | Output |
|------|--------|
| `--nifti` | Binary mask NIfTI |
| `--float-nifti` | Float32 mask NIfTI |
| `--json` | ROI statistics JSON |
| `--save-metadata` | Minimal metadata JSON |
| `--png-slices` | PNG mask slices |
| `--mesh` | Marching‑cubes PLY mesh |
| `--stl` | STL mesh |
| `--obj` | OBJ mesh |
| `--coords` | Voxel coordinates `.npy` |
| `--ply-rgb` | HU→RGB PLY point cloud |
| `--las` | LAS/LAZ point cloud |
| `--tri-mesh` | Triangulated contour mesh PLY |

---

## Batch processing

```
--config config.yaml
```

Example:

```yaml
- dicom_dir: ./dicom
  output_dir: ./out
  filter_name: ["GTV"]
  mesh: true
```

---

## Logging

```
--debug
```

Enables detailed logging.

Progress indicator:

```
[1/3] Exporting ROI 'GTV'
```

---

## Example

```
dicom2ply ./dicom ./out \
    --filter-pattern "*TV" \
    --json --mesh --png-slices --save-metadata --ply-rgb --las --tri-mesh
```

Produces:

- `roi_GTV.ply`, `roi_CTV.ply`
- `roi_GTV_points.ply` (RGB)
- `roi_GTV.las`
- `roi_GTV_mesh.ply` (triangulated)
- `GTV.json`, `CTV.json`
- `GTV_meta.json`, `CTV_meta.json`
- `GTV_mesh.ply`, `CTV_mesh.ply` (marching‑cubes)
- `GTV_slices/...`, `CTV_slices/...`

---

## Python API

```python
from dicom2ply.patient import Patient

p = Patient("/path/to/dicom")

p.dump_ply(directory="/path/to/out")
p.dump_exporters(directory="/path/to/out", export_ply_rgb=True, export_las=True)

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
    exporters.py
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
- Extended exporters (PLY‑RGB, LAS, triangulated mesh)  
- CLI execution  
- ROI filtering and listing  
- Metadata export  
- YAML config  
- Logging and error propagation  

---

## Assumptions

- Axial CT slices with consistent spacing  
- Standard RTSTRUCT contour encoding  
- No gantry tilt  
- No interpolation between missing slices  
- Marching‑cubes mesh export requires a non‑empty mask volume  
