# Multi Compartment Segmentation (MultiC)

A [Detectron2](https://github.com/facebookresearch/detectron2)-based panoptic segmentation model that segments kidney Whole Slide Images (WSI) into 6 functional tissue units (FTUs):

| Class | Description |
|---|---|
| `cortical_interstitium` | Cortical interstitial tissue |
| `medullary_interstitium` | Medullary interstitial tissue |
| `non_globally_sclerotic_glomeruli` | Non-sclerotic glomeruli |
| `globally_sclerotic_glomeruli` | Globally sclerotic glomeruli |
| `tubules` | Tubular structures |
| `arteries/arterioles` | Arteries and arterioles |

The codebase supports two deployment modes:

| Mode | Description | Docker image |
|---|---|---|
| **Girder plugin** | Runs as a DSA/HistomicsTK plugin — inputs from Girder, annotations uploaded back to DSA | `Dockerfile` |
| **Standalone (notebook)** | Girder-free — inputs are local file paths, outputs are JSON files on disk | `Dockerfile.notebook` |

---

## Repository Structure

```
multic/
├── cli/
│   └── MultiC/
│       ├── MultiC.py               # Girder-coupled CLI entry point (ctk-cli)
│       ├── MultiC.xml              # Slicer CLI parameter definition
│       └── MultiCLocal.py      # Standalone CLI entry point (argparse, no Girder)
├── segmentationschool/
│   ├── segmentation_school.py      # Router: train / predict
│   └── Codes/
│       ├── IterativePredict_1X.py       # Core prediction — Girder version
│       ├── IterativePredict_notebook.py # Core prediction — standalone version
│       ├── IterativeTraining_1X.py      # Model training
│       └── xml_to_json.py               # XML → HistomicsTK JSON conversion
├── notebook_runner.py              # Importable Python function for notebook use
Dockerfile                          # Girder plugin image (CUDA 12.1, slicer_cli_web)
Dockerfile.notebook                 # Standalone notebook image (CUDA 12.1, JupyterLab)
requirements-notebook.txt           # Dependencies for standalone image (no Girder)
test_decoupled.py                   # End-to-end test / runner for standalone mode
```

---

## Mode 1 — Girder Plugin (DSA / HistomicsUI)

### Build

```bash
docker build -t dsarchive/multicompartment-segmentation:latest .
```

### Quick Start in HistomicsUI

1. Create an account at [DSA](https://athena.rc.ufl.edu/) and log in.
2. Upload your WSI to a folder under **Collections** or your **User** directory.
3. Open the image in HistomicsUI by clicking the arrow icon.
4. From the **Analyses** tab select: `sarderlab/ComPRePS/segmentation/MultiC`
5. Populate the fields:
   - **Input Image** — select your WSI from DSA
   - **Segmentation Model File** — select the pretrained model from `Collections/models/segmentation_models/`
6. Click **Submit**. Once complete, segmented compartments will appear as annotation layers on the image.

---

## Mode 2 — Standalone (Jupyter Notebook / No Girder)

Inputs are local file paths. Outputs are per-class JSON annotation files written to a directory you specify.

### Build

```bash
docker build --platform=linux/amd64 -f Dockerfile.notebook \
  -t dsrithad/fusion1_decoupled:multicompartment-segmentation-notebook .
```

### Run segmentation

Mount your local slide, model, and output directory into the container:

```bash
docker run --rm \
  -v /path/to/slide.tif:/input/slide.tif \
  -v /path/to/FUSION_MCS_FFPE_v1.pth:/model/model.pth \
  -v /path/to/output:/output \
  dsrithad/fusion1_decoupled:multicompartment-segmentation-notebook \
  python /opt/MultiC/test_decoupled.py \
    --input_file /input/slide.tif \
    --modelfile  /model/model.pth \
    --output_dir /output/
```

Single-line version:

```bash
docker run --rm -v /path/to/slide.tif:/input/slide.tif -v /path/to/FUSION_MCS_FFPE_v1.pth:/model/model.pth -v /path/to/output:/output dsrithad/fusion1_decoupled:multicompartment-segmentation-notebook python /opt/MultiC/test_decoupled.py --input_file /input/slide.tif --modelfile /model/model.pth --output_dir /output/
```

Optional flags (all default to the same values as the Girder plugin):

| Flag | Default | Description |
|---|---|---|
| `--boxSize` | 2048 | Tile size in pixels |
| `--bordercrop` | 300 | Border pixels zeroed to avoid tile-edge artifacts |
| `--roi_thresh` | 0.01 | Detectron2 detection score threshold |
| `--white_percent` | 0.01 | Minimum tissue fraction required per tile |
| `--overlap_percentHR` | 0 | Tile overlap fraction (0–1) |
| `--Mag20X` | False | Flag: slide is 20X magnification |
| `--no_interstitium` | False | Flag: suppress interstitium output |

### Output

One JSON file per compartment class is saved to `--output_dir`:

```
output/
├── cortical_interstitium.json
├── medullary_interstitium.json
├── non_globally_sclerotic_glomeruli.json
├── globally_sclerotic_glomeruli.json
├── tubules.json
└── arteries_arterioles.json
```

Each file follows the HistomicsTK annotation schema:
```json
{
  "name": "tubules",
  "elements": [
    {
      "type": "polyline",
      "closed": true,
      "points": [[x, y, 0], ...],
      "lineColor": "rgb(0, 255, 128)",
      "fillColor": "rgba(0, 255, 128, 0.4)",
      "group": "Segmented FTU"
    }
  ]
}
```

### Start JupyterLab

```bash
docker run --rm -p 8888:8888 \
  -v /path/to/data:/data \
  dsrithad/fusion1_decoupled:multicompartment-segmentation-notebook
```

Then open `http://localhost:8888` in your browser. From a notebook cell:

```python
from multic.notebook_runner import run_notebook

run_notebook(
    input_file='/data/slide.tif',
    modelfile='/data/FUSION_MCS_FFPE_v1.pth',
    output_dir='/data/outputs/'
)
```

### Push to Docker Hub

```bash
docker push dsrithad/fusion1_decoupled:multicompartment-segmentation-notebook
```

---

## GPU Support

The standalone image automatically detects GPU availability:
- **With GPU**: requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/), add `--gpus all` to `docker run`
- **Without GPU**: falls back to CPU automatically (slower)

```bash
# With GPU
docker run --rm --gpus all \
  -v /path/to/slide.tif:/input/slide.tif \
  -v /path/to/model.pth:/model/model.pth \
  -v /path/to/output:/output \
  dsrithad/fusion1_decoupled:multicompartment-segmentation-notebook \
  python /opt/MultiC/test_decoupled.py \
    --input_file /input/slide.tif \
    --modelfile  /model/model.pth \
    --output_dir /output/
```

---

## See Also

*Correlating Deep Learning-Based Automated Reference Kidney Histomorphometry with Patient Demographics and Creatinine* by Lucarelli N. *et al.*

---

## Contact
Sayat Mimar (SWE): sayat.mimar@ufl.edu
Pinaki Sarder (PI): pinaki.sarder@ufl.edu
