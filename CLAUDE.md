# CLAUDE.md — AI Assistant Guide for `path_gene_multimodal`

## Project Overview

This repository implements a **multimodal computational pathology pipeline** for Whole Slide Image (WSI) spatial analysis, focused on Triple-Negative Breast Cancer (TNBC). The pipeline integrates tissue segmentation, tile-level classification via CLIP-based zero-shot embeddings, nuclei segmentation (HoverNeXt), and molecular feature prediction (IDaRS), producing GeoJSON spatial maps and molecular biomarker heatmaps.

The pipeline runs on HPC clusters via the **LSF job scheduler**, processing one WSI per job.

---

## Repository Layout

```
path_gene_multimodal/
├── tnbc_config.py                          # Central configuration (EDIT THIS FIRST)
├── main.py                                 # Primary pipeline entry point (8-step WSI processor)
├── Mussel_seg.py                           # Older entry point (no lock/race-condition handling)
├── validate_setup.py                       # Pre-run environment/config validation
├── generate_slide_list.py                  # Utility: list WSIs from DATA_PATH
├── wsi_list.txt                            # Shell snippet to generate slide list
│
├── tiling.py                               # Step 1: WSI tessellation via Mussel
├── extract_embedding_from_tiles.py         # Step 2: Tile feature extraction (CLIP/Virchow2)
├── create_embedding.py                     # Step 3: Class text embedding generation
├── find_annotation_from_embedding.py       # Step 4: Tile annotation via cosine similarity
├── load_annotation_with_coordinates.py     # Step 5: Merge annotations + H5 tile coords
├── create_and_overlay_polygon_from_prediction.py  # Steps 6-8: Polygons, GeoJSON, overlays
│
├── hovernet_inference.py                   # HoverNeXt inference setup + CLI wrapper
├── aggregated_hovernet_run.py              # HoverNeXt per-tile run + WSI coordinate mapping
│
├── molecular_feature_extraction.py         # IDaRS molecular prediction (MSI, HM, CIN, etc.)
├── run_molecular_loop.py                   # Batch runner for molecular extraction
│
├── polygon_morphology.py                   # Tissue/tumor boundary analysis from GeoJSON
├── postprocessing.py                       # Annotation loading + tumor summary utilities
├── polygon_and_preview.py                  # Additional polygon utilities
├── hovernet_plotting.py                    # HoverNeXt result visualization
├── tiling_info.py                          # Additional tiling utilities
├── tiatoolbox_utils.py                     # TIAToolbox helper utilities
├── extract_jeojson_file.py                 # GeoJSON extraction utility
├── publicly_annotated_file_tme_match.py    # TME matching utilities
├── untitled.py                             # Scratch/experimental script
│
├── coarse_ROI_determination.ipynb          # Notebook: coarse ROI selection
├── download_lung_cancer_data.ipynb         # Notebook: data download
├── final_mussel.ipynb                      # Notebook: full Mussel pipeline exploration
├── hovernet_tile_inference.ipynb           # Notebook: HoverNeXt tile inference
│
└── README.md                               # Pipeline overview with setup instructions
```

---

## Configuration — `tnbc_config.py`

**This file must be edited before any pipeline run.** It is the single source of truth for all paths and hyperparameters.

```python
# Tissue classes (5 TNBC classes, order matters for polygon priority)
classes = [
    "Invasive tumor epithelium (TNBC) or In situ carcinoma (DCIS / LCIS)",
    "Tumor-associated stroma",
    "Lymphocyte-rich stroma / TILs",
    "Lymphoid aggregate / TLS",
    "Necrosis / other non-viable tissue",
]

# Tumor classes used to define the TME ROI
TME_CLASSES = [
    "Invasive tumor epithelium (TNBC) or In situ carcinoma (DCIS / LCIS)",
    "Tumor-associated stroma",
]

# --- PATHS (must exist or be creatable) ---
DATA_PATH = Path("/lab/deasylab3/Jung/Data/Shared_Data/TCGA_TNBC/histology")
OUTROOT   = Path("/lab/deasylab3/Himangi/tnbc2/")

WSI_EXTS  = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}

# --- Pipeline parameters ---
PATCH_SIZE = 224          # tile size in pixels
MODEL_TYPE = "CLIP"       # "CLIP" or "Virchow2"
USE_GPU    = True
BATCH_SIZE = 128

THUMB_SIZE = (2000, 2000) # thumbnail resolution for overlays

# --- Polygon smoothing/filtering ---
SMOOTH_RADIUS_TILES = 1.0
BLUR_SIGMA          = None
AREA_MIN_TILES      = 3
SIMPLIFY_FRAC       = 0.2
MIN_POLYGON_AREA_PX = 3 * PATCH_SIZE * PATCH_SIZE

# --- Done flags ---
DONE_FLAG_NAME      = "_DONE.json"
DONE_FLAG_MOLECULAR = "_DONE_MOLECULAR.json"
```

> **Note:** `config.py` is in `.gitignore` (old name). The active config file is `tnbc_config.py`. Jupyter notebooks (`.ipynb`) are also gitignored.

---

## Pipeline Architecture

The main pipeline (`main.py`) processes one WSI through **8 sequential steps**:

```
WSI (.svs/.tif/...)
    │
    ▼  Step 1: tiling.py → run_tessellation()
    │  Mussel tessellates tissue region into 224px tiles
    │  Output: <slide>.h5, patches/, mask.png, thumbnail.png
    │
    ▼  Step 2: extract_embedding_from_tiles.py → run_extract_features_for_tessellation()
    │  CLIP (or Virchow2) encodes each tile patch
    │  Output: <slide>_features.h5, <slide>_features.pt
    │
    ▼  Step 3: create_embedding.py → run_create_class_embeddings()
    │  CLIP text encoder generates one embedding per class label
    │  Output: <slide>_classes.pt
    │
    ▼  Step 4: find_annotation_from_embedding.py → run_annotation_for_extracted_features()
    │  Cosine similarity between tile embeddings and class embeddings → argmax class assignment
    │  Output: <slide>_annotations.csv (tile_index, class scores)
    │
    ▼  Step 5: load_annotation_with_coordinates.py → load_annotations_with_coords()
    │  Joins annotation CSV with spatial (x,y) from H5; computes in_tme_roi flag
    │  Output: <slide>_annotations_with_coords.csv
    │
    ▼  Step 6: create_and_overlay_polygon_from_prediction.py → build_polygons_for_all_classes()
    │  Rasterizes tile grid → per-class masks → morphological smoothing →
    │  overlap resolution → contour extraction → shapely polygons
    │
    ▼  Step 7: export_geojson()
    │  Output: <slide>.geojson (FeatureCollection of tissue polygons in WSI pixel coords)
    │
    ▼  Step 8: Thumbnail + Overlays
       load_svs_thumbnail() → scale_geometry_to_thumb() →
       plot_overlays_all_classes() + plot_overlays_per_class()
       Output: <slide>_all_classes_overlay.png, <class>.png (per class)
```

### Additional Pipelines

**Nuclei segmentation** (`aggregated_hovernet_run.py`):
- Loads tile CSV, selects TME tiles (`in_tme_roi == True`)
- Runs HoverNeXt (`pannuke_convnextv2_tiny_3`) on each tile PNG
- Parses `class_inst.json` + `pinst_pp.zip` zarr for instance data
- Maps tile-local coordinates → WSI-global coordinates
- Saves: `<slide>_hovernet_nuclei_wsi.csv` + `.parquet`

**Molecular feature extraction** (`molecular_feature_extraction.py`):
- Uses TIAToolbox `PatchPredictor` with IDaRS ResNet34 models
- Predicts 6 molecular endpoints: MSI, HM, CIN, CIMP, BRAF, TP53
- Generates thumbnail-space probability heatmaps and overlay PNGs
- Saves: `<slide>_molecular_features.csv`, `<slide>_<task>_overlay.png`

---

## Output Directory Structure

All outputs are written flat into `OUTROOT/` (not per-slide subdirs) by `main.py`, but per-slide subdirs are used by `Mussel_seg.py` and the nuclei/molecular pipelines:

```
OUTROOT/
├── <slide_name>/                           # per-slide output directory
│   ├── <slide_name>.h5                     # tessellation (tile coords + metadata)
│   ├── <slide_name>_features.h5            # tile embeddings (HDF5)
│   ├── <slide_name>_features.pt            # tile embeddings (PyTorch tensor)
│   ├── <slide_name>_classes.pt             # class text embeddings
│   ├── <slide_name>_annotations.csv        # raw tile classifications
│   ├── <slide_name>_annotations_with_coords.csv  # annotations + (x,y) + in_tme_roi
│   ├── <slide_name>.geojson                # vector tissue polygons (level-0 coords)
│   ├── <slide_name>_all_classes_overlay.png
│   ├── <class_name>.png                    # per-class overlays
│   ├── patches/                            # tile PNG images (224x224)
│   ├── mask.png, grid_mask.png, thumbnail.png
│   ├── hovernet_tiles/                     # per-tile HoverNeXt outputs
│   │   └── <tile_stem>/
│   │       ├── class_inst.json
│   │       └── pinst_pp.zip
│   ├── <slide_name>_hovernet_nuclei_wsi.csv
│   ├── <slide_name>_hovernet_nuclei_wsi.parquet
│   ├── <slide_name>_molecular_features.csv
│   ├── <slide_name>_<task>_overlay.png     # msi, hm, cin, cimp, braf, tp53
│   └── <slide_name>._DONE.json             # completion marker
│
├── success_slides.txt                      # molecular loop success log
└── error_slides.txt                        # molecular loop error log
```

---

## Execution Model

### Entry Point
```bash
# WSI_PATH env var is set by LSF array job scheduler
export WSI_PATH=/path/to/slide.svs
python main.py
```

### Lock/Resume Mechanism (`main.py`)
- **Lock file**: `.processing.<slide_name>.lock` prevents concurrent processing of the same slide
- **Done flag**: `<slide_name>._DONE.json` contains metadata; presence skips re-processing
- **Error file**: `<slide_name>_ERROR.txt` written on failure with full traceback
- Stale locks (>48 hours old) are automatically removed

### Pre-run Validation
```bash
python validate_setup.py   # checks paths, imports, GPU, config values
```

### Slide List Generation
```bash
python generate_slide_list.py slide_list.txt
```

---

## Key Functions Reference

| Module | Function | Purpose |
|--------|----------|---------|
| `tiling.py` | `run_tessellation(wsi_path, Patch_size, base_output_dir)` | Mussel tissue segmentation + tile extraction |
| `extract_embedding_from_tiles.py` | `run_extract_features_for_tessellation(wsi_path, base_output_dir, model_type, ...)` | Tile feature embedding |
| `create_embedding.py` | `run_create_class_embeddings(classes, wsi_path, base_output_dir)` | Class text embedding |
| `find_annotation_from_embedding.py` | `run_annotation_for_extracted_features(wsi_path, class_embedding_pt_path, classes, ...)` | Tile-to-class assignment |
| `load_annotation_with_coordinates.py` | `load_annotations_with_coords(wsi_path, classes, tumor_classes, ...)` | Spatial join + TME ROI |
| `create_and_overlay_polygon_from_prediction.py` | `build_polygons_for_all_classes(df, classes, ...)` | Tile mask → polygons |
| `create_and_overlay_polygon_from_prediction.py` | `export_geojson(features, wsi_path, base_output_dir)` | GeoJSON export |
| `aggregated_hovernet_run.py` | `run_hovernet_pipeline_on_wsi_tiles(wsi_path, tiles_csv, ...)` | Nuclei segmentation pipeline |
| `molecular_feature_extraction.py` | `extract_molecular_features(wsi_path, tiles_info_csv, outdir, ...)` | Molecular biomarker prediction |
| `polygon_morphology.py` | `process_one_slide_make_csv_and_plot(wsi_path, ...)` | Tissue/tumor boundary extraction |

---

## External Dependencies

| Library | Usage |
|---------|-------|
| `mussel` (pathology-data-mining/Mussel) | Tessellation, feature extraction, tile annotation |
| `tiffslide` | WSI reading (SVS, TIF, NDPI, MRXS) |
| `tiatoolbox` | IDaRS molecular prediction models, WSI reader |
| `zarr` | Read HoverNeXt instance segmentation maps (`pinst_pp.zip`) |
| `h5py` | Read tessellation H5 files for tile coordinates |
| `shapely`, `geopandas` | Spatial geometry operations |
| `skimage` | Morphological operations, contour finding, regionprops |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib` | Overlay visualization |
| `omegaconf` | Mussel config dataclass serialization |
| `torch` | Deep learning backend (CLIP/Virchow2/HoverNeXt) |
| `cv2` | Tile size inference from PNG |

### HoverNeXt Setup
HoverNeXt is a separate repository expected at `hover_next_inference/` (relative path). The script does `os.chdir("hover_next_inference")` and imports from `src/inference/`. This must be cloned/installed separately.

---

## Data Conventions

### Coordinate System
- All tile coordinates `(x, y)` are **top-left corner** in **level-0 (full-resolution) WSI pixel space**
- `wsi_centroid_x/y`, `wsi_bbox_*`, `wsi_polygon` follow the same level-0 convention
- Thumbnail overlays use scaled coordinates: `scale_x = thumb_w / level0_w`

### HDF5 Tile Coordinate Layout
The code supports multiple H5 schema variants (tries each in order):
1. `coords` dataset — Nx2 or Nx3 array
2. `locations` dataset — Nx2
3. `tiles/coords` — Nx2
4. Separate `x` and `y` datasets
5. `tiles/x` and `tiles/y` datasets

### Annotation CSV Columns
After Step 5, the canonical CSV (`_annotations_with_coords.csv`) contains:
- `tile_index` — integer tile ID
- `x`, `y` — top-left tile coordinates (level-0)
- `<class_name>` columns — cosine similarity scores per class
- `predicted_class` — argmax class name
- `in_tme_roi` — boolean; True if tile is within TME region
- `png_path` — absolute path to tile PNG (if patches saved)
- `level` — pyramid level (if available in H5)

### Nuclei DataFrame Columns
After HoverNeXt processing (`aggregated_hovernet_run.py`):
- `nuc_id` — UUID hex identifier
- `inst_id`, `type`, `type_name` — nucleus instance + classification (1=neoplastic, 2=inflammatory, 3=connective, 4=dead, 5=epithelial)
- `centroid`, `bounding_box`, `polygon` — tile-local coordinates
- `wsi_centroid_x/y`, `wsi_bbox_*`, `wsi_polygon` — WSI-space coordinates
- `tile_x`, `tile_y` — tile top-left offset used for the shift

---

## Development Conventions

### Python Style
- Python 3.10+ syntax used throughout (union types `str | Path`, `list[str]`)
- `pathlib.Path` used consistently; avoid raw string paths
- Function signatures use keyword arguments for optional config parameters
- Error handling via specific exceptions (`FileNotFoundError`, `RuntimeError`, `KeyError`, `ValueError`)

### Adding New Pipeline Steps
1. Create a new module with a single top-level function (e.g., `run_my_step(wsi_path, base_output_dir, ...)`)
2. Add the step to `main.py`'s `run_one_wsi()` with numbered print output (`[N/M]`)
3. Update `tnbc_config.py` with any new parameters
4. Update `validate_setup.py` to include the new module in the import check

### Configuration Pattern
- All hardcoded paths belong in `tnbc_config.py`, never inline in module files
- Module functions accept paths as arguments; they do not import config directly (except `main.py` and `Mussel_seg.py`)
- `run_molecular_loop.py` and `hovernet_inference.py` contain embedded config at the top — treat these as script-style runners

### File Naming
- Per-slide outputs always use `slide_name = wsi_path.stem` as prefix
- Done flags: `<slide_name>._DONE.json` (main pipeline), `_DONE` (molecular loop)
- Error files: `<slide_name>_ERROR.txt`
- Lock files: `.processing.<slide_name>.lock` (hidden file)

---

## Common Issues and Notes

### `postprocessing.py` vs `load_annotation_with_coordinates.py`
Two files implement similar functionality. `load_annotation_with_coordinates.py` is the **canonical version** used by `main.py` — it adds `predicted_class` and `in_tme_roi`. `postprocessing.py` is an older version with additional standalone tumor summary functions; it contains module-level code that executes on import (lines 151–159), so avoid importing it directly.

### `Mussel_seg.py` vs `main.py`
`Mussel_seg.py` is an earlier version of the entry point. It lacks per-slide lock files, robust error logging to files, and the `json_safe()` serializer. Use `main.py` for production runs.

### `run_molecular_loop.py` Path
Line 23 hardcodes a cluster path: `sys.path.append("/cluster/home/srivash/venvs/Mussel/path_gene_multimodal")`. Update this to the actual project root before running on a new system.

### HoverNeXt Working Directory
`hovernet_inference.py` uses `os.chdir("hover_next_inference")`, which is a relative path. Run this script from the project root directory, and ensure `hover_next_inference/` exists there.

### GPU Requirements
- Feature extraction (Step 2) and class embedding (Step 3): GPU strongly recommended
- HoverNeXt inference: GPU required for practical speed
- Molecular prediction (IDaRS): GPU recommended (`device="cuda"`)
- Set `USE_GPU = False` in `tnbc_config.py` for CPU-only testing

### Mussel Environment
The Mussel library requires a separate Python environment (`uv sync --extra torch-gpu`). See README.md for setup instructions. The `mussel.cli.*` modules must be on `sys.path`.

---

## Running the Pipeline

### Single Slide (manual)
```bash
export WSI_PATH=/path/to/TCGA-XX-XXXX.svs
python main.py
```

### Validation
```bash
python validate_setup.py
```

### Molecular Feature Extraction (batch)
```bash
# Edit DATA_PATH and OUT_BASE in run_molecular_loop.py first
python run_molecular_loop.py
```

### HoverNeXt Nuclei Segmentation
```python
from aggregated_hovernet_run import run_hovernet_pipeline_on_wsi_tiles

nuc_df = run_hovernet_pipeline_on_wsi_tiles(
    wsi_path="/path/to/slide.svs",
    tiles_csv="/path/to/outputs/<slide>_annotations_with_coords.csv",
    base_output_dir="/path/to/outputs/",
    only_tme_tiles=True,
    cp="pannuke_convnextv2_tiny_3",
)
```
