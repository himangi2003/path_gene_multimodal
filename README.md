## 🧬 Pipeline Overview

This project implements a complete **Whole Slide Image (WSI) spatial analysis pipeline** consisting of the following major stages:

---

### 1️⃣ Tissue Segmentation & Tiling — *Mussel*
We use **Mussel** for initial WSI tissue segmentation and tile extraction:

🔗 https://github.com/pathology-data-mining/Mussel

**Steps:**
- Load WSI
- Segment foreground tissue
- Generate uniform tiles from tissue regions
- Save:
  - Tile coordinates `(x, y)`
  - Tile images (`.png`)
  - Metadata (`.h5`, `.csv`, `thumbnail.png`)

**Output:**
- Tile grid covering tissue regions of the WSI
- Thumbnail overview of the slide

---

### 2️⃣ Tile-Level Classification (Tumor / TME)
Each extracted tile is classified into:

- Tumor epithelium  
- Tumor-associated stroma (desmoplastic stroma)  
- Vessel endothelium  
- Necrosis  
- Lymphoid aggregate / TLS  

**Outputs:**
- `predicted_class`
- Softmax class probabilities
- `in_tme_roi` flag for selecting tumor microenvironment (TME) tiles

---

### 3️⃣ Nuclei Segmentation — *HoverNet / HoverNeXt*
For all selected tiles (optionally only TME tiles), we run **HoverNet / HoverNeXt** for instance-level nuclei segmentation.

**For each tile:**
- Nucleus instance ID
- Nucleus type
- Bounding box
- Polygon contour
- Centroid (tile-local coordinates)

**Outputs (per tile):**
- `class_inst.json`
- `pinst_pp.zip`
- Instance-level nucleus features

---

### 4️⃣ Mapping Nuclei Back to WSI Space
Tile-local nucleus coordinates are transformed into **global WSI coordinates** using tile top-left `(x, y)` offsets.

Converted features:
- `wsi_centroid_x`, `wsi_centroid_y`
- `wsi_bbox_xmin`, `wsi_bbox_ymin`, `wsi_bbox_xmax`, `wsi_bbox_ymax`
- `wsi_polygon` (global nucleus contour)

**Final Output:**
- One unified nuclei CSV / Parquet file per WSI in **WSI coordinate space**

---

### 5️⃣ Spatial Graph Construction
From the WSI-level nuclei centroids and types, we construct a **spatial cell graph**:

- Nodes = nuclei
- Edges = spatial proximity (e.g. kNN, radius graph)
- Node attributes:
  - Nucleus type
  - Area, shape, etc.
- Edge attributes:
  - Distance
  - Neighborhood composition

---

### 6️⃣ Spatial Graph Analysis
The constructed graphs are then used for downstream analysis:

- Cell–cell interaction patterns
- Tumor–immune spatial organization
- Graph statistics (degree, clustering, centrality)
- Tissue architecture quantification
- Predictive modeling & biomarkers

---

## ✅ End-to-End Summary


