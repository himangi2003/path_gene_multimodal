# config.py
from __future__ import annotations

from pathlib import Path

# ----------------------------
# Classes
# ----------------------------
classes = [
    "Invasive tumor epithelium (TNBC)",
    "In situ carcinoma (DCIS / LCIS)",
    "Tumor-associated stroma",
    "Lymphocyte-rich stroma / TILs",
    "Lymphoid aggregate / TLS",
    "Necrosis / other non-viable tissue",
]

TUMOR_CLASSES = [
    "Invasive tumor epithelium (TNBC)",
    "In situ carcinoma (DCIS / LCIS)",
    "Tumor-associated stroma",
]

# ----------------------------
# Paths
# ----------------------------
DATA_PATH = Path("TCGA_TNBC/histology")   # input folder containing WSIs
OUTROOT = Path("Himangi/tnbc")            # output root folder

# Slide file extensions you want to process
WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}

# If you want a list of slides at import time (optional)
# NOTE: This lists only files with WSI_EXTS.
image_files = sorted(
    [p.name for p in DATA_PATH.iterdir() if p.is_file() and p.suffix.lower() in WSI_EXTS]
)

# ----------------------------
# Pipeline settings (optional defaults)
# ----------------------------
PATCH_SIZE = 224
MODEL_TYPE = "CLIP"
USE_GPU = True
BATCH_SIZE = 128

THUMB_SIZE = (2000, 2000)

# Polygon parameters
SMOOTH_RADIUS_TILES = 1.0
BLUR_SIGMA = None
AREA_MIN_TILES = 3
SIMPLIFY_FRAC = 0.2
MIN_POLYGON_AREA_PX = 3 * 253 * 253

# Done flags
DONE_FLAG_NAME = "_DONE.json"
DONE_FLAG_MOLECULAR = "_DONE_MOLECULAR.json"
