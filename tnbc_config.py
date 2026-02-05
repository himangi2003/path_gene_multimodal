# tnbc_config.py

from pathlib import Path

# ----------------------------
# Classes
# ----------------------------
classes = [
    "Invasive tumor epithelium (TNBC) or In situ carcinoma (DCIS / LCIS)",
    "Tumor-associated stroma",
    "Lymphocyte-rich stroma / TILs",
    "Lymphoid aggregate / TLS",
    "Necrosis / other non-viable tissue",
]

TME_CLASSES = [
    "Invasive tumor epithelium (TNBC) or In situ carcinoma (DCIS / LCIS)",
    "Tumor-associated stroma",
]


# ----------------------------
# Paths
# ----------------------------
DATA_PATH = Path("/lab/deasylab3/Jung/Data/Shared_Data/TCGA_TNBC/histology")   # folder containing WSIs (can be nested)
OUTROOT = Path("/lab/deasylab3/Himangi/tnbc2/")            # output root folder (per-slide subfolders)

WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}

# Optional: list WSI paths
if DATA_PATH.exists():
    image_files = sorted([p for p in DATA_PATH.rglob("*") if p.is_file() and p.suffix.lower() in WSI_EXTS])
else:
    image_files = []

# ----------------------------
# Pipeline settings
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
MIN_POLYGON_AREA_PX = 3 * PATCH_SIZE * PATCH_SIZE

# Done flags
DONE_FLAG_NAME = "_DONE.json"
DONE_FLAG_MOLECULAR = "_DONE_MOLECULAR.json"

