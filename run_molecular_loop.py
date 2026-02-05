#!/usr/bin/env python3
"""
Loop runner for molecular_feature_extraction.py

- Iterates through WSIs in a folder
- Uses the per-slide tiles CSV: <out_base>/<slide_name>/<slide_name>_annotations_with_coords.csv
- Skips slides with missing CSV
- Skips slides already completed (via _DONE flag, and/or existing outputs)
- Catches exceptions and continues
- Writes success + error logs
- Headless-safe (no GUI backend)
"""

import sys
import traceback
from pathlib import Path

# --- Headless-safe matplotlib (IMPORTANT on cluster nodes) ---
import matplotlib
matplotlib.use("Agg")

# --- If your module isn't on PYTHONPATH, point to it here ---
sys.path.append("/cluster/home/srivash/venvs/Mussel/path_gene_multimodal")

from molecular_feature_extraction import (
    MolecularExtractionConfig,
    extract_molecular_features,
)

# -----------------------
# USER SETTINGS
# -----------------------
DATA_PATH = Path("/lab/deasylab3/Jung/Data/Shared_Data/TCGA_TNBC/histology/")  # WSIs
OUT_BASE  = Path("/lab/deasylab3/Himangi/tnbc/")                               # outputs

# Filter WSI types (adjust to your lab’s reality)
ALLOWED_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}

# Cluster/headless: keep this False
SHOW_PLOT = False

# Skip slide if already has outputs (resume-friendly)
SKIP_IF_DONE = True

# Also consider these as "done" markers (optional but handy)
USE_DONE_FLAG = True          # write/read <outdir>/_DONE
FALLBACK_DONE_MARKERS = True  # treat existing outputs as done too

# Log files (written under OUT_BASE)
OUT_BASE.mkdir(parents=True, exist_ok=True)
SUCCESS_LOG = OUT_BASE / "success_slides.txt"
ERROR_LOG   = OUT_BASE / "error_slides.txt"

# -----------------------
# CONFIG (same as yours)
# -----------------------
CFG = MolecularExtractionConfig(
    only_tme=True,
    tme_mask_col="in_tme_roi",
    device="cuda",
    batch_size=64,
    num_loader_workers=4,
    save_overlays=True,
    save_prob_maps_npz=False,
)

def is_wsi(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXTS

def is_done(outdir: Path, slide_name: str) -> bool:
    """
    Decide whether a slide is already completed.
    Priority:
      1) _DONE flag (most reliable)
      2) fallback: molecular_features.csv exists
      3) fallback: at least one overlay exists (e.g. msi overlay)
    """
    done_flag = outdir / "_DONE"
    if USE_DONE_FLAG and done_flag.exists():
        return True

    if not FALLBACK_DONE_MARKERS:
        return False

    # Primary output from extract_molecular_features
    done_csv = outdir / f"{slide_name}_molecular_features.csv"
    if done_csv.exists():
        return True

    # Overlay is saved per task; pick one that should exist if overlays are enabled
    done_overlay = outdir / f"{slide_name}_msi_overlay.png"
    if done_overlay.exists():
        return True

    return False

def write_done_flag(outdir: Path) -> None:
    if USE_DONE_FLAG:
        (outdir / "_DONE").write_text("ok\n")

def main():
    wsis = sorted([p for p in DATA_PATH.iterdir() if is_wsi(p)])
    print(f"Found {len(wsis)} WSIs in {DATA_PATH}")

    with open(SUCCESS_LOG, "a") as slog, open(ERROR_LOG, "a") as elog:
        for i, wsi_path in enumerate(wsis, start=1):
            slide_name = wsi_path.stem
            outdir = OUT_BASE / slide_name
            outdir.mkdir(parents=True, exist_ok=True)

            tiles_csv = outdir / f"{slide_name}_annotations_with_coords.csv"

            # Skip if tiles CSV missing
            if not tiles_csv.exists():
                msg = f"[{i}/{len(wsis)}] SKIP (missing tiles CSV): {tiles_csv}"
                print(msg)
                elog.write(f"{wsi_path}\tMISSING_TILES_CSV\t{tiles_csv}\n")
                elog.flush()
                continue

            # Skip if already completed
            if SKIP_IF_DONE and is_done(outdir, slide_name):
                print(f"[{i}/{len(wsis)}] SKIP DONE: {wsi_path.name}")
                continue

            print(f"[{i}/{len(wsis)}] RUN: {wsi_path.name}")

            try:
                merged_df, prob_maps, overlay_paths = extract_molecular_features(
                    wsi_path=wsi_path,
                    tiles_info_csv=tiles_csv,
                    outdir=outdir,
                    slide_name=slide_name,
                    config=CFG,
                    show_plot=SHOW_PLOT,
                )

                # Mark completion
                write_done_flag(outdir)

                # Log success
                slog.write(f"{wsi_path}\n")
                slog.flush()

                print(f"  OK: completed {wsi_path.name}")

            except Exception as ex:
                print(f"  ERROR on {wsi_path.name}: {ex}")
                elog.write(f"{wsi_path}\tERROR\t{repr(ex)}\n")
                elog.write(traceback.format_exc() + "\n")
                elog.flush()
                # keep going
                continue

    print("Done.")
    print(f"Success log: {SUCCESS_LOG}")
    print(f"Error log:   {ERROR_LOG}")

if __name__ == "__main__":
    main()
