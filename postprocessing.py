import pandas as pd

import pandas as pd
from pathlib import Path
import pandas as pd
import h5py

def load_annotations_with_coords(
    wsi_path: str,
    base_output_dir: str = "outputs",
    annotations_csv: str | None = None,
    tiles_h5_path: str | None = None,
    patches_dir: str | None = None,
) -> pd.DataFrame:
    """
    Load annotation CSV and enrich with tile coordinates (and optional patch filenames).

    Assumes (by default):
      outputs/<slide>/<slide>_annotations.csv
      outputs/<slide>/<slide>.h5
      outputs/<slide>/patches/   (if you enabled saving PNG patches)
    """
    slide = Path(wsi_path)
    name = slide.stem
    outdir = Path(base_output_dir) / name

    # defaults
    if annotations_csv is None:
        annotations_csv = outdir / f"{name}_annotations.csv"
    if tiles_h5_path is None:
        tiles_h5_path = outdir / f"{name}.h5"
    if patches_dir is None:
        pdir = outdir / "patches"
        patches_dir = str(pdir) if pdir.exists() else None

    annotations_csv = Path(annotations_csv)
    tiles_h5_path = Path(tiles_h5_path)

    if not annotations_csv.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {annotations_csv}")
    if not tiles_h5_path.exists():
        raise FileNotFoundError(f"Tessellation H5 not found: {tiles_h5_path}")

    df = pd.read_csv(annotations_csv)

    # We need a tile index to join on. Many annotate scripts emit it; if not,
    # assume row order is the tile index.
    if "tile_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "tile_index"})

    # Read coords from H5. Common field names differ a bit across repos; try several.
    with h5py.File(tiles_h5_path, "r") as f:
        # try common datasets
        candidates = [
            ("coords", None),                   # Nx2 or Nx3
            ("locations", None),
            ("tiles/coords", None),
            ("x", "y"),                         # separate arrays
            ("tiles/x", "tiles/y"),
        ]

        coords = None
        x = y = level = None

        for cand in candidates:
            if cand[1] is None:
                ds = cand[0]
                if ds in f:
                    arr = f[ds][:]
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        x = arr[:, 0]
                        y = arr[:, 1]
                        if arr.shape[1] >= 3:
                            level = arr[:, 2]
                        break
            else:
                dsx, dsy = cand
                if dsx in f and dsy in f:
                    x = f[dsx][:]; y = f[dsy][:]
                    # optional level
                    lvl_key = "level" if "level" in f else ("tiles/level" if "tiles/level" in f else None)
                    level = f[lvl_key][:] if lvl_key else None
                    break

        if x is None or y is None:
            # final fallback: sometimes stored under 'indices' or 'patches/...'
            for key in f.keys():
                if key.lower().endswith("coords") and f[key].ndim == 2 and f[key].shape[1] >= 2:
                    arr = f[key][:]
                    x = arr[:, 0]; y = arr[:, 1]
                    level = arr[:, 2] if arr.shape[1] >= 3 else None
                    break

        if x is None or y is None:
            raise RuntimeError("Could not find coordinate datasets in the H5 file.")

        meta = {"tile_index": range(len(x)), "x": x, "y": y}
        if level is not None:
            meta["level"] = level
        df_coords = pd.DataFrame(meta)

    # Join annotations with coords on tile_index
    df_merged = df.merge(df_coords, on="tile_index", how="left")

    # If you saved patches as PNGs, populate a path column (common naming: idx.png)
    if patches_dir:
        png_paths = [str(Path(patches_dir) / f"{i}.png") for i in df_merged["tile_index"].values]
        df_merged.insert(df_merged.shape[1], "png_path", png_paths)

    return df_merged

    
def summarize_tumor_area(
    df: pd.DataFrame,
    patch_size: int = 224,   # or the size you used in tessellation
) -> dict:
    """
    Summarize tumor tiles: coordinates, count, and total area.

    Args:
        df: DataFrame with columns ['tile_index', 'x', 'y', 'predicted_class'].
        patch_size: Size of each tile in pixels.

    Returns:
        dict with:
          - tumor_tiles: DataFrame of tumor-only rows
          - count: number of tumor tiles
          - total_area_px2: total tumor area in pixels²
          - bbox: bounding box covering all tumor tiles (x_min, y_min, x_max, y_max)
    """
    tumor_tiles = df[df["predicted_class"] == "tumor"].copy()

    if tumor_tiles.empty:
        return {"tumor_tiles": tumor_tiles, "count": 0, "total_area_px2": 0, "bbox": None}

    count = len(tumor_tiles)
    total_area_px2 = count * (patch_size ** 2)

    x_min = tumor_tiles["x"].min()
    y_min = tumor_tiles["y"].min()
    x_max = tumor_tiles["x"].max() + patch_size
    y_max = tumor_tiles["y"].max() + patch_size
    bbox = (x_min, y_min, x_max, y_max)

    return {
        "tumor_tiles": tumor_tiles,
        "count": count,
        "total_area_px2": total_area_px2,
        "bbox": bbox,
    }
# Suppose you already added predicted_class to your DataFrame
tumor_summary = summarize_tumor_area(df, patch_size=224)

print("Tumor tile count:", tumor_summary["count"])
print("Tumor area (pixels²):", tumor_summary["total_area_px2"])
print("Tumor bounding box:", tumor_summary["bbox"])

# Inspect coordinates of all tumor tiles
print(tumor_summary["tumor_tiles"][["tile_index", "x", "y"]].head())
def tumor_bounding_boxes(df: pd.DataFrame, patch_size: int = 224):
    """
    Get bounding boxes for tiles predicted as 'tumor'.

    Args:
        df: DataFrame with at least ['x', 'y', 'predicted_class'].
        patch_size: Side length of each tile in pixels.

    Returns:
        dict with:
          - all_tumor_bbox: (x_min, y_min, x_max, y_max) across all tumor tiles
          - tile_bboxes: list of (x_min, y_min, x_max, y_max) for each tumor tile
    """
    tumor_tiles = df[df["predicted_class"] == "tumor"]

    if tumor_tiles.empty:
        return {"all_tumor_bbox": None, "tile_bboxes": []}

    # Global bounding box (covering all tumor tiles)
    x_min = tumor_tiles["x"].min()
    y_min = tumor_tiles["y"].min()
    x_max = tumor_tiles["x"].max() + patch_size
    y_max = tumor_tiles["y"].max() + patch_size
    all_bbox = (x_min, y_min, x_max, y_max)

    # Individual bounding boxes per tile
    tile_bboxes = [
        (row["x"], row["y"], row["x"] + patch_size, row["y"] + patch_size)
        for _, row in tumor_tiles.iterrows()
    ]

    return {"all_tumor_bbox": all_bbox, "tile_bboxes": tile_bboxes}