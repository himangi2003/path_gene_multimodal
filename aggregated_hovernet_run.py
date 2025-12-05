import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import zarr
from skimage.measure import regionprops, find_contours, approximate_polygon
from hovernet_inference import*
# -------------------------------------------------
# Tile size inference
# -------------------------------------------------
def infer_tile_size(coords: np.ndarray) -> int:
    """Infer a constant tile size from a 1D array of tile top-left coordinates by
    taking the mode of forward differences (>0). If empty or single value, default to 256."""
    if coords.size < 2:
        return 256
    diffs = np.diff(np.sort(np.unique(coords)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 256
    vals, counts = np.unique(diffs, return_counts=True)
    return int(vals[np.argmax(counts)])


# -------------------------------------------------
# Load tile annotations CSV
# -------------------------------------------------
def load_tile_annotations(
    tiles_csv: str | Path,
) -> pd.DataFrame:
    """
    Load tile-level CSV with columns like:
      tile_index, x, y, png_path, predicted_class, in_tme_roi, ...
    """
    tiles_csv = Path(tiles_csv)
    if not tiles_csv.exists():
        raise FileNotFoundError(f"Tile annotations CSV not found: {tiles_csv}")
    df = pd.read_csv(tiles_csv)
    required = {"tile_index", "x", "y", "png_path", "predicted_class"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in tiles CSV: {missing}")
    return df


# -------------------------------------------------
# Choose tiles to run HoverNet on
# -------------------------------------------------
def select_tiles_for_hovernet(
    tiles_df: pd.DataFrame,
    only_tme: bool = True,
    tme_mask_col: str = "in_tme_roi",
) -> List[Path]:
    """
    Decide which tiles to send to HoverNet:
    - if only_tme=True: restrict to tiles where `in_tme_roi == True`
    - otherwise: use all tiles.
    Returns a list of png_paths as Path objects.
    """
    df = tiles_df.copy()
    if only_tme:
        if tme_mask_col not in df.columns:
            raise KeyError(f"Column '{tme_mask_col}' not found in tiles_df.")
        df = df[df[tme_mask_col] == True]
        if df.empty:
            raise ValueError("No tiles marked as TME; `in_tme_roi == True` produced empty set.")
    png_paths = sorted({Path(p) for p in df["png_path"].tolist()})
    return png_paths


# -------------------------------------------------
# HoverNet type names
# -------------------------------------------------
TYPE_NAMES = {
    1: "neoplastic",
    2: "inflammatory",
    3: "connective",
    4: "dead",
    5: "epithelial",
}


# -------------------------------------------------
# Run HoverNet on ONE tile
# -------------------------------------------------
def run_hovernet_on_tile(
    png_path: Path,
    tile_outdir: Path,
    cp: str = "pannuke_convnextv2_tiny_3",
) -> pd.DataFrame:
    """
    Run HoverNet on a single PNG tile and return a dataframe of nuclei
    in tile-local coordinates. Returns empty DataFrame on failure.

    tile_outdir is the directory where HoverNet outputs for THIS tile will live.
    """
    # Clean and recreate tile_outdir
    tile_outdir = Path(tile_outdir)
    if tile_outdir.exists() and tile_outdir.is_dir():
        shutil.rmtree(tile_outdir)
    tile_outdir.mkdir(parents=True, exist_ok=True)

    params = {
        "input": str(png_path),
        "output_dir": str(tile_outdir),
        "cp": cp,
        "metric": "f1",
        "batch_size": 32,
        "tta": 4,
        "save_polygon": True,
        "tile_size": 256,
        "overlap": 0.96875,
        "inf_workers": 4,
        "inf_writers": 2,
        "pp_tiling": 8,
        "pp_overlap": 256,
        "pp_workers": 16,
        "keep_raw": False,
        "cache": "/tmp",
        "only_inference": False,
    }

    print(f"\n=== Running HoverNet on {png_path.name} ===")
    infer(params)

    class_inst_path = tile_outdir / "class_inst.json"
    pinst_path = tile_outdir / "pinst_pp.zip"

    if not class_inst_path.is_file() or not pinst_path.is_file():
        print(f"  WARNING: Missing HoverNet outputs for {png_path.name}, skipping.")
        return pd.DataFrame()

    # ---------- 1. Parse class_inst.json ----------
    with open(class_inst_path, "r") as f:
        class_info = json.load(f)

    rows = []
    for key, val in class_info.items():
        inst_id = int(key)        # instance label in pinst_pp
        type_id = int(val[0])     # 1..5 depending on model

        # val[1] = [0, cx, cy]
        _, cx, cy = val[1]
        centroid = [float(cx), float(cy)]

        rows.append(
            {
                "inst_id": inst_id,
                "type": type_id,
                "centroid": centroid,
            }
        )

    if not rows:
        print(f"  WARNING: No instances in class_inst.json for {png_path.name}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ---------- 2. Load instance map ----------
    z = zarr.open(str(pinst_path), mode="r")
    inst_map = np.asarray(z)
    if inst_map.ndim == 3:
        inst_map = inst_map[0]

    H, W = inst_map.shape
    print(f"  inst_map shape for {png_path.name}: {inst_map.shape}")

    # ---------- 3. Bboxes & polygons ----------
    props = regionprops(inst_map)

    bbox_dict = {}
    poly_dict = {}

    for r in props:
        inst_id = r.label

        min_row, min_col, max_row, max_col = r.bbox
        bbox_dict[inst_id] = [int(min_col), int(min_row),
                              int(max_col), int(max_row)]  # [x_min, y_min, x_max, y_max]

        mask = (inst_map == inst_id)
        contours = find_contours(mask.astype(float), level=0.5)
        if not contours:
            continue

        contour = max(contours, key=lambda c: c.shape[0])

        ys = contour[:, 0]
        xs = contour[:, 1]
        poly_coords = np.stack([xs, ys], axis=1)

        poly_simplified = approximate_polygon(poly_coords, tolerance=0.5)
        poly = poly_simplified.tolist()

        poly_dict[inst_id] = poly

    df["bounding_box"] = df["inst_id"].map(bbox_dict.get)
    df["polygon"] = df["inst_id"].map(poly_dict.get)

    # ---------- 4. Type name + nuc_id ----------
    df["type_name"] = df["type"].map(TYPE_NAMES)
    df["nuc_id"] = df["inst_id"].apply(lambda _: uuid.uuid4().hex)

    # ---------- 5. Tile metadata ----------
    df["tile_name"] = png_path.stem
    df["tile_path"] = str(png_path)

    final_df = df[
        [
            "nuc_id",
            "inst_id",
            "type",
            "type_name",
            "bounding_box",
            "centroid",
            "polygon",
            "tile_name",
            "tile_path",
        ]
    ]

    return final_df


# -------------------------------------------------
# Run HoverNet on MANY tiles & aggregate
# -------------------------------------------------
def run_hovernet_on_tiles(
    png_paths: List[Path],
    out_root: Path,
    cp: str = "pannuke_convnextv2_tiny_3",
) -> pd.DataFrame:
    """
    Run HoverNet on a list of PNG tiles and aggregate all nuclei into one dataframe.
    Each tile gets its own output subdirectory under out_root.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    all_dfs: List[pd.DataFrame] = []

    print(f"Running HoverNet on {len(png_paths)} tiles.")
    for png_path in png_paths:
        tile_outdir = out_root / png_path.stem
        tile_df = run_hovernet_on_tile(png_path, tile_outdir, cp=cp)
        if not tile_df.empty:
            all_dfs.append(tile_df)

    if not all_dfs:
        print("No nuclei found in any tile.")
        return pd.DataFrame()

    wsi_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined nuclei dataframe shape: {wsi_df.shape}")
    return wsi_df


# -------------------------------------------------
# Map nuclei back to WSI coordinates
# -------------------------------------------------
def add_wsi_coords_to_nuclei(
    nuc_df: pd.DataFrame,
    tiles_df: pd.DataFrame,
    tile_key_col_nuc: str = "tile_path",   # in nuclei df
    tile_key_col_tiles: str = "png_path",  # in tiles df
) -> pd.DataFrame:
    """
    For each nucleus, add WSI-space coordinates by shifting tile-local coordinates
    (centroid, bounding_box, polygon) by the tile's top-left (x, y).

    Returns a copy of nuc_df with extra columns:
      tile_x, tile_y,
      centroid_x, centroid_y,
      wsi_centroid_x, wsi_centroid_y,
      bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
      wsi_bbox_xmin, wsi_bbox_ymin, wsi_bbox_xmax, wsi_bbox_ymax,
      wsi_polygon (polygon shifted into slide / WSI space).
    """
    nuc_df = nuc_df.copy()
    tiles_df = tiles_df.copy()

    # Normalize keys via filename stem
    tiles_df["tile_key"] = tiles_df[tile_key_col_tiles].apply(lambda p: Path(p).stem)
    nuc_df["tile_key"] = nuc_df[tile_key_col_nuc].apply(lambda p: Path(p).stem)

    tile_xy = (
        tiles_df[["tile_key", "x", "y"]]
        .drop_duplicates("tile_key")
        .set_index("tile_key")
    )

    nuc_df = nuc_df.join(tile_xy, on="tile_key", how="left", rsuffix="_tile")
    nuc_df.rename(columns={"x": "tile_x", "y": "tile_y"}, inplace=True)

    if nuc_df["tile_x"].isna().any():
        missing = nuc_df[nuc_df["tile_x"].isna()]["tile_key"].unique()
        raise ValueError(f"Some nuclei have tile_key with no matching tile coords: {missing}")

    # Expand centroid
    cent_arr = np.vstack(nuc_df["centroid"].to_numpy())
    nuc_df["centroid_x"] = cent_arr[:, 0]
    nuc_df["centroid_y"] = cent_arr[:, 1]

    nuc_df["wsi_centroid_x"] = nuc_df["tile_x"] + nuc_df["centroid_x"]
    nuc_df["wsi_centroid_y"] = nuc_df["tile_y"] + nuc_df["centroid_y"]

    # Expand bbox
    bbox_arr = np.vstack(nuc_df["bounding_box"].to_numpy())
    nuc_df["bbox_xmin"] = bbox_arr[:, 0]
    nuc_df["bbox_ymin"] = bbox_arr[:, 1]
    nuc_df["bbox_xmax"] = bbox_arr[:, 2]
    nuc_df["bbox_ymax"] = bbox_arr[:, 3]

    nuc_df["wsi_bbox_xmin"] = nuc_df["bbox_xmin"] + nuc_df["tile_x"]
    nuc_df["wsi_bbox_ymin"] = nuc_df["bbox_ymin"] + nuc_df["tile_y"]
    nuc_df["wsi_bbox_xmax"] = nuc_df["bbox_xmax"] + nuc_df["tile_x"]
    nuc_df["wsi_bbox_ymax"] = nuc_df["bbox_ymax"] + nuc_df["tile_y"]

    # Shift polygons
    def shift_poly(poly, dx, dy):
        if poly is None:
            return None
        return [[float(x) + dx, float(y) + dy] for x, y in poly]

    nuc_df["wsi_polygon"] = [
        shift_poly(poly, tx, ty)
        for poly, tx, ty in zip(
            nuc_df["polygon"],
            nuc_df["tile_x"],
            nuc_df["tile_y"],
        )
    ]

    return nuc_df


# -------------------------------------------------
# Full pipeline: beginning → end
# -------------------------------------------------
def run_hovernet_pipeline_on_wsi_tiles(
    wsi_path: str | Path,
    tiles_csv: str | Path,
    base_output_dir: str | Path,
    only_tme_tiles: bool = True,
    cp: str = "pannuke_convnextv2_tiny_3",
) -> pd.DataFrame:
    """
    Full pipeline:
      1) load tile annotations
      2) select tiles (optionally only TME)
      3) run HoverNet on those tiles
      4) map nuclei back to WSI coordinates
      5) save combined nuclei CSV & Parquet in a single folder

    Returns nuclei dataframe with WSI coordinates (wsi_centroid_x, etc.).
    """
    wsi_path = Path(wsi_path)
    base_output_dir = Path(base_output_dir)
    slide_name = wsi_path.stem

    tiles_df = load_tile_annotations(tiles_csv)

    # optional: see inferred tile size
    patch_w = infer_tile_size(tiles_df["x"].values)
    patch_h = infer_tile_size(tiles_df["y"].values)
    print(f"Inferred tile / patch size: {patch_w} x {patch_h}")

    png_paths = select_tiles_for_hovernet(
        tiles_df,
        only_tme=only_tme_tiles,
        tme_mask_col="in_tme_roi",
    )

    out_root = base_output_dir / slide_name / "hovernet_tiles"
    nuc_df_local = run_hovernet_on_tiles(
        png_paths=png_paths,
        out_root=out_root,
        cp=cp,
    )

    if nuc_df_local.empty:
        print("No nuclei detected; returning empty dataframe.")
        return nuc_df_local

    nuc_df_wsi = add_wsi_coords_to_nuclei(
        nuc_df_local,
        tiles_df=tiles_df,
        tile_key_col_nuc="tile_path",
        tile_key_col_tiles="png_path",
    )

    # Save combined outputs
    combined_dir = base_output_dir / slide_name
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = combined_dir / f"{slide_name}_hovernet_nuclei_wsi.csv"
    combined_parquet = combined_dir / f"{slide_name}_hovernet_nuclei_wsi.parquet"

    nuc_df_wsi.to_csv(combined_csv, index=False)
    nuc_df_wsi.to_parquet(combined_parquet, index=False)

    print(f"Saved WSI nuclei CSV:     {combined_csv}")
    print(f"Saved WSI nuclei Parquet: {combined_parquet}")

    return nuc_df_wsi
