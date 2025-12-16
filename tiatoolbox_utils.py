from pathlib import Path
from typing import List

import pandas as pd

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



def select_tiles_for_tme(
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