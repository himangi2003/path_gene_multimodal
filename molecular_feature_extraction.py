# molecular_feature_extraction.py

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional but commonly available in your environment
import cv2
import matplotlib.pyplot as plt

from tiatoolbox.models.engine.patch_predictor import PatchPredictor
from tiatoolbox.utils.visualization import overlay_probability_map
from tiatoolbox.wsicore.wsireader import WSIReader


DEFAULT_TASKS: Dict[str, str] = {
    "msi":  "resnet34-idars-msi",
    "hm":   "resnet34-idars-hm",
    "cin":  "resnet34-idars-cin",
    "cimp": "resnet34-idars-cimp",
    "braf": "resnet34-idars-braf",
    "tp53": "resnet34-idars-tp53",
}


@dataclass
class MolecularExtractionConfig:
    only_tme: bool = True
    tme_mask_col: str = "in_tme_roi"

    device: str = "cuda"  # "cpu" also OK
    batch_size: int = 64
    num_loader_workers: int = 4

    # Thumbnail params (TIAToolbox slide_thumbnail)
    thumbnail_resolution: float = 4.0
    thumbnail_units: str = "power"

    # Overlay params
    overlay_alpha: float = 0.5
    overlay_min_val: float = 0.1
    colour_map: Optional[str] = None  # e.g. "jet" if you want

    # I/O
    save_overlays: bool = True
    save_prob_maps_npz: bool = False  # optionally save prob maps too


def load_tile_annotations(tiles_csv: str | Path) -> pd.DataFrame:
    """Load tile-level CSV with required columns."""
    tiles_csv = Path(tiles_csv)
    if not tiles_csv.exists():
        raise FileNotFoundError(f"Tile annotations CSV not found: {tiles_csv}")

    df = pd.read_csv(tiles_csv)
    required = {"tile_index", "x", "y", "png_path", "predicted_class"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in tiles CSV: {missing}")

    return df


def select_tiles(
    tiles_df: pd.DataFrame,
    only_tme: bool = True,
    tme_mask_col: str = "in_tme_roi",
) -> List[Path]:
    """Choose tiles for inference, optionally restricted to TME ROI."""
    df = tiles_df.copy()
    if only_tme:
        if tme_mask_col not in df.columns:
            raise KeyError(f"Column '{tme_mask_col}' not found in tiles_df.")
        df = df[df[tme_mask_col] == True]
        if df.empty:
            raise ValueError(
                "No tiles marked as TME; filtering produced empty set "
                f"(expected {tme_mask_col} == True)."
            )

    png_paths = sorted({Path(p) for p in df["png_path"].tolist()})
    return png_paths


def _clean_existing_paths(png_paths: List[Path]) -> List[str]:
    clean_tiles: List[str] = []
    for p in png_paths:
        p_str = str(p).strip()
        if p_str and os.path.isfile(p_str):
            clean_tiles.append(p_str)
    return clean_tiles


def infer_tile_size(tile_path: str | Path) -> int:
    """Infer square tile size from the first tile image."""
    img = cv2.imread(str(tile_path))
    if img is None:
        raise ValueError(f"cv2.imread failed for tile: {tile_path}")
    h, w = img.shape[:2]
    if h != w:
        raise ValueError(f"Tiles are not square: {tile_path} has shape {h}x{w}")
    return w


def run_idars_predictions(
    tile_paths: List[str],
    tasks: Dict[str, str],
    device: str = "cuda",
    batch_size: int = 64,
    num_loader_workers: int = 4,
) -> pd.DataFrame:
    """Run TIAToolbox PatchPredictor models on tiles, returning probs per task."""
    pred_df = pd.DataFrame({"png_path": tile_paths})

    for task_name, model_name in tasks.items():
        print(f"Running {task_name} ({model_name}) on {len(tile_paths)} tiles...")

        predictor = PatchPredictor(
            pretrained_model=model_name,
            batch_size=batch_size,
            num_loader_workers=num_loader_workers,
        )

        out = predictor.predict(
            imgs=tile_paths,
            mode="patch",
            return_probabilities=True,
            device=device,
        )

        probs = np.array(out["probabilities"])[:, 1]  # positive class probability
        pred_df[f"{task_name}_prob"] = probs

    return pred_df


def get_wsi_overview_and_dims(
    wsi_path: str | Path,
    resolution: float = 4.0,
    units: str = "power",
) -> Tuple[np.ndarray, int, int, int, int]:
    """Return (overview_img, wsi_w, wsi_h, thumb_w, thumb_h)."""
    wsi = WSIReader.open(str(wsi_path))
    wsi_w, wsi_h = wsi.info.level_dimensions[0]

    overview = wsi.slide_thumbnail(resolution=resolution, units=units)
    thumb_h, thumb_w = overview.shape[:2]
    return overview, wsi_w, wsi_h, thumb_w, thumb_h


def make_prob_map_for_task(
    df: pd.DataFrame,
    prob_col: str,
    wsi_w: int,
    wsi_h: int,
    thumb_w: int,
    thumb_h: int,
    tile_size: int,
) -> np.ndarray:
    """Aggregate tile-level probabilities into a thumbnail-sized heatmap."""
    prob_map = np.zeros((thumb_h, thumb_w), dtype=float)
    count_map = np.zeros((thumb_h, thumb_w), dtype=float)

    for _, row in df.iterrows():
        x0 = int(row["x"])
        y0 = int(row["y"])
        p = float(row[prob_col])

        tx0 = int(x0 / wsi_w * thumb_w)
        ty0 = int(y0 / wsi_h * thumb_h)
        tx1 = int((x0 + tile_size) / wsi_w * thumb_w)
        ty1 = int((y0 + tile_size) / wsi_h * thumb_h)

        tx0 = max(0, min(thumb_w, tx0))
        tx1 = max(0, min(thumb_w, tx1))
        ty0 = max(0, min(thumb_h, ty0))
        ty1 = max(0, min(thumb_h, ty1))

        if tx1 > tx0 and ty1 > ty0:
            prob_map[ty0:ty1, tx0:tx1] += p
            count_map[ty0:ty1, tx0:tx1] += 1.0

    mask = count_map > 0
    prob_map[mask] /= count_map[mask]
    return np.clip(prob_map, 0.0, 1.0)


def make_overlays(
    overview_img: np.ndarray,
    prob_maps: Dict[str, np.ndarray],
    alpha: float = 0.5,
    min_val: float = 0.1,
    colour_map: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    overlays: Dict[str, np.ndarray] = {}
    for task, pmap in prob_maps.items():
        overlays[task] = overlay_probability_map(
            img=overview_img,
            prediction=pmap,
            alpha=alpha,
            min_val=min_val,
            return_ax=False,
            colour_map=colour_map,
        )
    return overlays


def plot_overlays(overlays: Dict[str, np.ndarray], title: str = "Probability Map Overlays") -> None:
    tasks = list(overlays.keys())
    n = len(tasks)
    cols = 3 if n >= 3 else n
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, task in enumerate(tasks, start=1):
        ax = plt.subplot(rows, cols, i)
        ax.imshow(overlays[task])
        ax.set_title(task.upper())
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_overlays(
    overlays: Dict[str, np.ndarray],
    outdir: str | Path,
    slide_name: str,
) -> Dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    for task, img in overlays.items():
        p = outdir / f"{slide_name}_{task}_overlay.png"
        plt.imsave(str(p), img)
        paths[task] = p
    return paths


def extract_molecular_features(
    *,
    wsi_path: str | Path,
    tiles_info_csv: str | Path,
    outdir: str | Path,
    slide_name: str,
    tasks: Dict[str, str] = DEFAULT_TASKS,
    config: MolecularExtractionConfig = MolecularExtractionConfig(),
    tile_size: Optional[int] = None,
    show_plot: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Path]]:
    """
    End-to-end extraction:
      1) load tile annotations
      2) filter tiles (optionally TME)
      3) run IDaRS tasks on tiles
      4) merge preds into tile df and save CSV
      5) build thumbnail prob maps + overlays
      6) optionally plot and/or save overlay images

    Returns:
      merged_df, prob_maps, overlay_paths
    """
    # Keep logging from duplicating handlers in notebook contexts
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tiles_df = load_tile_annotations(tiles_info_csv)
    png_paths = select_tiles(tiles_df, config.only_tme, config.tme_mask_col)
    clean_tiles = _clean_existing_paths(png_paths)

    print(f"Original tiles: {len(png_paths)} | Clean & existing: {len(clean_tiles)}")
    if not clean_tiles:
        raise ValueError("No valid tile PNG files found after filtering and file existence checks.")

    pred_df = run_idars_predictions(
        tile_paths=clean_tiles,
        tasks=tasks,
        device=config.device,
        batch_size=config.batch_size,
        num_loader_workers=config.num_loader_workers,
    )

    merged = tiles_df.merge(pred_df, on="png_path", how="inner")
    molecular_features_path = outdir / f"{slide_name}_molecular_features.csv"
    merged.to_csv(molecular_features_path, index=False)
    print("Saved predictions to:", molecular_features_path)

    # WSI thumbnail + dims
    overview, wsi_w, wsi_h, thumb_w, thumb_h = get_wsi_overview_and_dims(
        wsi_path=wsi_path,
        resolution=config.thumbnail_resolution,
        units=config.thumbnail_units,
    )

    # Tile size inference if not provided
    if tile_size is None:
        tile_size = infer_tile_size(merged["png_path"].iloc[0])
    print("Using TILE_SIZE:", tile_size)

    # Prob maps
    prob_maps: Dict[str, np.ndarray] = {}
    for task in tasks.keys():
        prob_col = f"{task}_prob"
        if prob_col not in merged.columns:
            raise KeyError(f"Expected probability column missing: {prob_col}")
        prob_maps[task] = make_prob_map_for_task(
            df=merged,
            prob_col=prob_col,
            wsi_w=wsi_w,
            wsi_h=wsi_h,
            thumb_w=thumb_w,
            thumb_h=thumb_h,
            tile_size=tile_size,
        )

    # Optional save prob maps
    if config.save_prob_maps_npz:
        npz_path = outdir / f"{slide_name}_prob_maps.npz"
        np.savez_compressed(npz_path, **prob_maps)
        print("Saved prob maps to:", npz_path)

    # Overlays
    overlays = make_overlays(
        overview_img=overview,
        prob_maps=prob_maps,
        alpha=config.overlay_alpha,
        min_val=config.overlay_min_val,
        colour_map=config.colour_map,
    )

    overlay_paths: Dict[str, Path] = {}
    if config.save_overlays:
        overlay_paths = save_overlays(overlays, outdir, slide_name)
        print("Saved overlay images to:", outdir)

    if show_plot:
        plot_overlays(overlays, title=f"{slide_name}: Molecular Probability Overlays")

    return merged, prob_maps, overlay_paths
