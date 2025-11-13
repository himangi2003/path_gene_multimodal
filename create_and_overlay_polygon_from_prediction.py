"""
Tile → Contours pipeline for histology.

Given a table like:
    tile_index, x, y, predicted_class
and a fixed (or inferable) tile size, this module will:
 1) Place tiles back into slide space
 2) Build per-class raster masks
 3) Denoise & smooth masks
 4) Resolve overlaps across classes
 5) Extract connected components
 6) Convert components to vector polygons in slide pixel coordinates
 7) Merge/clean/tag polygons
 8) Export to GeoJSON or Parquet

Dependencies: numpy, pandas, shapely, scikit-image, geojson (optional), geopandas (optional)
Install: pip install numpy pandas shapely scikit-image geopandas

NOTE: The rasterization works on a TILE-GRID (each tile becomes 1 pixel). We then map
      grid coords → slide pixel coords using the tile size and origin. This avoids
      memory blow-ups while producing faithful tile-edge polygons. If you need
      pixel-resolution masks, swap the rasterization step for a PIL rectangle fill.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import json
import math
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from shapely import affinity
from skimage import measure, morphology, filters
import json
from collections import defaultdict

import matplotlib.pyplot as plt
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely import affinity
import tiffslide
# ----------------------------
# 0) Data model & helpers
# ----------------------------

@dataclass
class TileGrid:
    label_grid: np.ndarray           # (H, W) int class indices per tile (argmax); -1 for background/empty
    prob_grids: Optional[np.ndarray] # (K, H, W) per-class scores/probs; None if not provided
    x_coords: np.ndarray             # sorted unique tile x (top-left) in slide px
    y_coords: np.ndarray             # sorted unique tile y (top-left) in slide px
    tile_w: int                      # inferred or provided tile width (px)
    tile_h: int                      # inferred or provided tile height (px)
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]


def infer_tile_size(coords: np.ndarray) -> int:
    """Infer a constant tile size from a 1D array of tile top-left coordinates by taking the mode
    of forward differences (>0). If empty or single value, default to 256.
    """
    if coords.size < 2:
        return 256
    diffs = np.diff(np.sort(np.unique(coords)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 256
    # mode of integer diffs
    vals, counts = np.unique(diffs, return_counts=True)
    return int(vals[np.argmax(counts)])


# ----------------------------
# 1) Place tiles back into slide space → tile grid
# ----------------------------

def tiles_to_grid(
    df: pd.DataFrame,
    classes: List[str],
    tile_w: Optional[int] = None,
    tile_h: Optional[int] = None,
    background_label: str = "Background / artifact",
    prob_cols: Optional[Dict[str, str]] = None,  # map class name -> column of soft score
) -> TileGrid:
    """Build a TILE-GRID where each tile becomes one grid cell.

    Args:
        df: must contain columns ['x','y','predicted_class'] and optional per-class prob columns.
        classes: ordered list of class names (defines indices 0..K-1)
        tile_w, tile_h: if None, inferred from x,y coordinate gaps.
        background_label: name that marks background tiles in predicted_class (optional)
        prob_cols: mapping from class name to df column name with score/prob (optional)

    Returns: TileGrid
    """
    assert {'x','y','predicted_class'}.issubset(df.columns)
    K = len(classes)
    class_to_idx = {c:i for i,c in enumerate(classes)}
    idx_to_class = {i:c for c,i in class_to_idx.items()}

    x_vals = np.sort(df['x'].unique())
    y_vals = np.sort(df['y'].unique())

    if tile_w is None:
        tile_w = infer_tile_size(x_vals)
    if tile_h is None:
        tile_h = infer_tile_size(y_vals)

    x_to_ix = {x:i for i,x in enumerate(x_vals)}
    y_to_iy = {y:i for i,y in enumerate(y_vals)}

    H, W = len(y_vals), len(x_vals)
    label_grid = -np.ones((H, W), dtype=np.int16)

    # Optional per-class scores
    prob_grids = None
    if prob_cols is not None:
        prob_grids = np.zeros((K, H, W), dtype=np.float32)

    for row in df.itertuples(index=False):
        x, y, cls = getattr(row, 'x'), getattr(row, 'y'), getattr(row, 'predicted_class')
        iy, ix = y_to_iy[y], x_to_ix[x]
        if cls in class_to_idx:
            ci = class_to_idx[cls]
            label_grid[iy, ix] = ci
            if prob_cols is not None:
                for cname, col in prob_cols.items():
                    ci2 = class_to_idx[cname]
                    prob_grids[ci2, iy, ix] = getattr(row, col)
        else:
            # unknown class → background index if present, otherwise leave -1
            if background_label in class_to_idx:
                label_grid[iy, ix] = class_to_idx[background_label]

    return TileGrid(label_grid, prob_grids, x_vals, y_vals, tile_w, tile_h, class_to_idx, idx_to_class)


# ----------------------------
# 2) Build per-class raster mask (grid space)
# ----------------------------

def build_class_mask(grid: TileGrid, class_name: str, use_probs: bool = False) -> np.ndarray:
    """Return a (H,W) float32 mask in TILE-GRID units for a class.
       If use_probs=True and prob_grids available, returns scores; else binary {0,1}.
    """
    K, H, W = (len(grid.class_to_idx),) + grid.label_grid.shape
    ci = grid.class_to_idx[class_name]
    if use_probs and grid.prob_grids is not None:
        return grid.prob_grids[ci]
    else:
        return (grid.label_grid == ci).astype(np.float32)


# ----------------------------
# 3) Denoise & smooth (grid space)
# ----------------------------

def smooth_mask(mask: np.ndarray, tile_radius: float = 1.0, blur_sigma: Optional[float] = None,
                area_min_tiles: int = 0) -> np.ndarray:
    """Apply morphological closing+opening and optional Gaussian blur+thresholding to a 0..1 mask.
       tile_radius ~ how many tiles to consider when smoothing (e.g., 0.5..2.0).
       area_min_tiles removes small connected components (in tiles^2 units).
    """
    m = mask.copy()
    # Morphology uses boolean
    b = m > 0.5
    r = max(1, int(round(tile_radius)))
    selem = morphology.disk(r)
    b = morphology.binary_closing(b, selem)
    b = morphology.binary_opening(b, selem)
    if blur_sigma is not None and blur_sigma > 0:
        # convert to float, blur in [0,1], then threshold at 0.5
        f = filters.gaussian(b.astype(np.float32), sigma=blur_sigma, preserve_range=True)
        b = f > 0.5
    if area_min_tiles and area_min_tiles > 0:
        b = morphology.remove_small_objects(b, min_size=area_min_tiles)
    return b.astype(np.uint8)


# ----------------------------
# 4) Resolve overlaps between multiple class masks
# ----------------------------

def resolve_overlaps(masks: Dict[str, np.ndarray], priorities: Optional[List[str]] = None,
                     probs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Given per-class binary masks (grid space), ensure exclusivity.
       If probs provided, choose argmax in overlaps; else use priority order.
       Returns a dict of disjoint binary masks.
    """
    class_names = list(masks.keys())
    H, W = next(iter(masks.values())).shape

    if probs is not None:
        # stack probabilities (fallback to mask as prob if missing)
        P = []
        for c in class_names:
            if probs.get(c) is not None:
                P.append(probs[c].astype(np.float32))
            else:
                P.append(masks[c].astype(np.float32))
        P = np.stack(P, axis=0)  # (K,H,W)
        assign = np.argmax(P, axis=0)  # (H,W)
        out = {c: (assign == i).astype(np.uint8) & (np.any(P>0, axis=0)) for i,c in enumerate(class_names)}
        return out
    else:
        # priority order: earlier wins
        if priorities is None:
            priorities = class_names
        taken = np.zeros((H,W), dtype=np.uint8)
        out = {}
        for c in priorities:
            m = masks[c].astype(np.uint8)
            m = m & (~taken)
            out[c] = m
            taken |= m
        return out


# ----------------------------
# 5) Connected components per class
# ----------------------------

def connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return (labeled, num) from a binary mask in grid space."""
    labeled = measure.label(mask.astype(bool), connectivity=1)
    return labeled, int(labeled.max())


# ----------------------------
# 6) Convert components to polygons in SLIDE PIXELS
# ----------------------------

def component_to_polygon(component_mask: np.ndarray, x0: float, y0: float, tile_w: int, tile_h: int,
                          simplify_tol: Optional[float] = None) -> List[Polygon]:
    """Trace contours in grid units and map to slide pixels.
       Returns a list of shapely Polygons (may be empty).
    """
    # Use skimage.find_contours at level 0.5 on the binary mask
    contours = measure.find_contours(component_mask.astype(np.uint8), level=0.5)
    polys: List[Polygon] = []
    for cnt in contours:
        # cnt is array of (row, col) in grid space. Convert to tile-edge polygon in slide px.
        # Flip to (x,y), scale by tile_w/h, and offset by origin (min x/y).
        yy, xx = cnt[:, 0], cnt[:, 1]
        X = x0 + xx * tile_w
        Y = y0 + yy * tile_h
        coords = list(zip(X, Y))
        if len(coords) >= 3:
            poly = Polygon(coords)
            if simplify_tol and simplify_tol > 0:
                poly = poly.simplify(simplify_tol, preserve_topology=True)
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)
    return polys


def mask_to_polygons(mask: np.ndarray, grid: TileGrid, simplify_frac: float = 0.25) -> List[Polygon]:
    """Convert a binary mask (grid space) to slide-pixel polygons.
       simplify_frac is a fraction of tile size used as simplification tolerance.
    """
    x0, y0 = grid.x_coords.min(), grid.y_coords.min()
    tol = max(grid.tile_w, grid.tile_h) * simplify_frac
    # Find connected components first so holes are handled per region
    labeled, n = connected_components(mask)
    all_polys: List[Polygon] = []
    for k in range(1, n+1):
        comp = (labeled == k).astype(np.uint8)
        polys = component_to_polygon(comp, x0, y0, grid.tile_w, grid.tile_h, simplify_tol=tol)
        all_polys.extend(polys)
    return all_polys


# ----------------------------
# 7) Merge/clean/tag polygons
# ----------------------------

def merge_touching(polys: List[Polygon]) -> List[Polygon]:
    if not polys:
        return []
    u = unary_union(polys)
    if isinstance(u, Polygon):
        return [u]
    elif isinstance(u, MultiPolygon):
        return list(u.geoms)
    else:
        return []


def tag_polygons(polys: List[Polygon], class_name: str, min_area_px: int = 0) -> List[Dict]:
    out = []
    for p in polys:
        if min_area_px and p.area < min_area_px:
            continue
        out.append({
            'class': class_name,
            'area_px2': float(p.area),
            'perimeter_px': float(p.length),
            'geometry': mapping(p),  # GeoJSON-like dict
        })
    return out


# ----------------------------
# 8) Orchestrator & Export
# ----------------------------

def build_polygons_for_all_classes(
    df: pd.DataFrame,
    classes: List[str],
    tile_w: Optional[int] = None,
    tile_h: Optional[int] = None,
    use_probs: bool = False,
    priorities: Optional[List[str]] = None,
    smooth_radius_tiles: float = 1.0,
    blur_sigma: Optional[float] = None,
    area_min_tiles: int = 0,
    simplify_frac: float = 0.25,
    min_polygon_area_px: int = 0,
) -> List[Dict]:
    """Full pipeline returning a list of feature dicts with geometry + class.
    """
    grid = tiles_to_grid(df, classes, tile_w=tile_w, tile_h=tile_h,
                         prob_cols=None)  # extend if you have per-class probs

    # Build raw masks per class
    raw_masks: Dict[str, np.ndarray] = {}
    prob_masks: Dict[str, np.ndarray] = {}
    for c in classes:
        m = build_class_mask(grid, c, use_probs=False)
        raw_masks[c] = m
        if use_probs and grid.prob_grids is not None:
            prob_masks[c] = build_class_mask(grid, c, use_probs=True)

    # Smooth / denoise per class
    smoothed: Dict[str, np.ndarray] = {}
    for c in classes:
        smoothed[c] = smooth_mask(raw_masks[c], tile_radius=smooth_radius_tiles,
                                  blur_sigma=blur_sigma, area_min_tiles=area_min_tiles)

    # Resolve overlaps
    resolved = resolve_overlaps(smoothed, priorities=priorities,
                                probs=prob_masks if (use_probs and prob_masks) else None)

    # Polygons per class
    features: List[Dict] = []
    for c in classes:
        mask = resolved[c]
        polys = mask_to_polygons(mask, grid, simplify_frac=simplify_frac)
        polys = merge_touching(polys)
        tagged = tag_polygons(polys, c, min_area_px=min_polygon_area_px)
        features.extend(tagged)
    return features


def export_geojson(features: List[Dict], path: str) -> None:
    """Write features (with 'geometry' + 'class') to a GeoJSON FeatureCollection."""
    gj = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {k:v for k,v in f.items() if k != 'geometry'}, "geometry": f['geometry']}
            for f in features
        ]
    }
    with open(path, 'w') as f:
        json.dump(gj, f)


# ----------------------------
# Minimal usage example (not executed):
#
# df = pd.read_csv('tile_predictions.csv')  # columns: tile_index,x,y,predicted_class
# classes = [
#   "Tumor epithelium",
#   "Tumor-associated stroma (desmoplastic stroma)",
#   "Normal alveolar parenchyma",
#   "Bronchial epithelium / cartilage",
#   "Necrosis",
#   "Hemorrhage / blood",
#   "Vessel endothelium",
#   "Lymphoid aggregate / TLS",
#   "Adipose",
#   "Background / artifact",
# ]
# features = build_polygons_for_all_classes(
#     df, classes, tile_w=565, tile_h=565, priorities=classes,
#     smooth_radius_tiles=1.0, blur_sigma=None, area_min_tiles=3, simplify_frac=0.2,
#     min_polygon_area_px=3*565*565,
# )
# export_geojson(features, 'lung_annotations.geojson')





# ---- 1) Load a thumbnail and compute the scale from level-0 (full-res) ----
def load_svs_thumbnail(svs_path, size=None):
    """
    Returns (thumb PIL.Image, scale_x, scale_y, level0_size)
    - If size=None, uses the smallest pyramid level.
    - If size=(W,H), uses a scaled thumbnail.
    """
    slide = tiffslide.TiffSlide(svs_path)
    level0_w, level0_h = slide.level_dimensions[0]

    if size is None:
        level = slide.level_count - 1
        thumb = slide.read_region(location=(0, 0), level=level,
                                  size=slide.level_dimensions[level])
        thumb_w, thumb_h = slide.level_dimensions[level]
    else:
        thumb = slide.get_thumbnail(size=size)
        thumb_w, thumb_h = thumb.size

    scale_x = thumb_w / float(level0_w)
    scale_y = thumb_h / float(level0_h)
    return thumb, scale_x, scale_y, (level0_w, level0_h)


# ---- 2) Scale GeoJSON-like polygons from level-0 to thumbnail space ----
def scale_geometry_to_thumb(geom_dict, scale_x, scale_y):
    """
    geom_dict is a GeoJSON-like geometry (Polygon or MultiPolygon)
    in LEVEL-0 pixel coordinates.
    Returns a shapely geometry scaled to THUMBNAIL coords.
    """
    g = shape(geom_dict)
    # scale around origin (0,0) because coords are absolute pixel positions
    return affinity.scale(g, xfact=scale_x, yfact=scale_y, origin=(0, 0))


# ---- 3) Draw overlays for all classes on the same thumbnail ----
def plot_overlays_all_classes(thumb, features, class_colors=None, alpha=0.35, linewidth=1.0):
    """
    features: list of dicts with keys {'class', 'geometry', ...}
              geometry is GeoJSON-like (as returned by export_geojson or your pipeline)
    class_colors: optional dict {class_name: color}
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(thumb)
    ax = plt.gca()
    ax.set_axis_off()

    # default color palette if not provided
    if class_colors is None:
        default_palette = [
            "#d62728","#1f77b4","#2ca02c","#9467bd","#8c564b",
            "#e377c2","#7f7f7f","#bcbd22","#17becf","#ff7f0e"
        ]
        classes_seen = sorted({f["class"] for f in features})
        class_colors = {c: default_palette[i % len(default_palette)] for i, c in enumerate(classes_seen)}

    # group polygons by class
    by_class = defaultdict(list)
    for f in features:
        by_class[f["class"]].append(f["geometry"])

    # draw filled overlays + edges
    handles = []
    labels = []
    for cls, geoms in by_class.items():
        color = class_colors.get(cls, "#ff00ff")
        for gd in geoms:
            g = shape(gd)  # already in thumbnail space if you scaled earlier
            if isinstance(g, Polygon):
                ax.fill(*g.exterior.xy, facecolor=color, alpha=alpha, edgecolor=color, linewidth=linewidth)
                for ring in g.interiors:
                    ax.plot(*ring.xy, color=color, linewidth=linewidth)
            elif isinstance(g, MultiPolygon):
                for p in g.geoms:
                    ax.fill(*p.exterior.xy, facecolor=color, alpha=alpha, edgecolor=color, linewidth=linewidth)
                    for ring in p.interiors:
                        ax.plot(*ring.xy, color=color, linewidth=linewidth)

        # for legend
        handles.append(plt.Line2D([0],[0], color=color, lw=6, alpha=alpha))
        labels.append(cls)

    ax.legend(handles, labels, loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.show()


# ---- 4) Draw one class per image (quick browsing/export) ----
def plot_overlays_per_class(thumb, features, out_dir=None, alpha=0.35, linewidth=1.0):
    by_class = defaultdict(list)
    for f in features:
        by_class[f["class"]].append(f["geometry"])

    for cls, geoms in by_class.items():
        plt.figure(figsize=(8, 8))
        plt.imshow(thumb)
        ax = plt.gca()
        ax.set_axis_off()
        for gd in geoms:
            g = shape(gd)
            if isinstance(g, Polygon):
                ax.fill(*g.exterior.xy, facecolor="#ff0000", alpha=alpha, edgecolor="#ff0000", linewidth=linewidth)
                for ring in g.interiors:
                    ax.plot(*ring.xy, color="#ff0000", linewidth=linewidth)
            elif isinstance(g, MultiPolygon):
                for p in g.geoms:
                    ax.fill(*p.exterior.xy, facecolor="#ff0000", alpha=alpha, edgecolor="#ff0000", linewidth=linewidth)
                    for ring in p.interiors:
                        ax.plot(*ring.xy, color="#ff0000", linewidth=linewidth)
        plt.title(cls)
        plt.tight_layout()
        if out_dir:
            import os
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f"{out_dir}/{cls.replace('/','_')}.png", dpi=200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
