from __future__ import annotations
import numpy as np
import pandas as pd
import cv2
import tiffslide
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from PIL import Image, ImageDraw

def mask_contour_from_tiles(
    df: pd.DataFrame,
    wsi_path: str,
    patch_size: int,
    tumor_labels={"invasive tumor", "in-situ tumor"},
    xy_is_top_left: bool = True,
    mask_max_dim: int = 6000,          # longest side of the raster mask
    close_frac: float = 0.35,          # closing kernel ~ fraction of (patch_size at mask scale)
    open_frac: float = 0.12,           # opening kernel smaller than close
    min_island_tiles: int = 12,        # drop tiny islands by area (in tiles)
    simplify_tol_px: float = 2.0,      # simplify polygon in level-0 pixels
):
    # 1) choose raster scale from level-0 size
    slide = tiffslide.TiffSlide(str(wsi_path))
    W0, H0 = slide.level_dimensions[0]
    s = float(mask_max_dim) / max(W0, H0)
    W, H = max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))

    # 2) rasterize tumor tiles
    mask = np.zeros((H, W), np.uint8)
    half = patch_size / 2.0
    scaled_patch = patch_size * s

    sel = df[df["predicted_class"].isin(tumor_labels)]
    for x, y in zip(sel["x"].astype(float), sel["y"].astype(float)):
        if xy_is_top_left:
            x0, y0, x1, y1 = x, y, x + patch_size, y + patch_size
        else:
            x0, y0, x1, y1 = x - half, y - half, x + half, y + half
        ix0, iy0, ix1, iy1 = map(lambda v: int(round(v * s)), (x0, y0, x1, y1))
        cv2.rectangle(mask, (ix0, iy0), (ix1, iy1), 255, thickness=-1)

    # 3) morphology: CLOSE then OPEN
    def _odd(n): return int(n) + (1 - int(n) % 2)  # force odd >=1
    k_close = max(3, _odd(close_frac * scaled_patch))
    k_open  = max(3, _odd(open_frac  * scaled_patch))

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    if k_open >= 3:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2)

    # 4) drop small connected components
    min_area_mask_px = (patch_size**2) * (s**2) * max(1, int(min_island_tiles))
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_mask_px:
            clean[labels == i] = 255

    # 5) contours → shapely polygons at level-0 scale
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No tumor region after raster post-processing.")
    invs = 1.0 / s
    polys = []
    for c in cnts:
        pts = c[:, 0, :].astype(np.float64) * invs
        if len(pts) >= 3:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.area > 0:
                polys.append(poly)
    merged = unary_union(polys)
    if simplify_tol_px > 0:
        merged = merged.simplify(simplify_tol_px, preserve_topology=True)

    return merged, clean, s, (W, H)  # polygon at level-0, clean mask at thumbnail scale


def overlay_polygon_on_wsi_thumbnail_tiffslide(
    wsi_path, poly, max_dim=2000, fill_rgba=(255,0,0,90), outline="black", outline_w=2
):
    slide = tiffslide.TiffSlide(str(wsi_path))
    W0, H0 = slide.level_dimensions[0]
    level = slide.get_best_level_for_downsample(max(W0, H0)/float(max_dim))
    W, H = slide.level_dimensions[level]
    thumb = slide.read_region((0,0), level, (W, H)).convert("RGBA")
    sx, sy = W/float(W0), H/float(H0)
    from PIL import Image
    mask = Image.new("L", (W, H), 0)
    drawm = ImageDraw.Draw(mask)
    def draw_one(p):
        drawm.polygon([(x*sx, y*sy) for x,y in p.exterior.coords], fill=255)
        for ring in p.interiors:
            drawm.polygon([(x*sx, y*sy) for x,y in ring.coords], fill=0)
    (draw_one(poly) if isinstance(poly, Polygon) else [draw_one(p) for p in poly.geoms])
    overlay = Image.new("RGBA", (W, H), fill_rgba); overlay.putalpha(mask)
    out = Image.alpha_composite(thumb, overlay)
    # outline
    draw = ImageDraw.Draw(out)
    def outline_one(p):
        pts = [(x*sx, y*sy) for x,y in p.exterior.coords]
        draw.line(pts+[pts[0]], fill=outline, width=outline_w)
        for ring in p.interiors:
            pts = [(x*sx, y*sy) for x,y in ring.coords]
            draw.line(pts+[pts[0]], fill=outline, width=outline_w)
    (outline_one(poly) if isinstance(poly, Polygon) else [outline_one(p) for p in poly.geoms])
    return out