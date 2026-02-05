from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import tiffslide

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity

from skimage import color, morphology, measure


# ============================================================
# Helpers
# ============================================================
def _clean_gdf(geojson_path: str | Path) -> gpd.GeoDataFrame:
    """Read + clean geometries (buffer(0), drop empty)."""
    gdf = gpd.read_file(str(geojson_path))
    gdf = gdf.set_crs(None, allow_override=True)
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    return gdf


def _to_polygons(geom) -> list[Polygon]:
    """Extract polygon parts from union output."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if isinstance(g, Polygon)]
    return []


def _iter_lines(geom):
    """Yield line-like parts that support `.xy`."""
    if geom is None:
        return
    if hasattr(geom, "xy"):
        yield geom
        return
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from _iter_lines(g)


# ============================================================
# 1) Tumor / TIL / TLS boundaries (thumbnail space)
# ============================================================
def get_tumor_til_tls_boundaries_thumbnail_space(
    geojson_path: str | Path,
    tumor_classes: list[str],
    til_classes: list[str],
    tls_classes: list[str],
    slide,
    thumb_hw: Tuple[int, int],
):
    """
    Returns:
      tumor_boundary_thumb, til_boundary_thumb, tls_boundary_thumb in THUMBNAIL coords
    """
    H, W = thumb_hw
    gdf = _clean_gdf(geojson_path)

    tumor = gdf[gdf["class"].isin(tumor_classes)]
    til = gdf[gdf["class"].isin(til_classes)]
    tls = gdf[gdf["class"].isin(tls_classes)]

    tumor_boundary = unary_union(list(tumor.geometry)).boundary if not tumor.empty else None
    til_boundary = unary_union(list(til.geometry)).boundary if not til.empty else None
    tls_boundary = unary_union(list(tls.geometry)).boundary if not tls.empty else None

    # Scale LEVEL-0 -> thumbnail coords
    level0_w, level0_h = slide.level_dimensions[0]
    sx = W / float(level0_w)
    sy = H / float(level0_h)

    def scale(g):
        if g is None or g.is_empty:
            return None
        return affinity.scale(g, xfact=sx, yfact=sy, origin=(0, 0))

    return scale(tumor_boundary), scale(til_boundary), scale(tls_boundary)


# ============================================================
# 2) Tissue boundary (thumbnail space) + mask
# ============================================================
def get_tissue_boundary_thumbnail_space(
    wsi_path: str | Path,
    thumb_size: Tuple[int, int] = (4000, 4000),
    sat_thresh: float = 0.04,
    close_radius: int = 6,
    min_object_size: int = 5000,
):
    """
    Returns:
      tissue_boundary_thumb : shapely boundary geometry in THUMB coords
      (H, W)               : thumbnail shape
      slide                : tiffslide.TiffSlide object
      tissue_mask          : HxW bool mask
    """
    slide = tiffslide.TiffSlide(str(wsi_path))
    thumb = slide.get_thumbnail(thumb_size)
    img = np.asarray(thumb)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    H, W = img.shape[:2]

    sat = color.rgb2hsv(img)[..., 1]
    tissue = sat > sat_thresh

    tissue = morphology.binary_closing(tissue, morphology.disk(close_radius))
    tissue = morphology.remove_small_objects(tissue, min_size=min_object_size)
    tissue = morphology.remove_small_holes(tissue, area_threshold=min_object_size)

    lab = measure.label(tissue, connectivity=2)
    props = measure.regionprops(lab)
    if not props:
        raise RuntimeError("No tissue detected in WSI thumbnail.")

    polys: list[Polygon] = []
    for p in props:
        mask = (lab == p.label).astype(np.uint8)
        contours = measure.find_contours(mask, level=0.5)
        if not contours:
            continue

        cnt = max(contours, key=len)  # (row=y, col=x)
        coords = [(float(x), float(y)) for y, x in cnt]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            polys.append(poly)

    tissue_geom = unary_union(polys)
    tissue_boundary = tissue_geom.boundary

    return tissue_boundary, (H, W), slide, tissue.astype(bool)


# ============================================================
# 3) Plot
# ============================================================
def plot_boundaries_only(
    tissue_boundary,
    tumor_boundary=None,
    til_boundary=None,
    tls_boundary=None,
    thumb_hw: Optional[Tuple[int, int]] = None,
    tissue_color="green",
    tumor_color="red",
    til_color="blue",
    tls_color="purple",
    tissue_lw=2.0,
    tumor_lw=1.5,
    til_lw=1.5,
    tls_lw=1.5,
    figsize=(10, 10),
    save_path: Optional[str | Path] = None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=figsize)

    for g in _iter_lines(tissue_boundary):
        ax.plot(*g.xy, color=tissue_color, linewidth=tissue_lw)

    for g in _iter_lines(tumor_boundary):
        ax.plot(*g.xy, color=tumor_color, linewidth=tumor_lw)

    for g in _iter_lines(til_boundary):
        ax.plot(*g.xy, color=til_color, linewidth=til_lw)

    for g in _iter_lines(tls_boundary):
        ax.plot(*g.xy, color=tls_color, linewidth=tls_lw)

    if thumb_hw is not None:
        H, W = thumb_hw
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 4) Island table (LEVEL-0)
# ============================================================
def island_table_one_slide_level0(
    slide_id: str,
    geojson_path: str | Path,
    tumor_classes: list[str],
    til_classes: list[str],
    tls_classes: list[str],
    tissue_area_px2: float,
) -> pd.DataFrame:
    """
    One row per island (tumor / til / tls) in LEVEL-0 coords.
    Includes area, perimeter, centroid, bbox + tissue area.
    """
    gdf = _clean_gdf(geojson_path)

    tumor = gdf[gdf["class"].isin(tumor_classes)]
    til = gdf[gdf["class"].isin(til_classes)]
    tls = gdf[gdf["class"].isin(tls_classes)]

    tumor_union = unary_union(list(tumor.geometry)) if not tumor.empty else None
    til_union = unary_union(list(til.geometry)) if not til.empty else None
    tls_union = unary_union(list(tls.geometry)) if not tls.empty else None

    rows: list[dict] = []

    def add_rows(polys: list[Polygon], typ: str):
        for idx, p in enumerate(polys, start=1):
            cx, cy = p.centroid.x, p.centroid.y
            xmin, ymin, xmax, ymax = p.bounds
            rows.append(
                {
                    "slide_id": slide_id,
                    "type": typ,
                    "island_id": idx,
                    "area_px2": float(p.area),
                    "perimeter_px": float(p.length),
                    "centroid_x": float(cx),
                    "centroid_y": float(cy),
                    "bbox_xmin": float(xmin),
                    "bbox_ymin": float(ymin),
                    "bbox_xmax": float(xmax),
                    "bbox_ymax": float(ymax),
                    "tissue_area_px2": float(tissue_area_px2),
                }
            )

    add_rows(_to_polygons(tumor_union), "tumor")
    add_rows(_to_polygons(til_union), "til")
    add_rows(_to_polygons(tls_union), "tls")

    return pd.DataFrame(rows)



def process_one_slide_make_csv_and_plot(
    wsi_path: str | Path,
    tumor_classes: list[str],
    til_classes: list[str],
    tls_classes: list[str],
    out_dir: str | Path = "outputs",
    geojson_path: Optional[str | Path] = None,
    csv_path: Optional[str | Path] = None,
    thumb_size: Tuple[int, int] = (4000, 4000),
    do_plot: bool = True,
) -> pd.DataFrame:
    """
    One-slide pipeline using only wsi_path (others inferred by default).

    Defaults:
      slide_id      = Path(wsi_path).stem
      slide_out_dir  = <out_dir>/<slide_id>/
      geojson_path   = <slide_out_dir>/<slide_id>.geojson
      csv_path       = <slide_out_dir>/<slide_id>_islands.csv
      plot_path      = <slide_out_dir>/<slide_id>_boundaries.png
    """
    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    slide_id = wsi_path.stem
    slide_out_dir = Path(out_dir) / slide_id
    slide_out_dir.mkdir(parents=True, exist_ok=True)

    if geojson_path is None:
        geojson_path = slide_out_dir / f"{slide_id}.geojson"
    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    if csv_path is None:
        csv_path = slide_out_dir / f"{slide_id}_islands.csv"
    csv_path = Path(csv_path)

    # 1) Tissue boundary + mask (thumbnail)
    tissue_boundary_thumb, (H, W), slide, tissue_mask = get_tissue_boundary_thumbnail_space(
        wsi_path=wsi_path,
        thumb_size=thumb_size,
        sat_thresh=0.04,
        close_radius=6,
        min_object_size=5000,
    )

    # 2) Tumor/TIL/TLS boundaries scaled to thumbnail
    tumor_b, til_b, tls_b = get_tumor_til_tls_boundaries_thumbnail_space(
        geojson_path=geojson_path,
        tumor_classes=tumor_classes,
        til_classes=til_classes,
        tls_classes=tls_classes,
        slide=slide,
        thumb_hw=(H, W),
    )

    # 3) Tissue AREA (level-0) from mask pixels scaled to level-0
    tissue_area_thumb_px2 = float(tissue_mask.sum())
    level0_w, level0_h = slide.level_dimensions[0]
    sx = level0_w / float(W)
    sy = level0_h / float(H)
    tissue_area_level0_px2 = tissue_area_thumb_px2 * sx * sy

    # 4) Island table (level-0)
    df = island_table_one_slide_level0(
        slide_id=slide_id,
        geojson_path=geojson_path,
        tumor_classes=tumor_classes,
        til_classes=til_classes,
        tls_classes=tls_classes,
        tissue_area_px2=tissue_area_level0_px2,
    )

    # 5) Save CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    # 6) Save plot
    if do_plot:
        plot_path = slide_out_dir / f"{slide_id}_boundaries.png"
        plot_boundaries_only(
            tissue_boundary=tissue_boundary_thumb,
            tumor_boundary=tumor_b,
            til_boundary=til_b,
            tls_boundary=tls_b,
            thumb_hw=(H, W),
            save_path=plot_path,
            show=False,
        )

    return df
