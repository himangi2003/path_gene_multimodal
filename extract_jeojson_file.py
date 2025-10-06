from pathlib import Path
from typing import Optional, Set, Dict

import pandas as pd
from shapely.geometry import box, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union

def infer_patch_size_from_any_png(df: pd.DataFrame) -> Optional[int]:
    # Try to infer patch size from the first existing png_path
    try:
        from PIL import Image
        for p in df.get("png_path", []):
            if isinstance(p, str) and Path(p).exists():
                with Image.open(p) as im:
                    w, h = im.size
                if w != h:
                    raise ValueError(f"Patch is not square: {w}x{h}")
                return int(w)
    except Exception:
        pass
    return None

def tumor_polygon_from_patches(
    df: pd.DataFrame,
    positive_classes: Set[str] = frozenset({"invasive tumor", "in-situ tumor"}),
    patch_size: Optional[int] = None,   # if None, infer from pngs or fallback to 256
    xy_is_center: bool = False,         # set True if (x,y) are patch centers
    smooth_frac: float = 0.25,          # buffer(+r).buffer(-r), r = patch_size * frac
    simplify_tol: float = 0.0,          # Douglas-Peucker simplification tolerance (px)
    min_area: float = 0.0,              # drop pieces smaller than this area (px^2)
    take: str = "all",                  # "all" or "largest"
) -> Polygon | MultiPolygon:
    pos = df[df["predicted_class"].isin(positive_classes)]
    if pos.empty:
        raise ValueError("No positive patches found for the specified classes.")

    if patch_size is None:
        patch_size = infer_patch_size_from_any_png(pos) or 256
    half = patch_size / 2.0

    squares = []
    for x, y in zip(pos["x"].values, pos["y"].values):
        x = float(x); y = float(y)
        if xy_is_center:
            squares.append(box(x - half, y - half, x + half, y + half))
        else:
            squares.append(box(x, y, x + patch_size, y + patch_size))

    geom = unary_union(squares)

    if smooth_frac and smooth_frac > 0:
        r = patch_size * float(smooth_frac)
        geom = geom.buffer(r).buffer(-r)

    if min_area > 0:
        if isinstance(geom, Polygon):
            geom = geom if geom.area >= min_area else MultiPolygon([])
        else:
            parts = [g for g in geom.geoms if g.area >= min_area]
            geom = parts[0] if len(parts) == 1 else (MultiPolygon(parts) if parts else MultiPolygon([]))

    if simplify_tol and simplify_tol > 0:
        geom = geom.simplify(simplify_tol, preserve_topology=True)

    if take == "largest" and isinstance(geom, MultiPolygon) and len(geom.geoms) > 0:
        geom = max(geom.geoms, key=lambda p: p.area)

    return geom

def slide_name_from_png_path(p: str) -> str:
    # Given your paths like outputs/<SLIDE>/patches/125.png → take the parent folder name
    try:
        return Path(p).parents[1].name  # .../<SLIDE>/patches/<idx>.png
    except Exception:
        return "slide"

def build_tumor_polygons_for_all_slides(
    df: pd.DataFrame,
    positive_classes: Set[str] = frozenset({"invasive tumor", "in-situ tumor"}),
    xy_is_center: bool = False,
    patch_size: Optional[int] = None,
    smooth_frac: float = 0.25,
    simplify_tol: float = 0.0,
    min_area: float = 0.0,
    take: str = "all",
) -> Dict[str, Polygon | MultiPolygon]:
    # Ensure we have a 'slide' column to group by
    if "slide" not in df.columns:
        if "png_path" in df.columns:
            df = df.copy()
            df["slide"] = df["png_path"].apply(slide_name_from_png_path)
        else:
            df = df.copy()
            df["slide"] = "slide"

    results = {}
    for slide, g in df.groupby("slide", sort=False):
        geom = tumor_polygon_from_patches(
            g,
            positive_classes=positive_classes,
            patch_size=patch_size,      # if None, inferred per slide
            xy_is_center=xy_is_center,
            smooth_frac=smooth_frac,
            simplify_tol=simplify_tol,
            min_area=min_area,
            take=take,
        )
        results[slide] = geom
    return results

def save_polygons_to_geojson(polys_by_slide: Dict[str, Polygon | MultiPolygon], out_dir: str | Path) -> None:
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for slide, geom in polys_by_slide.items():
        feat = {"type": "Feature", "properties": {"slide": slide, "label": "tumor"}, "geometry": mapping(geom)}
        fc = {"type": "FeatureCollection", "features": [feat]}
        with open(out_dir / f"{slide}_tumor.geojson", "w") as f:
            json.dump(fc, f)