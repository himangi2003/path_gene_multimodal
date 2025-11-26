import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from shapely.geometry import shape, mapping
from shapely.affinity import scale as shapely_scale

from tiffslide import TiffSlide
from PIL import Image


# ----------------------------------------------------------------------
# 1. Export GeoJSON
# ----------------------------------------------------------------------
def export_geojson(
    features: List[Dict],
    wsi_path: str,
    base_output_dir: str,
    output_pt_path: Optional[str] = None,
) -> Path:
    """
    Write features (with 'geometry' + other properties like 'class')
    to a GeoJSON FeatureCollection.

    Default path:
        base_output_dir/<slide_name>/<slide_name>.geojson

    Returns:
        Path to saved GeoJSON file.
    """
    wsi = Path(wsi_path)
    slide_name = wsi.stem

    # Slide-specific directory
    outdir = Path(base_output_dir) / slide_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Explicit path or default
    out_path = Path(output_pt_path) if output_pt_path else (outdir / f"{slide_name}.geojson")

    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {k: v for k, v in f.items() if k != "geometry"},
                "geometry": f["geometry"],
            }
            for f in features
        ],
    }

    with open(out_path, "w") as f:
        json.dump(gj, f, indent=2)

    print(f"[✓] Saved GeoJSON ({len(features)} features) → {out_path}")
    return out_path


# ----------------------------------------------------------------------
# 2. Load TIFF WSI thumbnail with tiffslide
# ----------------------------------------------------------------------
def load_tiff_thumbnail(
    wsi_path: str | Path,
    size: Tuple[int, int] = (2000, 2000),
) -> Tuple[Image.Image, float, float, TiffSlide]:
    """
    Load a TIFF WSI using tiffslide and return:
        - thumb: PIL.Image thumbnail
        - sx: scale factor in x (thumb / level-0)
        - sy: scale factor in y (thumb / level-0)
        - slide: TiffSlide object
    """
    wsi_path = Path(wsi_path)
    slide = TiffSlide(str(wsi_path))

    level0_w, level0_h = slide.dimensions  # (width, height)
    # tiffslide implements a .get_thumbnail(size) similar to OpenSlide
    thumb = slide.get_thumbnail(size)

    thumb_w, thumb_h = thumb.size
    sx = thumb_w / level0_w
    sy = thumb_h / level0_h

    return thumb, sx, sy, slide


# ----------------------------------------------------------------------
# 3. Scale geometry from level-0 coords to thumbnail coords
# ----------------------------------------------------------------------
def scale_geometry_to_thumb(geom: Dict, sx: float, sy: float):
    """
    Scale a GeoJSON-like geometry from level-0 pixel coords into
    thumbnail coords, using sx, sy.
    Returns a Shapely geometry in thumbnail space.
    """
    g = shape(geom)
    g_scaled = shapely_scale(g, xfact=sx, yfact=sy, origin=(0, 0))
    return g_scaled


# ----------------------------------------------------------------------
# 4. Overlay ALL classes at once (for viewing only)
# ----------------------------------------------------------------------
def plot_overlays_all_classes(
    thumb: Image.Image,
    features_thumb: List[Dict],
    alpha: float = 0.4,
):
    """
    Show thumbnail with all class polygons overlaid.
    Does NOT save any files; purely for interactive viewing.
    """
    fig, ax = plt.subplots()
    ax.imshow(thumb)
    ax.axis("off")

    for f in features_thumb:
        geom = shape(f["geometry"])

        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            x, y = poly.exterior.xy
            patch = MplPolygon(
                list(zip(x, y)),
                closed=True,
                fill=True,
                alpha=alpha,
            )
            ax.add_patch(patch)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 5. Overlay PER class → save PNGs
# ----------------------------------------------------------------------
def sanitize_for_filename(s: str) -> str:
    """Make class names safe for filenames."""
    return (
        s.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def plot_overlays_per_class(
    thumb: Image.Image,
    features_thumb: List[Dict],
    out_dir: str | Path,
    alpha: float = 0.4,
) -> List[Path]:
    """
    For each unique 'class' in features_thumb, save a PNG overlay:

        <out_dir>/<class>_overlay.png

    Returns:
        List of Paths to saved overlays.
        The *last* overlay is saved_paths[-1].
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted({f["class"] for f in features_thumb})
    saved_paths: List[Path] = []

    for cls in classes:
        fig, ax = plt.subplots()
        ax.imshow(thumb)
        ax.axis("off")

        for f in features_thumb:
            if f["class"] != cls:
                continue

            geom = shape(f["geometry"])
            if geom.geom_type == "Polygon":
                polys = [geom]
            elif geom.geom_type == "MultiPolygon":
                polys = list(geom.geoms)
            else:
                continue

            for poly in polys:
                x, y = poly.exterior.xy
                patch = MplPolygon(
                    list(zip(x, y)),
                    closed=True,
                    fill=True,
                    alpha=alpha,
                )
                ax.add_patch(patch)

        plt.tight_layout()
        cls_clean = sanitize_for_filename(str(cls))
        out_path = out_dir / f"{cls_clean}_overlay.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[✓] Saved overlay for class '{cls}' → {out_path}")
        saved_paths.append(out_path)

    if saved_paths:
        print(f"[✓] Last overlay saved: {saved_paths[-1]}")

    return saved_paths

