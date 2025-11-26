import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from tiffslide import TiffSlide
from pathlib import Path
import tiffslide
from PIL import Image
import matplotlib.pyplot as plt

# ---------- helpers ----------
def base_no_ext(p): 
    return os.path.splitext(os.path.basename(p))[0]

def parse_asap_polygons(xml_path):
    """Return list of (name, group, type, [(x,y), ...]) from ASAP XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annos = root.findall(".//Annotation")
    if not annos:
        annos = root.findall(".//Annotations/Annotation")

    polys = []
    for ann in annos:
        name  = ann.get("Name") or ""
        a_type = ann.get("Type") or ""
        group = ann.get("PartOfGroup") or ""

        coords_elems = ann.findall(".//Coordinates")
        if not coords_elems:
            ce = ann.find("Coordinates")
            coords_elems = [ce] if ce is not None else []

        for coords_elem in coords_elems:
            pts = []
            for c in coords_elem.findall(".//Coordinate"):
                try:
                    order = int(c.get("Order"))
                except (TypeError, ValueError):
                    order = len(pts)
                x = float(c.get("X")); y = float(c.get("Y"))
                pts.append((order, x, y))
            pts.sort(key=lambda t: t[0])
            xy = [(x, y) for _, x, y in pts]
            if xy:
                polys.append((name, group, a_type, xy))
    return polys

def get_thumbnail_and_scale_tiffslide(image_path, max_dim=2048):
    """Return (thumb PIL.Image, scale_x, scale_y) relative to level-0 coords."""
    with TiffSlide(image_path) as slide:
        w0, h0 = slide.dimensions
        scale = min(max_dim / w0, max_dim / h0)
        tw, th = max(1, int(w0 * scale)), max(1, int(h0 * scale))
        thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    return thumb, (thumb.size[0] / w0), (thumb.size[1] / h0)

def draw_polygons(ax, polys, sx, sy, linewidth=1.2, alpha=0.9):
    for (_, _, _, xy) in polys:
        scaled = [(x * sx, y * sy) for (x, y) in xy]
        if scaled[0] != scaled[-1]:
            scaled = scaled + [scaled[0]]
        ax.add_patch(MplPolygon(scaled, fill=False, linewidth=linewidth, alpha=alpha))

# ---------- main entry ----------


def save_thumbnail_overlay_for_pair(slide_path, xml_path, out_dir, max_dim=2048, dpi=200):
    """
    Create and save a thumbnail overlay for one slide + ASAP XML pair.

    Args:
        slide_path (str|Path): Path to the .tif/.tiff image.
        xml_path   (str|Path): Path to the matching ASAP .xml file.
        out_dir    (str|Path): Output directory (created if missing).
        max_dim    (int): Max longest side of the thumbnail.
        dpi        (int): Output PNG DPI.

    Returns:
        str | None: Path to the saved PNG, or None on failure.
    """
    slide_path = Path(slide_path)
    xml_path   = Path(xml_path)
    out_dir    = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if not slide_path.exists():
        print(f"[ERR] Not found slide: {slide_path}")
        return None
    if not xml_path.exists():
        print(f"[ERR] Not found XML: {xml_path}")
        return None

    base = base_no_ext(slide_path.name)

    # Parse ASAP polygons
    try:
        polys = parse_asap_polygons(str(xml_path))
        if not polys:
            print(f"[WARN] No polygons in {xml_path.name}")
    except Exception as e:
        print(f"[ERR] Parsing {xml_path.name}: {e}")
        return None

    # Build thumbnail & scales via tiffslide
    try:
        thumb, sx, sy = get_thumbnail_and_scale_tiffslide(str(slide_path), max_dim=max_dim)
    except Exception as e:
        print(f"[ERR] Thumbnail for {slide_path.name}: {e}")
        return None

    # Plot overlay
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.imshow(thumb)
    if polys:
        draw_polygons(ax, polys, sx, sy)
    ax.set_axis_off()
    ax.set_title(f"{base} — {len(polys)} annotation(s)")

    # Save
    out_path = out_dir / f"{base}_thumb_overlay.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")
    return str(out_path)




def show_tiff_thumbnail(tif_path, size=None):
    """
    Display a thumbnail of a TIF whole-slide image using tiffslide.
    
    Parameters:
        tif_path (str): Path to the .tif file
        size (tuple or None): (width, height) for a fixed-size thumbnail.
                              If None, smallest slide level is used.
    """
    slide = tiffslide.TiffSlide(tif_path)

    if size is None:
        # Use smallest resolution level
        level = slide.level_count - 1
        thumb = slide.read_region(
            location=(0, 0),
            level=level,
            size=slide.level_dimensions[level]
        )
    else:
        # Use tiffslide's built-in thumbnail scaler
        thumb = slide.get_thumbnail(size=size)

    plt.figure(figsize=(6, 6))
    plt.imshow(thumb)
    plt.axis("off")
    plt.show()


import os

# Check current working directory
print("Current directory:", os.getcwd())

# Change directory
os.chdir("/lab/deasylab3/")

# Verify change
print("New directory:", os.getcwd())


data_path = 'Jung/tiger/wsibulk'
annotations = 'Jung/tiger/wsibulk/annotations-tumor-bulk'
images =  'Jung/tiger/wsibulk/images'
xmls = 'Jung/tiger/wsibulk/annotations-tumor-bulk/xmls'

annotation_files = os.listdir(annotations)
image_files = os.listdir(images)
xmls_files = os.listdir(xmls)

imgs_names = os.listdir(images)
imgs_names.sort()
xmls_names = os.listdir(xmls)
xmls_names.sort()



tif = image_files[2]
wsi_path = os.path.join(images, tif)
base = base_no_ext(tif)

# Find matching XML by base name
xml_candidates = [x for x in os.listdir(xmls)
                  if x.lower().endswith(".xml") and base_no_ext(x) == base]
xml_path = os.path.join(xmls, xml_candidates[0])

show_tiff_thumbnail(wsi_path)

out = "/cluster/home/srivash/venvs/Mussel/path_gene_multimodal"
out_dir = os.path.join(out, "_csv_out")
os.makedirs(out_dir, exist_ok=True)



