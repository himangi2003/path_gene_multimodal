
import os
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime

# If your project modules live here, keep it. Make it safe:
MUSSEL_PATH = "/cluster/home/srivash/venvs/Mussel/path_gene_multimodal"
if MUSSEL_PATH not in sys.path:
    sys.path.append(MUSSEL_PATH)

import tnbc_config as config

from tiling import run_tessellation
from extract_embedding_from_tiles import run_extract_features_for_tessellation
from create_embedding import run_create_class_embeddings
from find_annotation_from_embedding import run_annotation_for_extracted_features
from load_annotation_with_coordinates import load_annotations_with_coords
from create_and_overlay_polygon_from_prediction import (
    build_polygons_for_all_classes,
    export_geojson,
    load_svs_thumbnail,
    scale_geometry_to_thumb,
    plot_overlays_all_classes,
    plot_overlays_per_class,
)


def already_done(out_dir: Path) -> bool:
    done_flag = out_dir / config.DONE_FLAG_NAME
    if done_flag.exists():
        return True

    # Fallback (older runs): consider done if it has both png + geojson
    overlay_pngs = list(out_dir.rglob("*.png"))
    geojsons = list(out_dir.rglob("*.geojson"))
    return (len(overlay_pngs) > 0) and (len(geojsons) > 0)


def write_done_flag(out_dir: Path, payload: dict) -> None:
    (out_dir / config.DONE_FLAG_NAME).write_text(json.dumps(payload, indent=2) + "\n")


def run_one_wsi(wsi_path: Path) -> None:
    slide_name = wsi_path.stem
    out_dir = config.OUTROOT / slide_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if already_done(out_dir):
        print(f"[SKIP] {slide_name} already done: {out_dir}")
        return

    print(f"[RUN] {slide_name}")
    print(f"      WSI: {wsi_path}")
    print(f"      OUT: {out_dir}")

    # 1) Tiling
    run_tessellation(wsi_path=wsi_path, Patch_size=config.PATCH_SIZE, base_output_dir=out_dir)

    # 2) Feature extraction
    run_extract_features_for_tessellation(
        wsi_path,
        base_output_dir=out_dir,
        model_type=config.MODEL_TYPE,
        use_gpu=config.USE_GPU,
        batch_size=config.BATCH_SIZE,
    )

    # 3) Create embeddings
    class_pt = run_create_class_embeddings(config.classes, wsi_path, out_dir)

    # 4) Annotate tiles
    csv_path = run_annotation_for_extracted_features(
        wsi_path,
        class_embedding_pt_path=class_pt,
        classes=config.classes,
        base_output_dir=out_dir,
    )

    # 5) Load annotations with coordinates
    df = load_annotations_with_coords(
        wsi_path=wsi_path,
        classes=config.classes,
        tumor_classes=config.TUMOR_CLASSES,
        base_output_dir=out_dir,
    )

    # 6) Build polygons
    features = build_polygons_for_all_classes(
        df,
        config.classes,
        tile_w=None,
        tile_h=None,
        priorities=config.classes,
        smooth_radius_tiles=config.SMOOTH_RADIUS_TILES,
        blur_sigma=config.BLUR_SIGMA,
        area_min_tiles=config.AREA_MIN_TILES,
        simplify_frac=config.SIMPLIFY_FRAC,
        min_polygon_area_px=config.MIN_POLYGON_AREA_PX,
    )

    # 7) Export GeoJSON
    export_geojson(features=features, wsi_path=wsi_path, base_output_dir=out_dir, output_pt_path=None)

    # 8) Thumbnail + overlays
    thumb, sx, sy, _ = load_svs_thumbnail(str(wsi_path), size=config.THUMB_SIZE)

    features_thumb = []
    for f in features:
        g_thumb = scale_geometry_to_thumb(f["geometry"], sx, sy)
        features_thumb.append(
            {"class": f["class"], "geometry": json.loads(json.dumps(g_thumb.__geo_interface__))}
        )

    out_path = plot_overlays_all_classes(
        thumb, features_thumb, wsi_path=wsi_path, base_output_dir=out_dir, show=False
    )

    saved = plot_overlays_per_class(
        thumb, features_thumb, wsi_path=wsi_path, base_output_dir=out_dir
    )

    # DONE
    write_done_flag(
        out_dir,
        payload={
            "slide_name": slide_name,
            "wsi_path": str(wsi_path),
            "out_dir": str(out_dir),
            "timestamp": datetime.now().isoformat(),
            "csv_path": str(csv_path) if csv_path is not None else "",
            "overlay_all_path": str(out_path) if out_path is not None else "",
            "per_class_outputs": str(saved) if saved is not None else "",
            "status": "ok",
        },
    )

    print(f"[OK] {slide_name} completed ✅")


def main() -> None:
    wsi_env = os.environ.get("WSI_PATH", "").strip()
    if not wsi_env:
        raise RuntimeError("WSI_PATH env var not set (LSF array sets this).")

    wsi_path = Path(wsi_env)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    config.OUTROOT.mkdir(parents=True, exist_ok=True)

    try:
        run_one_wsi(wsi_path)
    except Exception as e:
        slide_name = wsi_path.stem
        out_dir = config.OUTROOT / slide_name
        out_dir.mkdir(parents=True, exist_ok=True)
        err_txt = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        (out_dir / "_ERROR.txt").write_text(err_txt + "\n")
        print(f"[FAIL] {slide_name} ❌")
        print(err_txt)
        raise


if __name__ == "__main__":
    main()
