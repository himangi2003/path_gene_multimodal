import os
import sys
from pathlib import Path
from datetime import datetime
import json
import traceback

# CRITICAL FIX: Add module path BEFORE importing custom modules
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def try_acquire_lock(out_dir: Path) -> bool:
    """Try to acquire processing lock to prevent race conditions."""
    lock_file = out_dir / ".processing.lock"
    try:
        # exist_ok=False means this will fail if file already exists
        lock_file.touch(exist_ok=False)
        # Write PID and timestamp for debugging
        lock_file.write_text(f"PID: {os.getpid()}\nStarted: {datetime.now().isoformat()}\n")
        return True
    except FileExistsError:
        # Check if lock is stale (older than 48 hours)
        if lock_file.exists():
            age_hours = (datetime.now().timestamp() - lock_file.stat().st_mtime) / 3600
            if age_hours > 48:
                print(f"  WARNING: Removing stale lock (age: {age_hours:.1f} hours)")
                lock_file.unlink()
                return try_acquire_lock(out_dir)  # Try again
        return False


def release_lock(out_dir: Path) -> None:
    """Release processing lock."""
    lock_file = out_dir / ".processing.lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
        except Exception as e:
            print(f"  WARNING: Could not remove lock file: {e}")


def already_done(out_dir: Path) -> bool:
    """Check if slide has already been processed."""
    done_flag = out_dir / config.DONE_FLAG_NAME
    if done_flag.exists():
        return True

    # Fallback (older runs): consider done if it has both png + geojson
    overlay_pngs = list(out_dir.rglob("*.png"))
    geojsons = list(out_dir.rglob("*.geojson"))
    return (len(overlay_pngs) > 0) and (len(geojsons) > 0)


def write_done_flag(out_dir: Path, payload: dict) -> None:
    """Write completion flag with metadata."""
    done_path = out_dir / config.DONE_FLAG_NAME
    done_path.write_text(json.dumps(payload, indent=2) + "\n")


def validate_wsi_path(wsi_path: Path) -> None:
    """Validate WSI file exists and has correct format."""
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")
    
    if not wsi_path.is_file():
        raise ValueError(f"WSI path is not a file: {wsi_path}")
    
    if wsi_path.suffix.lower() not in config.WSI_EXTS:
        raise ValueError(
            f"Invalid WSI format: {wsi_path.suffix}. "
            f"Expected one of {config.WSI_EXTS}"
        )


def run_one_wsi(wsi_path: Path) -> None:
    """Process a single WSI through the entire pipeline."""
    slide_name = wsi_path.stem
    out_dir = config.OUTROOT / slide_name
    
    # Create output directory with proper error handling
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise RuntimeError(f"Cannot create output directory {out_dir}: {e}")

    # Check if already completed
    if already_done(out_dir):
        print(f"[SKIP] {slide_name} already done: {out_dir}")
        return

    # Try to acquire lock to prevent race conditions
    if not try_acquire_lock(out_dir):
        print(f"[SKIP] {slide_name} is being processed by another job")
        return

    try:
        print(f"\n{'='*70}")
        print(f"[RUN] {slide_name}")
        print(f"{'='*70}")
        print(f"WSI:    {wsi_path}")
        print(f"Output: {out_dir}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        # 1) Tiling
        print(f"[1/8] Running tessellation (patch size: {config.PATCH_SIZE})...")
        run_tessellation(
            wsi_path=wsi_path, 
            Patch_size=config.PATCH_SIZE, 
            base_output_dir=out_dir
        )
        print(f"      ✓ Tessellation complete")

        # 2) Feature extraction
        print(f"[2/8] Extracting features (model: {config.MODEL_TYPE}, batch: {config.BATCH_SIZE})...")
        run_extract_features_for_tessellation(
            wsi_path,
            base_output_dir=out_dir,
            model_type=config.MODEL_TYPE,
            use_gpu=config.USE_GPU,
            batch_size=config.BATCH_SIZE,
        )
        print(f"      ✓ Feature extraction complete")

        # 3) Create embeddings
        print(f"[3/8] Creating class embeddings ({len(config.classes)} classes)...")
        class_pt = run_create_class_embeddings(config.classes, wsi_path, out_dir)
        if class_pt is None:
            raise RuntimeError("Failed to create class embeddings")
        print(f"      ✓ Class embeddings created: {class_pt}")

        # 4) Annotate tiles
        print(f"[4/8] Annotating tiles...")
        csv_path = run_annotation_for_extracted_features(
            wsi_path,
            class_embedding_pt_path=class_pt,
            classes=config.classes,
            base_output_dir=out_dir,
        )
        print(f"      ✓ Tile annotation complete")

        # 5) Load annotations with coordinates
        print(f"[5/8] Loading annotations with coordinates...")
        df = load_annotations_with_coords(
            wsi_path=wsi_path,
            classes=config.classes,
            tumor_classes=config.TME_CLASSES,
            base_output_dir=out_dir,
        )
        if df is None or df.empty:
            raise RuntimeError("No annotations loaded - empty dataframe")
        print(f"      ✓ Loaded {len(df)} annotated tiles")

        # 6) Build polygons
        print(f"[6/8] Building polygons...")
        features = build_polygons_for_all_classes(
            df,
            config.classes,
            tile_w=config.PATCH_SIZE,  # FIX: Use actual patch size
            tile_h=config.PATCH_SIZE,  # FIX: Use actual patch size
            priorities=config.classes,
            smooth_radius_tiles=config.SMOOTH_RADIUS_TILES,
            blur_sigma=config.BLUR_SIGMA,
            area_min_tiles=config.AREA_MIN_TILES,
            simplify_frac=config.SIMPLIFY_FRAC,
            min_polygon_area_px=config.MIN_POLYGON_AREA_PX,
        )
        print(f"      ✓ Built {len(features)} polygon features")

        # 7) Export GeoJSON
        print(f"[7/8] Exporting GeoJSON...")
        geojson_path = export_geojson(
            features=features, 
            wsi_path=wsi_path, 
            base_output_dir=out_dir, 
            output_pt_path=None
        )
        print(f"      ✓ GeoJSON exported")

        # 8) Thumbnail + overlays
        print(f"[8/8] Creating overlay visualizations (thumb size: {config.THUMB_SIZE})...")
        thumb, sx, sy, _ = load_svs_thumbnail(str(wsi_path), size=config.THUMB_SIZE)
        print(f"      - Thumbnail loaded: {thumb.shape}")

        features_thumb = []
        for f in features:
            g_thumb = scale_geometry_to_thumb(f["geometry"], sx, sy)
            features_thumb.append({
                "class": f["class"], 
                "geometry": json.loads(json.dumps(g_thumb.__geo_interface__))
            })

        out_path = plot_overlays_all_classes(
            thumb, features_thumb, wsi_path=wsi_path, 
            base_output_dir=out_dir, show=False
        )
        print(f"      - All classes overlay: {out_path}")

        saved = plot_overlays_per_class(
            thumb, features_thumb, wsi_path=wsi_path, 
            base_output_dir=out_dir
        )
        print(f"      ✓ Overlay visualizations complete")

        # Write completion flag
        write_done_flag(
            out_dir,
            payload={
                "slide_name": slide_name,
                "wsi_path": str(wsi_path),
                "out_dir": str(out_dir),
                "timestamp": datetime.now().isoformat(),
                "csv_path": str(csv_path) if csv_path else "",
                "geojson_path": str(geojson_path) if geojson_path else "",
                "overlay_all_path": str(out_path) if out_path else "",
                "per_class_outputs": str(saved) if saved else "",
                "num_features": len(features),
                "num_tiles": len(df),
                "classes_processed": config.classes,
                "patch_size": config.PATCH_SIZE,
                "model_type": config.MODEL_TYPE,
                "status": "ok",
            },
        )

        print(f"\n{'='*70}")
        print(f"[OK] {slide_name} completed ✅")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
    finally:
        # Always release lock, even if processing failed
        release_lock(out_dir)


def main() -> None:
    """Main entry point."""
    # Get WSI path from environment variable (set by LSF array job)
    wsi_env = os.environ.get("WSI_PATH", "").strip()
    if not wsi_env:
        raise RuntimeError(
            "WSI_PATH environment variable not set. "
            "This should be set by the LSF array job script."
        )

    wsi_path = Path(wsi_env)
    
    # Validate WSI path
    validate_wsi_path(wsi_path)
    
    # Ensure output root exists
    config.OUTROOT.mkdir(parents=True, exist_ok=True)

    # Process the slide
    try:
        run_one_wsi(wsi_path)
    except Exception as e:
        slide_name = wsi_path.stem
        out_dir = config.OUTROOT / slide_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Write detailed error information
        err_txt = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        error_file = out_dir / "_ERROR.txt"
        error_file.write_text(
            f"Slide: {slide_name}\n"
            f"WSI Path: {wsi_path}\n"
            f"Timestamp: {datetime.now().isoformat()}\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Message: {str(e)}\n"
            f"\n{'='*70}\n"
            f"Full Traceback:\n"
            f"{'='*70}\n"
            f"{err_txt}\n"
        )
        
        # Also release lock on error
        release_lock(out_dir)
        
        print(f"\n{'='*70}")
        print(f"[FAIL] {slide_name} ❌")
        print(f"{'='*70}")
        print(err_txt)
        print(f"Error details written to: {error_file}")
        print(f"{'='*70}\n")
        raise


if __name__ == "__main__":
    main()