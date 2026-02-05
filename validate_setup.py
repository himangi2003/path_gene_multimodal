
"""
Validation script to check configuration and environment before running pipeline.
Usage: python validate_setup.py
"""

import sys
from pathlib import Path

# -------------------------------------------------
# Project root (dynamic, no hard-coded paths)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tnbc_config as config


def check_paths():
    """Validate all paths in configuration."""
    print("\n" + "="*70)
    print("CHECKING PATHS")
    print("="*70)
    
    errors = []
    warnings = []
    
    # Check DATA_PATH
    if not config.DATA_PATH.exists():
        errors.append(f"DATA_PATH does not exist: {config.DATA_PATH}")
    else:
        print(f"✓ DATA_PATH exists: {config.DATA_PATH}")
        
        # Count WSI files
        wsi_files = [
            p for p in config.DATA_PATH.iterdir() 
            if p.is_file() and p.suffix.lower() in config.WSI_EXTS
        ]
        print(f"  Found {len(wsi_files)} WSI files")
        
        if len(wsi_files) == 0:
            warnings.append(f"No WSI files found in {config.DATA_PATH}")
        else:
            # Show first few
            for i, f in enumerate(wsi_files[:5]):
                print(f"    - {f.name}")
            if len(wsi_files) > 5:
                print(f"    ... and {len(wsi_files)-5} more")
    
    # Check OUTROOT
    if not config.OUTROOT.exists():
        print(f"ℹ OUTROOT will be created: {config.OUTROOT}")
        try:
            config.OUTROOT.mkdir(parents=True, exist_ok=True)
            print(f"✓ Successfully created OUTROOT")
        except Exception as e:
            errors.append(f"Cannot create OUTROOT: {e}")
    else:
        print(f"✓ OUTROOT exists: {config.OUTROOT}")
    
    return errors, warnings


def check_config_values():
    """Validate configuration values."""
    print("\n" + "="*70)
    print("CHECKING CONFIGURATION VALUES")
    print("="*70)
    
    errors = []
    warnings = []
    
    # Check classes
    print(f"✓ Classes defined: {len(config.classes)}")
    for i, cls in enumerate(config.classes, 1):
        print(f"  {i}. {cls}")
    
    # Check tumor classes
    print(f"\n✓ Tumor classes: {len(config.TME_CLASSES)}")
    for cls in config.TME_CLASSES:
        if cls not in config.classes:
            errors.append(f"TUMOR_CLASS '{cls}' not in classes list")
        else:
            print(f"  - {cls}")
    
    # Check numeric values
    print(f"\n✓ Pipeline parameters:")
    print(f"  - PATCH_SIZE: {config.PATCH_SIZE}")
    print(f"  - MODEL_TYPE: {config.MODEL_TYPE}")
    print(f"  - USE_GPU: {config.USE_GPU}")
    print(f"  - BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"  - THUMB_SIZE: {config.THUMB_SIZE}")
    
    if config.PATCH_SIZE <= 0:
        errors.append(f"PATCH_SIZE must be positive: {config.PATCH_SIZE}")
    
    if config.BATCH_SIZE <= 0:
        errors.append(f"BATCH_SIZE must be positive: {config.BATCH_SIZE}")
    
    # Check polygon parameters
    print(f"\n✓ Polygon parameters:")
    print(f"  - SMOOTH_RADIUS_TILES: {config.SMOOTH_RADIUS_TILES}")
    print(f"  - BLUR_SIGMA: {config.BLUR_SIGMA}")
    print(f"  - AREA_MIN_TILES: {config.AREA_MIN_TILES}")
    print(f"  - SIMPLIFY_FRAC: {config.SIMPLIFY_FRAC}")
    print(f"  - MIN_POLYGON_AREA_PX: {config.MIN_POLYGON_AREA_PX}")
    
    return errors, warnings


def check_imports():
    """Check if all required modules can be imported."""
    print("\n" + "="*70)
    print("CHECKING IMPORTS")
    print("="*70)
    
    errors = []
    
    required_modules = [
        "tiling",
        "extract_embedding_from_tiles",
        "create_embedding",
        "find_annotation_from_embedding",
        "load_annotation_with_coordinates",
        "create_and_overlay_polygon_from_prediction",
    ]
    
    for module_name in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            errors.append(f"Cannot import {module_name}: {e}")
            print(f"✗ {module_name}: {e}")
    
    return errors


def check_gpu():
    """Check GPU availability if USE_GPU is True."""
    print("\n" + "="*70)
    print("CHECKING GPU")
    print("="*70)
    
    warnings = []
    
    if config.USE_GPU:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ GPU available")
                print(f"  - Device count: {torch.cuda.device_count()}")
                print(f"  - Current device: {torch.cuda.current_device()}")
                print(f"  - Device name: {torch.cuda.get_device_name(0)}")
            else:
                warnings.append("USE_GPU=True but CUDA not available")
                print(f"⚠ CUDA not available but USE_GPU=True")
        except ImportError:
            warnings.append("USE_GPU=True but PyTorch not installed")
            print(f"⚠ PyTorch not installed")
    else:
        print(f"ℹ GPU not required (USE_GPU=False)")
    
    return warnings


def main():
    """Run all validation checks."""
    print("\n" + "="*70)
    print("PIPELINE CONFIGURATION VALIDATION")
    print("="*70)
    print(f"Config file: tnbc_config.py")
    print(f"Module path: {PROJECT_ROOT}")
    
    all_errors = []
    all_warnings = []
    
    # Run checks
    errors, warnings = check_paths()
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    
    errors, warnings = check_config_values()
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    
    errors = check_imports()
    all_errors.extend(errors)
    
    warnings = check_gpu()
    all_warnings.extend(warnings)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if all_errors:
        print(f"\n❌ ERRORS ({len(all_errors)}):")
        for i, err in enumerate(all_errors, 1):
            print(f"  {i}. {err}")
    
    if all_warnings:
        print(f"\n⚠️  WARNINGS ({len(all_warnings)}):")
        for i, warn in enumerate(all_warnings, 1):
            print(f"  {i}. {warn}")
    
    if not all_errors and not all_warnings:
        print("\n✅ All checks passed! Configuration is valid.")
        print("\nReady to submit jobs:")
        print("  ./setup_and_submit.sh /path/to/slides")
        return 0
    elif not all_errors:
        print("\n✅ No critical errors. Warnings may be ignored.")
        print("\nReady to submit jobs (with warnings):")
        print("  ./setup_and_submit.sh /path/to/slides")
        return 0
    else:
        print("\n❌ Critical errors found. Fix errors before running.")
        return 1


if __name__ == "__main__":
    sys.exit(main())