"""
Generate slide list from DATA_PATH in tnbc_config.py
Usage: python generate_slide_list.py [output_file]
"""

import sys
from pathlib import Path

# Add module path
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tnbc_config as config


def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else "slide_list.txt"
    
    print(f"Scanning for WSI files in: {config.DATA_PATH}")
    print(f"Looking for extensions: {config.WSI_EXTS}")
    
    if not config.DATA_PATH.exists():
        print(f"ERROR: DATA_PATH does not exist: {config.DATA_PATH}")
        return 1
    
    # Find all WSI files
    wsi_files = sorted([
        p for p in config.DATA_PATH.iterdir()
        if p.is_file() and p.suffix.lower() in config.WSI_EXTS
    ])
    
    if not wsi_files:
        print(f"ERROR: No WSI files found in {config.DATA_PATH}")
        print(f"Expected extensions: {config.WSI_EXTS}")
        return 1
    
    # Write to file
    output_path = Path(output_file)
    with output_path.open('w') as f:
        for wsi in wsi_files:
            f.write(str(wsi.absolute()) + '\n')
    
    print(f"\n✅ Found {len(wsi_files)} slides")
    print(f"✅ Wrote list to: {output_path}")
    
    # Show summary by extension
    from collections import Counter
    ext_counts = Counter(p.suffix.lower() for p in wsi_files)
    print(f"\nBreakdown by extension:")
    for ext, count in sorted(ext_counts.items()):
        print(f"  {ext}: {count}")
    
    # Show first few files
    print(f"\nFirst 10 slides:")
    for i, wsi in enumerate(wsi_files[:10], 1):
        print(f"  {i:3d}. {wsi.name}")
    
    if len(wsi_files) > 10:
        print(f"  ... and {len(wsi_files)-10} more")
    
    print(f"\nNext steps:")
    print(f"  1. Review the slide list: cat {output_file}")
    print(f"  2. Update LSF script paths in run_wsi_pipeline.lsf")
    print(f"  3. Submit jobs: bsub < run_wsi_pipeline.lsf")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())