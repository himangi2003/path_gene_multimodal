import os
from pathlib import Path
from omegaconf import OmegaConf

import mussel.cli.tessellate
from mussel.cli.tessellate import TessellateConfig, SegConfig

def run_tessellation(wsi_path: str, base_output_dir: str = "outputs", workers: int = 4):
    """
    Run Mussel tessellation on a WSI and save results in an output folder
    named after the WSI file (without extension).

    Args:
        wsi_path (str): Path to the whole-slide image (.svs, .tif, etc.).
        base_output_dir (str): Root directory for results (default: 'outputs').
        workers (int): Number of parallel workers.

    Returns:
        str: Path to the output directory
    """
    wsi = Path(wsi_path)
    slide_name = wsi.stem
    outdir = Path(base_output_dir) / slide_name
    outdir.mkdir(parents=True, exist_ok=True)

    output_h5_path = outdir / f"{slide_name}.h5"

    seg_config = SegConfig(segment_threshold=20)

    cfg = TessellateConfig(
        slide_path=str(wsi),
        output_h5_path=str(output_h5_path),
        output_png_dir=str(outdir / "patches"),
        output_mask_path=str(outdir / "mask.png"),
        output_grid_mask_path=str(outdir / "grid_mask.png"),
        output_thumbnail_path=str(outdir / "thumbnail.png"),
        thumbnail_size=(1024, 1024),
        seg_config=seg_config,
        num_workers=workers,
    )

    mussel.cli.tessellate.main(OmegaConf.create(cfg))

    if output_h5_path.exists():
        print(f"Tessellation complete! Results saved in {outdir}")
        return str(outdir)
    else:
        raise RuntimeError(f"Tessellation failed for {wsi_path}")