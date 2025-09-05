import os
from pathlib import Path
from omegaconf import OmegaConf

import mussel.cli.extract_features
from mussel.cli.extract_features import ExtractFeaturesConfig


def run_extract_features_for_tessellation(
    wsi_path: str,
    base_output_dir: str = "outputs",
    patch_h5_path: str | None = None,
    model_type: str = "Virchow2",
    batch_size: int = 128,
    use_gpu: bool = True,
    num_workers: int = 16,
) -> dict:
    """
    Run Mussel feature extraction on tessellated patches.

    Args:
        wsi_path: Path to the whole-slide image.
        base_output_dir: Where tessellation/feature outputs are stored.
        patch_h5_path: Path to tessellation .h5 (defaults to outputs/<slide>/<slide>.h5).
        model_type: Which backbone to use (e.g. "Virchow2", "CLIP").
        batch_size: Batch size.
        use_gpu: Run on GPU if True, CPU if False.
        num_workers: Data loader workers.

    Returns:
        dict with {"features_h5", "features_pt", "tiles_h5", "outdir"}
    """
    wsi = Path(wsi_path)
    slide_name = wsi.stem
    outdir = Path(base_output_dir) / slide_name
    outdir.mkdir(parents=True, exist_ok=True)

    # default to standard tessellation output
    if patch_h5_path is None:
        patch_h5_path = outdir / f"{slide_name}.h5"
    patch_h5_path = Path(patch_h5_path)
    if not patch_h5_path.exists():
        raise FileNotFoundError(f"Tessellation file not found: {patch_h5_path}")

    features_h5 = outdir / f"{slide_name}_features.h5"
    features_pt = outdir / f"{slide_name}_features.pt"

    cfg = ExtractFeaturesConfig(
        slide_path=str(wsi),
        patch_h5_path=str(patch_h5_path),
        output_h5_path=str(features_h5),
        output_pt_path=str(features_pt),
        model_type=model_type,
        batch_size=batch_size,
        use_gpu=use_gpu,
        num_workers=num_workers,
    )

    mussel.cli.extract_features.main(OmegaConf.create(cfg))

    if not (features_h5.exists() and features_pt.exists()):
        raise RuntimeError("Feature extraction failed: outputs not created.")

    print(f"[extract_features] Done → {features_h5}, {features_pt}")
    return {
        "features_h5": str(features_h5),
        "features_pt": str(features_pt),
        "tiles_h5": str(patch_h5_path),
        "outdir": str(outdir),
    }
