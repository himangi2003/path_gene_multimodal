import os
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

import mussel.cli.create_class_embeddings
from mussel.cli.create_class_embeddings import ClassEmbeddingConfig




def run_create_class_embeddings(
    classes: list[str],
    wsi_path: str | Path,
    base_output_dir: str | Path = "outputs",
    output_pt_path: Optional[str | Path] = None,
    model_type: str = "CLIP",   # use "Virchow2" to match Virchow2 image embeddings
    model_path: Optional[str] = None,                   # optional override (e.g., "paige-ai/Virchow2")
) -> str:
    """
    Generate class embeddings for a list of annotation classes and save as a .pt file.

    Args:
        classes: List of class names to embed.
        wsi_path: Path to the whole-slide image (used to name the output folder/file).
        base_output_dir: Base directory to write outputs into.
        output_pt_path: Optional explicit path to save the generated class embeddings (.pt).
        model_type: Text encoder family to use (e.g., "CLIP", "Virchow2").
        model_path: Optional HF repo or local checkpoint path for the text encoder.

    Returns:
        Absolute path to the saved .pt file.
    """
    if not classes:
        raise ValueError("`classes` must be a non-empty list of strings.")

    wsi = Path(wsi_path)
    slide_name = wsi.stem
    outdir = Path(base_output_dir) / slide_name
    outdir.mkdir(parents=True, exist_ok=True)

    out = Path(output_pt_path) if output_pt_path is not None else (outdir / f"{slide_name}_classes.pt")

    # Build config; some versions of ClassEmbeddingConfig may not accept model_type/model_path
    try:
        cfg = ClassEmbeddingConfig(
            classes=classes,
            output_pt_path=str(out),
            model_type=model_type,   # may not exist in older versions
            model_path=model_path,   # may not exist in older versions
        )
    except TypeError:
        cfg = ClassEmbeddingConfig(
            classes=classes,
            output_pt_path=str(out),
        )

    # Some versions expect an OmegaConf DictConfig; others accept the dataclass directly.
    try:
        mussel.cli.create_class_embeddings.main(OmegaConf.create(cfg))
    except Exception:
        mussel.cli.create_class_embeddings.main(cfg)

    if not out.exists():
        raise RuntimeError(f"Class embeddings not created: {out}")

    print(f"[class-embeddings] Done → {out.resolve()}")
    return str(out.resolve())
