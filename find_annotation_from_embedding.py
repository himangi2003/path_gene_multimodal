import os
from pathlib import Path
from omegaconf import OmegaConf

import mussel.cli.annotate
from mussel.cli.annotate import AnnotateConfig


def run_annotation_for_extracted_features(
    wsi_path: str,
    class_embedding_pt_path: str,
    classes: list[str],
    base_output_dir: str = "outputs",
    output_csv_path: str | None = None,
) -> str:
    """
    Annotate tiles by comparing tile embeddings (from step 2) to class embeddings (step 3).

    Looks for:
        outputs/<slide>/<slide>_features.pt

    Writes:
        outputs/<slide>/<slide>_annotations.csv   (unless output_csv_path is provided)

    Args:
        wsi_path: Path to the WSI used earlier.
        class_embedding_pt_path: Path to the .pt from run_create_class_embeddings.
        classes: Class names to score/assign.
        base_output_dir: Root directory for outputs.
        output_csv_path: Optional override for the annotations CSV.

    Returns:
        str: Path to the created annotations CSV file.
    """
    slide = Path(wsi_path)
    slide_name = slide.stem
    outdir = Path(base_output_dir) / slide_name
    outdir.mkdir(parents=True, exist_ok=True)

    features_pt_path = outdir / f"{slide_name}_features.pt"
    if not features_pt_path.exists():
        raise FileNotFoundError(
            f"Tile embeddings not found: {features_pt_path}\n"
            "Run run_extract_features_for_tessellation(...) first."
        )

    class_pt = Path(class_embedding_pt_path)
    if not class_pt.exists():
        raise FileNotFoundError(f"Class embeddings file not found: {class_pt}")

    if output_csv_path is None:
        output_csv_path = outdir / f"{slide_name}_annotations.csv"
    out_csv = Path(output_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cfg = AnnotateConfig(
        features_pt_path=str(features_pt_path),
        classes=classes,
        class_embedding_pt_path=str(class_pt),
        output_csv_path=str(out_csv),
    )

    try:
        mussel.cli.annotate.main(OmegaConf.create(cfg))
    except Exception:
        mussel.cli.annotate.main(cfg)

    if not out_csv.exists():
        raise RuntimeError(f"Annotation failed: {out_csv} not created")

    print(f"[annotate] Done → {out_csv.resolve()}")
    return str(out_csv.resolve())
