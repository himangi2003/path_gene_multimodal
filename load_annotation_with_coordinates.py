import pandas as pd
from typing import Optional
from pathlib import Path
import h5py
from shapely.geometry import box
from shapely.ops import unary_union


def load_annotations_with_coords(
    wsi_path: str | Path,
    classes: list[str],
    tumor_classes: list[str],
    base_output_dir: str | Path = "outputs",
    annotations_csv: Optional[str | Path] = None,
    tiles_h5_path: Optional[str | Path] = None,
    patches_dir: Optional[str | Path] = None,
    save_merged: bool = True,
    merged_csv_name: Optional[str] = None,
    # TME ROI params
    add_tme_roi: bool = True,
    patch_size: int = 508,
    tme_margin_factor: float = 2.0,
) -> pd.DataFrame:
    """
    Load annotation CSV and enrich with tile coordinates (and optional patch filenames),
    compute predicted_class and optionally a TME-over-tumor flag column `in_tme_roi`,
    then optionally save the merged dataframe to disk.

    Args:
        wsi_path (str | pathlib.Path):
            Path to the whole-slide image (WSI). Only the stem (slide name) is
            used to infer default output locations.
        classes (list[str]):
            List of class column names in the annotations CSV. Each column should
            contain a score/probability per tile. Used to compute `predicted_class`
            via argmax across these columns.
        tumor_classes (list[str]):
            Subset of `classes` that should be considered tumor classes when building
            the tumor mask used for the TME ROI (e.g. ["Tumor epithelium"]).
        base_output_dir (str | pathlib.Path, optional):
            Base directory in which per-slide outputs are stored. Defaults to "outputs".
        annotations_csv (str | pathlib.Path | None, optional):
            Path to the annotations CSV. If None, this is assumed to be:
                <base_output_dir>/<slide>/<slide>_annotations.csv
            where <slide> is the stem of `wsi_path`.
        tiles_h5_path (str | pathlib.Path | None, optional):
            Path to the HDF5 file containing tile coordinates. If None, this is:
                <base_output_dir>/<slide>/<slide>.h5
        patches_dir (str | pathlib.Path | None, optional):
            Directory containing PNG patches. If None, this is assumed to be:
                <base_output_dir>/<slide>/patches
            If that directory does not exist, no `png_path` column is added.
        save_merged (bool, optional):
            If True, save the merged DataFrame (annotations + coordinates and
            optional TME ROI) as a CSV file. Defaults to True.
        merged_csv_name (str | None, optional):
            Filename for the merged CSV. If None, defaults to:
                <slide>_annotations_with_coords.csv
            under <base_output_dir>/<slide>/.
        add_tme_roi (bool, optional):
            If True, compute TME region-of-interest and add a boolean
            column `in_tme_roi`. Defaults to True.
        patch_size (int, optional):
            Size (in WSI coordinate units, typically pixels) of each tile/patch.
            Used to build square polygons for spatial operations. Defaults to 508.
        tme_margin_factor (float, optional):
            Factor multiplied by `patch_size` to obtain the buffer distance
            around the union of tumor tiles when defining the TME ROI.
            For example, 2.0 means a margin of 2 * patch_size. Defaults to 2.0.

    Returns:
        pandas.DataFrame:
            DataFrame containing:
                - original annotation columns,
                - tile_index,
                - x, y (tile top-left coordinates),
                - optional `level` (if available in H5),
                - optional `png_path` (if patches_dir is present),
                - `predicted_class` (argmax over `classes`),
                - optional `in_tme_roi` (bool) if `add_tme_roi` is True.

    Raises:
        FileNotFoundError:
            If the annotations CSV or HDF5 tiles file cannot be found.
        KeyError:
            If required columns or any of the specified class columns are missing.
        RuntimeError:
            If no coordinate datasets can be located within the HDF5 file.
        ValueError:
            If no tumor tiles are found for the specified `tumor_classes`,
            or no TME tiles are found for the given `classes`.
    """
    slide = Path(wsi_path)
    name = slide.stem
    outdir = Path(base_output_dir) / name

    # defaults
    if annotations_csv is None:
        annotations_csv = outdir / f"{name}_annotations.csv"
    if tiles_h5_path is None:
        tiles_h5_path = outdir / f"{name}.h5"
    if patches_dir is None:
        pdir = outdir / "patches"
        patches_dir = pdir if pdir.exists() else None

    annotations_csv = Path(annotations_csv)
    tiles_h5_path = Path(tiles_h5_path)
    patches_dir = Path(patches_dir) if patches_dir is not None else None

    if not annotations_csv.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {annotations_csv}")
    if not tiles_h5_path.exists():
        raise FileNotFoundError(f"Tessellation H5 not found: {tiles_h5_path}")

    df = pd.read_csv(annotations_csv)

    # Need a tile index to join on. If missing, use row order as tile_index.
    if "tile_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "tile_index"})

    # -------- Read coords from H5 (try multiple common layouts) --------
    with h5py.File(tiles_h5_path, "r") as f:
        candidates = [
            ("coords", None),            # Nx2 or Nx3
            ("locations", None),
            ("tiles/coords", None),
            ("x", "y"),                  # separate arrays
            ("tiles/x", "tiles/y"),
        ]

        x = y = level = None

        for cand in candidates:
            if cand[1] is None:
                ds = cand[0]
                if ds in f:
                    arr = f[ds][:]
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        x = arr[:, 0]
                        y = arr[:, 1]
                        if arr.shape[1] >= 3:
                            level = arr[:, 2]
                        break
            else:
                dsx, dsy = cand
                if dsx in f and dsy in f:
                    x = f[dsx][:]
                    y = f[dsy][:]
                    lvl_key = "level" if "level" in f else ("tiles/level" if "tiles/level" in f else None)
                    level = f[lvl_key][:] if lvl_key else None
                    break

        if x is None or y is None:
            # final fallback: scan for any "*coords" 2D dataset
            for key in f.keys():
                if key.lower().endswith("coords"):
                    arr = f[key][:]
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        x = arr[:, 0]
                        y = arr[:, 1]
                        level = arr[:, 2] if arr.shape[1] >= 3 else None
                        break

        if x is None or y is None:
            raise RuntimeError("Could not find coordinate datasets in the H5 file.")

        meta = {"tile_index": range(len(x)), "x": x, "y": y}
        if level is not None:
            meta["level"] = level
        df_coords = pd.DataFrame(meta)

    # -------- Join annotations with coords on tile_index --------
    df_merged = df.merge(df_coords, on="tile_index", how="left")

    # Optional: add PNG paths if patches were saved
    if patches_dir is not None:
        df_merged["png_path"] = df_merged.apply(
            lambda r: str(patches_dir / f"{int(r.x)}_{int(r.y)}.png"),
            axis=1,
        )

    # -------- Compute predicted class by argmax over class columns --------
    missing = [c for c in classes if c not in df_merged.columns]
    if missing:
        raise KeyError(f"Missing class score columns in annotations CSV: {missing}")
    df_merged["predicted_class"] = df_merged[classes].idxmax(axis=1)

    # -------- Optionally mark TME tiles (adds in_tme_roi) --------
    if add_tme_roi:
        # check required columns
        for col in ["x", "y", "tile_index", "predicted_class"]:
            if col not in df_merged.columns:
                raise KeyError(f"Column '{col}' is missing from merged dataframe.")

        tme_classes = classes
        tme_margin = patch_size * tme_margin_factor

        def make_patch_poly(xv, yv, size=patch_size):
            # (x, y) are TOP-LEFT coords of patch in WSI space
            return box(xv, yv, xv + size, yv + size)

        # tumor tiles
        df_tumor = df_merged[df_merged["predicted_class"].isin(tumor_classes)]
        if df_tumor.empty:
            raise ValueError(f"No tumor tiles found for tumor classes: {tumor_classes}")

        # all TME tiles (includes tumor class)
        df_tme = df_merged[df_merged["predicted_class"].isin(tme_classes)]
        if df_tme.empty:
            raise ValueError("No TME tiles for the given classes found.")

        # tumor union → buffer
        tumor_polys = [make_patch_poly(r.x, r.y) for r in df_tumor.itertuples()]
        tumor_buffer = unary_union(tumor_polys).buffer(tme_margin)

        # TME tiles intersecting tumor buffer
        tme_tile_indices = []
        for r in df_tme.itertuples():
            if make_patch_poly(r.x, r.y).intersects(tumor_buffer):
                tme_tile_indices.append(r.tile_index)

        df_merged["in_tme_roi"] = df_merged["tile_index"].isin(tme_tile_indices)

    # -------- Save merged CSV (with in_tme_roi if computed) --------
    if save_merged:
        outdir.mkdir(parents=True, exist_ok=True)
        if merged_csv_name is None:
            merged_csv_name = f"{name}_annotations_with_coords.csv"
        merged_path = outdir / merged_csv_name
        df_merged.to_csv(merged_path, index=False)

    return df_merged
