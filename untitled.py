from pathlib import Path
tumor_classes = [
     "Invasive tumor epithelium (TNBC) or In situ carcinoma (DCIS / LCIS)",
]

til_classes = [
    "Lymphocyte-rich stroma / TILs",
]

tls_classes = [
    "Lymphoid aggregate / TLS",
]

data_path = Path(data_path)
out_dir = Path(out_dir)

image_files = sorted(data_path.iterdir())

wsi_path = image_files[21]

slide_id = wsi_path.stem
slide_out_dir = out_dir / slide_id
slide_out_dir.mkdir(parents=True, exist_ok=True)
slide_out_dir = out_dir / name
geojson_path = slide_out_dir / f"{name}.geojson"

print("WSI path:", wsi_path)
print("GeoJSON path:", geojson_path)
csv_path = slide_out_dir / f"{slide_id}_islands.csv"

df = process_one_slide_make_csv_and_plot(
    slide_id=slide_id,
    wsi_path=wsi_path,
    geojson_path=geojson_path,
    tumor_classes=tumor_classes,
    til_classes=til_classes,
    tls_classes=tls_classes,   # <-- separate TLS
    out_csv_path=csv_path,
    thumb_size=(1024, 1024),   # faster for batch
    do_plot=True,
)

from datetime import datetime

def write_basic_size_burden_metrics_txt(
    df_islands,
    slide_id,
    out_txt_path,
):
    """
    Appends BASIC SIZE & BURDEN METRICS to a per-slide TXT file.
    Safe to call multiple times as you add more metric blocks later.
    """

    # ---- compute metrics ----
    tissue_area = float(df_islands["tissue_area_px2"].iloc[0])

    def sum_area(typ):
        sub = df_islands[df_islands["type"] == typ]
        return float(sub["area_px2"].sum()) if not sub.empty else 0.0

    tumor_area = sum_area("tumor")
    til_area   = sum_area("til")
    tls_area   = sum_area("tls")
    immune_area = til_area + tls_area

    tumor_frac  = tumor_area / tissue_area if tissue_area > 0 else None
    til_frac    = til_area / tissue_area if tissue_area > 0 else None
    tls_frac    = tls_area / tissue_area if tissue_area > 0 else None
    immune_frac = immune_area / tissue_area if tissue_area > 0 else None

    denom = tumor_area + immune_area
    immune_dom = immune_area / denom if denom > 0 else None

    # ---- write block ----
    with open(out_txt_path, "a") as f:
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("I. BASIC SIZE & BURDEN METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Slide ID: {slide_id}\n")
        f.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write(f"Tissue area (px^2):        {tissue_area:.3e}\n")
        f.write(f"Tumor area (px^2):         {tumor_area:.3e}\n")
        f.write(f"TIL area (px^2):           {til_area:.3e}\n")
        f.write(f"TLS area (px^2):           {tls_area:.3e}\n")
        f.write(f"Immune area (px^2):        {immune_area:.3e}\n\n")

        f.write(f"Tumor / tissue fraction:   {tumor_frac:.4f}\n" if tumor_frac is not None else "")
        f.write(f"TIL / tissue fraction:     {til_frac:.4f}\n" if til_frac is not None else "")
        f.write(f"TLS / tissue fraction:     {tls_frac:.4f}\n" if tls_frac is not None else "")
        f.write(f"Immune / tissue fraction:  {immune_frac:.4f}\n" if immune_frac is not None else "")
        f.write("\n")

        f.write(
            f"Immune dominance index\n"
            f"(immune / (tumor + immune)): {immune_dom:.4f}\n"
            if immune_dom is not None else
            "Immune dominance index: NA\n"
        )

        f.write("\n")

# ---- WRITE TXT METRICS ----
metrics_txt_path = slide_out_dir / f"{slide_id}_metrics.txt"

write_basic_size_burden_metrics_txt(
    df_islands=df,
    slide_id=slide_id,
    out_txt_path=metrics_txt_path,
)
