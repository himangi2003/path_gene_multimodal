import sys
sys.path.append("path_gene_multimodal")
from config import *
from tiling import *
from extract_embedding_from_tiles import *
from create_embedding import *
from find_annotation_from_embedding import *
from load_annotation_with_coordinates import *

#run tiling 
run_tessellation(wsi_path)
# Run deep feature extraction 
run_extract_features_for_tessellation(
    wsi_path,
    base_output_dir="outputs",
    model_type="VIRCHOW2",
    use_gpu=True,
    batch_size=128,
)

# find the labels for the patches

class_pt = run_create_class_embeddings(classes, wsi_path)

# 4) Annotate tiles (pixel coordinates, classes in a  CSV))
csv_path = run_annotation_for_extracted_features(
    wsi_path,
    class_embedding_pt_path=class_pt,
    classes=classes,
    base_output_dir="outputs",
)
df = load_annotations_with_coords(wsi_path,classes, base_output_dir="outputs")
print("Annotations_df:", df)

# Example: invasive tumor only
poly, mask_clean, s, (W, H) = mask_contour_from_tiles(
    df,
    wsi_path,
    256,
    tumor_labels,
    xy_is_top_left=True,
    mask_max_dim=6000,
    close_frac=0.35,
    open_frac=0.12,
    min_island_tiles=25,
    simplify_tol_px=3.0,
)

preview = overlay_polygon_on_wsi_thumbnail_tiffslide(
    wsi_path, poly, max_dim=2400
)
display(preview)


# ROI segmentation to get tumor shape, entropy analysis by reading the geojson file for each class
# work in progress
