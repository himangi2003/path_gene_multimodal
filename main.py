from config import *
#run tiling 
run_tessellation(wsi_path)
# Run on GPU (default)
run_extract_features_for_tessellation(
    wsi_path,
    base_output_dir="outputs",
    model_type="Virchow2",
    use_gpu=True,
    batch_size=128,
)
annotation_classes = [
    "carcinoma in situ",
    "invasive carcinoma",
    "collagenous stroma",
    "adipose",
    "vessel",
    "necrosis",
    "invasive adenocarcinoma",
    "sarcoma",
]

csv_path = run_annotation_for_extracted_features(
    wsi_path,
    class_embedding_pt_path="outputs/class_embeddings.pt",
    classes=annotation_classes,
    base_output_dir="outputs",
)
print("Annotations saved at:", csv_path)
