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
