import os
import numpy as np
import pandas as pd
import zarr
import numpy as np
import json
import os
import argparse
import sys
from timeit import default_timer as timer
from datetime import timedelta
import torch
from glob import glob
from inference.inference import inference_main, get_inference_setup
from inference.post_process import post_process_main
from inference.data_utils import copy_img

torch.backends.cudnn.benchmark = True
print(torch.cuda.device_count(), " cuda devices")


def prepare_input(params):
    """
    Check if input is a text file, glob pattern, or a directory, and return a list of input files

    Parameters
    ----------
    params: dict
        input parameters from argparse

    Returns
    -------
    list
        List of input file paths

    Raises
    ------
    FileNotFoundError
        If the input file or pattern does not exist
    ValueError
        If no files match the pattern
    """
    print("Input specified:", params["input"])
    
    if params["input"].endswith(".txt"):
        if os.path.exists(params["input"]):
            with open(params["input"], "r") as f:
                input_list = [line.strip() for line in f if line.strip()]
            if not input_list:
                raise ValueError(f"Text file {params['input']} is empty or contains no valid paths")
        else:
            raise FileNotFoundError(f"Input text file not found: {params['input']}")
    else:
        input_list = sorted(glob(params["input"].rstrip()))
        if not input_list:
            raise ValueError(f"No files found matching pattern: {params['input']}")
    
    print(f"Found {len(input_list)} file(s) to process")
    return input_list


def get_input_type(params):
    """
    Check if input is an image, numpy array, or whole slide image, and return the input type
    If you are trying to process other images that are supported by opencv (e.g. tiff), you can add the extension to the list

    Parameters
    ----------
    params: dict
        input parameters from argparse
    """
    params["ext"] = os.path.splitext(params["p"])[-1]
    if params["ext"] == ".npy":
        params["input_type"] = "npy"
    elif params["ext"] in [".jpg", ".png", ".jpeg", ".bmp"]:
        params["input_type"] = "img"
    else:
        params["input_type"] = "wsi"
    return params


def infer(params: dict):
    """
    Start nuclei segmentation and classification pipeline using specified parameters from argparse

    Parameters
    ----------
    params: dict
        input parameters from argparse
    
    Raises
    ------
    ValueError
        If required parameters are invalid
    """
    
    # Validate checkpoint parameter
    if not params["cp"]:
        raise ValueError("Checkpoint parameter (--cp) is required. Please specify a model checkpoint.")
    
    # Validate metric
    if params["metric"] not in ["mpq", "f1", "pannuke"]:
        print(f"Warning: Invalid metric '{params['metric']}', falling back to 'f1'")
        params["metric"] = "f1"
    else:
        print(f"Optimizing postprocessing for: {params['metric']}")

    params["data_dirs"] = params["cp"].split(",")
    
    # Create output directory if it doesn't exist
    os.makedirs(params["output_dir"], exist_ok=True)
    print(f"Results will be saved to: {params['output_dir']}")
    print(f"Loading model from: {params['data_dirs']}")

    # Run per tile inference and store results
    params, models, augmenter, color_aug_fn = get_inference_setup(params)

    input_list = prepare_input(params)
    print("Running inference on", len(input_list), "file(s)")

    for inp in input_list:
        start_time = timer()
        params["p"] = inp.rstrip()
        params = get_input_type(params)
        print("Processing ", params["p"])
        if params["cache"] is not None:
            print("Caching input at:")
            params["p"] = copy_img(params["p"], params["cache"])
            print(params["p"])

        params, z = inference_main(params, models, augmenter, color_aug_fn)
        print(
            "::: finished or skipped inference after",
            timedelta(seconds=timer() - start_time),
        )
        process_timer = timer()
        if params["only_inference"]:
            try:
                z[0].store.close()
                z[1].store.close()
            except TypeError:
                # if z is None, z cannot be indexed -> throws a TypeError
                pass
            print("Exiting after inference")
            sys.exit(2)
        # Stitch tiles together and postprocess to get instance segmentation
        if not os.path.exists(os.path.join(params["output_dir"], "pinst_pp.zip")):
            print("running post-processing")

            z_pp = post_process_main(
                params,
                z,
            )
            if not params["keep_raw"]:
                try:
                    os.remove(params["model_out_p"] + "_inst.zip")
                    os.remove(params["model_out_p"] + "_cls.zip")
                except FileNotFoundError:
                    pass
        else:
            z_pp = None
        print(
            "::: postprocessing took",
            timedelta(seconds=timer() - process_timer),
            "total elapsed time",
            timedelta(seconds=timer() - start_time),
        )
        if z_pp is not None:
            z_pp.store.close()
    print("done")


data_path = '/lab/deasylab3/Jung/Data/Shared_data/TCGA_TNBC/Histology/'
dir_TIFF_images_low_risk = data_path + "TCGA-EW-A6SB-01Z-00-DX1.D56E1922-01A9-4AEE-AB95-D69447DD13EE.tif"
dir_TIFF_images_high_risk = data_path + "TCGA-A2-A0D0-01Z-00-DX1.4FF6B8E5-703B-400F-920A-104F56E0F874.tif"


wsi_path = dir_TIFF_images_high_risk
output_dir = "/cluster/home/srivash/venvs/Mussel/hovernet_outputs6"


os.chdir("hover_next_inference")
import sys
sys.path.append("src")



params = {
    "input": wsi_path,        # your WSI
    "output_dir": output_dir,      # output folder
    "cp": "pannuke_convnextv2_tiny_3",            # one or more checkpoints
    "metric": "f1",
    "batch_size": 32,
    "tta": 4,
    "save_polygon": True,
    "tile_size": 256,
    "overlap": 0.96875,
    "inf_workers": 4,
    "inf_writers": 2,
    "pp_tiling": 8,
    "pp_overlap": 256,
    "pp_workers": 16,
    "keep_raw": False,
    "cache": "/tmp",                               # optional cache
    "only_inference": False
}


infer(params)




