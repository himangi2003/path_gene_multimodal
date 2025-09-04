
import os
import numpy as np
import pandas as pd


data_path = '/lab/deasylab3/Jung/tiger/'
dir_TIFF_images = data_path + "/wsirois/wsi-level-annotations/images/"


imgs_names = os.listdir(dir_TIFF_images)
imgs_names.sort()
imgs_names = [i for i in imgs_names if i.startswith('TCGA')]  
wsi_path = dir_TIFF_images + imgs_names[5]
