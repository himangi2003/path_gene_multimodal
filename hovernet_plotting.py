import pandas as pd
import numpy as np

def load_xy_tsv(tsv_path, x_col_guess="x", y_col_guess="y", name_col_guess="name"):
    """
    Reads a TSV with at least x and y columns (level-0 pixel coords).
    'name' is optional; if missing we infer from which file it came.
    Handles messy headers, extra spaces, and trailing commas in names.
    """
    # fast path for TSV; fall back to python engine if needed
    try:
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, engine="c")
    except Exception:
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, engine="python")

    # normalize headers
    canon = {c.lower().strip(): c for c in df.columns}
    xcol = canon.get(x_col_guess, x_col_guess)
    ycol = canon.get(y_col_guess, y_col_guess)
    ncol = canon.get(name_col_guess)  # may be None

    # numeric coords (coerce bad rows to NaN and drop)
    x = pd.to_numeric(df[xcol].str.strip(), errors="coerce")
    y = pd.to_numeric(df[ycol].str.strip(), errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid].to_numpy(float)
    y = y[valid].to_numpy(float)

    # clean names if present
    if ncol:
        names = (df.loc[valid, ncol]
                   .astype(str)
                   .str.strip()
                   .str.rstrip(",")  # handles "connective,"
                   .str.lower()
                   .to_numpy())
    else:
        names = None

    return np.c_[x, y], names



'''
Instance map: 2D full-size matrix where each pixels value corresponds to the associated instance (value>0) or background (value=0)
'''

# open: file-like interaction with zarr-array
instance_map = zarr.open( output_dir + "/pinst_pp.zip", mode="r")
# selecting a ROI will yield a numpy array
roi = instance_map[10000:20000,10000:20000]
# or with [:] to load the entire array
full_instance_map = instance_map[:]
# alternatively, use load, which will directly create a numpy array:
full_instance_map = zarr.load( output_dir + "/pinst_pp.zip") 

'''
Class dictionary: Lookup for the instance map, also contains centroid coordinates. If only centroid coordinates are of interest, you can skip loading the instance map.
'''

# load the dictionary
with open(output_dir + "class_inst.json","r") as f:
    class_info = json.load(f)
# create a centroid info array
centroid_array = np.array([[int(k),v[0],*v[1]] for k,v in class_info.items()])
# [instance_id, class_id, y, x]

# or alternatively create a lookup for the instance map to get a corresponding class map
pcls_list = np.array([0] + [v[0] for v in class_info.values()])
pcls_keys = np.array(["0"] + list(class_info.keys())).astype(int)
lookup = np.zeros(pcls_keys.max() + 1,dtype=np.uint8)
lookup[pcls_keys] = pcls_list
cls_map = lookup[full_instance_map]



# one TSV per class
TSV_BY_CLASS = {
    "connective":   output_dir + "/pred_connective.tsv",
    "dead":         output_dir + "/pred_dead.tsv",
    "epithelial":   output_dir + "/pred_epithelial.tsv",
    "inflammatory": output_dir + "/pred_inflammatory.tsv",
    "neoplastic":   output_dir + "/pred_neoplastic.tsv",
}

CLASS_ID = {
    "connective":   1,
    "dead":         2,
    "epithelial":   3,
    "inflammatory": 4,
    "neoplastic":   5,
}



all_xy, all_cls_name, all_cls_id = [], [], []

for cls_name, tsv_path in TSV_BY_CLASS.items():
    xy, names = load_xy_tsv(tsv_path)
    if xy.size == 0:
        continue
    all_xy.append(xy)
    all_cls_name.append(np.full(xy.shape[0], cls_name, dtype=object))
    all_cls_id.append(np.full(xy.shape[0], CLASS_ID[cls_name], dtype=np.uint8))

xy = np.vstack(all_xy) if all_xy else np.empty((0,2))
cls_names = np.concatenate(all_cls_name) if all_cls_name else np.empty((0,), dtype=object)
cls_ids   = np.concatenate(all_cls_id)   if all_cls_id   else np.empty((0,), dtype=np.uint8)

print(f"Loaded {len(xy)} points across {len(TSV_BY_CLASS)} TSV files.")


import matplotlib.pyplot as plt
import numpy as np

# xy: Nx2 float array ([:,0]=x, [:,1]=y)
# cls_ids: N-length uint8/int array of class IDs
# CLASS_ID: dict like {"connective":1, "dead":2, ...}

# Build a stable color + label mapping from your CLASS_ID
# (edit colors if you like)
CLASS_COLORS = {
    "connective":   "#8dd3c7",
    "dead":         "#ffffb3",
    "epithelial":   "#bebada",
    "inflammatory": "#fb8072",
    "neoplastic":   "#80b1d3",
}
# Inverse map id->name (assumes unique IDs)
ID_TO_NAME = {v: k for k, v in CLASS_ID.items()}

plt.figure(figsize=(8, 8))

# Plot each class separately for a crisp legend
for cid in sorted(np.unique(cls_ids)):
    name = ID_TO_NAME.get(int(cid), f"class_{int(cid)}")
    color = CLASS_COLORS.get(name, "gray")
    m = (cls_ids == cid)
    if np.any(m):
        plt.scatter(
            xy[m, 0], xy[m, 1],
            s=3, c=color, alpha=0.9, linewidths=0, label=name,
            rasterized=True  # helps with huge point counts in PDFs
        )

plt.title("Centroid Locations by Class")
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.gca().set_aspect("equal", adjustable="box")
plt.gca().invert_yaxis()  # matches slide coordinate orientation (0,0 at top-left)
plt.grid(True, alpha=0.25)
plt.legend(markerscale=3, frameon=True, loc="best")
plt.tight_layout()
plt.show()