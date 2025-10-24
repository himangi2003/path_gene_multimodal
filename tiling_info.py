def read_tiles(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError("No 'coords' dataset found. Run `describe_h5` to see available keys.")

        ds = f["coords"]
        arr = ds[()]  # numpy array

        # figure out column names
        cols = None
        if hasattr(ds, "attrs") and b"columns" in ds.attrs:
            # sometimes writers store an explicit column list
            raw_cols = ds.attrs[b"columns"]
            cols = [c.decode() if isinstance(c, bytes) else str(c) for c in raw_cols]

        if cols is None:
            # sensible defaults based on width
            if arr.ndim == 1:
                arr = arr.reshape(-1, 2)  # fallback
            if arr.shape[1] == 2:
                cols = ["x", "y"]
            elif arr.shape[1] == 3:
                cols = ["x", "y", "level"]
            elif arr.shape[1] == 4:
                cols = ["x", "y", "w", "h"]
            else:
                cols = [f"col{i}" for i in range(arr.shape[1])]

        df = pd.DataFrame(arr, columns=cols)

        # try to pull useful attrs from dataset or file
        attrs = {}
        for src in (f, ds):
            for k, v in getattr(src, "attrs", {}).items():
                k = k.decode() if isinstance(k, bytes) else k
                attrs[k] = v.decode() if isinstance(v, bytes) else v

        # if width/height missing but tile_size exists, create w/h
        tile_size = attrs.get("tile_size") or attrs.get("patch_size") or attrs.get("size")
        if tile_size is not None and "w" not in df.columns and "h" not in df.columns:
            df["w"] = int(tile_size)
            df["h"] = int(tile_size)

        # add derived boxes (x1,y1,x2,y2) when w/h known
        if {"x","y","w","h"}.issubset(df.columns):
            df["x1"] = df["x"]
            df["y1"] = df["y"]
            df["x2"] = df["x"] + df["w"]
            df["y2"] = df["y"] + df["h"]

        # tack on global metadata as constant columns (handy downstream)
        for k in ("level", "mpp", "stride", "downsample", "slide_id"):
            if k in attrs and k not in df.columns:
                df[k] = attrs[k]

        return df, attrs

df, attrs = read_tiles(h5_path)
print(df.head())
print("\nMetadata:", attrs)

# Save for downstream steps
out_csv = Path(h5_path).with_suffix(".tiles.csv")
df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)