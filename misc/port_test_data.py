"""
Port legacy test assets into portable, version-agnostic files.
"""

import pickle, shutil, time
from pathlib import Path

import pandas as pd
import rioxarray
import xarray as xr

# major parameters
INPUT_DIR = Path("/home/cefect/LS/10_IO/FloodDownscaler2/test_data")
OUTPUT_DIR = Path("tests/data/port")
NODATA_VALUE = -9999.0
COMPRESS = "LZW"
OVERWRITE = True


def main_port_test_data(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, overwrite=OVERWRITE):
    """Port legacy test data into GeoTIFF/csv copies with mirrored folders.

    Parameters
    ----------
    input_dir : pathlib.Path
        Root directory containing legacy test assets.
    output_dir : pathlib.Path
        Root directory for the ported dataset.
    overwrite : bool
        If True, remove the output directory before writing.
    """

    # input assertions
    assert isinstance(input_dir, Path), type(input_dir)
    assert isinstance(output_dir, Path), type(output_dir)
    assert input_dir.exists(), input_dir
    assert input_dir.is_dir(), input_dir
    assert isinstance(overwrite, bool), overwrite

    # clean output directory when requested
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # iterate all files and route by suffix
    t0 = time.time()
    meta_l = []
    for src_fp in sorted(input_dir.rglob("*")):
        if not src_fp.is_file():
            continue

        rel_fp = src_fp.relative_to(input_dir)
        dst_fp = output_dir / rel_fp
        ext = src_fp.suffix.lower()
        meta_d = {"source_fp": str(src_fp), "relative_fp": str(rel_fp), "suffix": ext}

        # copy plain raster and tabular files as-is
        if ext in {".tif", ".csv"}:
            dst_fp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_fp, dst_fp)
            meta_d.update({"action": "copied", "output_fp": str(dst_fp)})
            meta_l.append(meta_d)
            continue

        # convert pickle DataArrays to GeoTIFF
        if ext == ".pkl":
            with open(src_fp, "rb") as f:
                obj = pickle.load(f)

            if not isinstance(obj, xr.DataArray):
                meta_d.update(
                    {
                        "action": "skipped_non_raster_pickle",
                        "pickle_type": type(obj).__name__,
                        "output_fp": "",
                    }
                )
                meta_l.append(meta_d)
                continue

            # refresh DataArray to avoid stale serialized rio accessor caches
            crs = None
            try:
                crs = obj.rio.crs
            except Exception:
                crs = None

            da = xr.DataArray(obj.data, coords=obj.coords, dims=obj.dims, attrs=obj.attrs, name=obj.name)
            if crs is not None:
                da = da.rio.write_crs(crs)

            # write GeoTIFF with a single band and explicit nodata
            da = da.assign_coords(band=1).expand_dims(dim="band")
            da = da.fillna(NODATA_VALUE).rio.write_nodata(NODATA_VALUE)

            dst_tif_fp = dst_fp.with_suffix(".tif")
            dst_tif_fp.parent.mkdir(parents=True, exist_ok=True)
            da.rio.to_raster(dst_tif_fp, compute=True, compress=COMPRESS)

            meta_d.update({"action": "pickle_to_geotiff", "output_fp": str(dst_tif_fp)})
            meta_l.append(meta_d)
            continue

        # ignore all other file types
        meta_d.update({"action": "ignored", "output_fp": ""})
        meta_l.append(meta_d)

    # write summary table for audit/debug
    meta_df = pd.DataFrame(meta_l)
    meta_fp = output_dir / "_port_meta.csv"
    meta_df.to_csv(meta_fp, index=False)

    # print concise summary
    cnt = meta_df["action"].value_counts().to_dict()
    print(f"Port finished in {time.time() - t0:,.2f}s")
    print(f"Input:\n    {input_dir}")
    print(f"Output:\n    {output_dir}")
    print(f"Summary:\n    {cnt}")
    print(f"Meta table:\n    {meta_fp}")

    return output_dir, meta_fp, meta_df


if __name__ == "__main__":
    main_port_test_data()
