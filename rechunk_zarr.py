#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xarray as xr
import zarr
import rechunker
from pathlib import Path
import argparse

def rechunk_zarr(input_dir, output_dir, tmp_dir, target_chunks, max_mem="64GB"):
    print(f"🔍 Opening: {input_dir}")
    ds = xr.open_zarr(input_dir, consolidated=True, chunks={})
    print(f"📐 Dimensions: {ds.dims}")

    print(f"🧩 Target chunks: {target_chunks}")
    print(f"📦 Rechunking to: {output_dir}")

    plan = rechunker.rechunk(
        ds,
        target_chunks=target_chunks,
        max_mem=max_mem,
        target_store=zarr.DirectoryStore(output_dir),
        temp_store=zarr.DirectoryStore(tmp_dir),
        executor="python",
    )
    plan.execute()
    print("✅ Rechunking complete!")


def main():
    parser = argparse.ArgumentParser(description="Rechunk HRES or ERA5 Zarr dataset.")
    parser.add_argument("--dataset", type=str, required=True, choices=["HRES", "ERA5"],
                        help="Which dataset to rechunk: 'HRES' or 'ERA5'")
    args = parser.parse_args()

    base = Path("/Projects/HydroMet/currierw")

    if args.dataset == "HRES":
        in_path  = base / "HRES/timeseries.zarr"
        tmp_path = base / "HRES/HRES_rechunk_tmp.zarr"
        out_path = base / "HRES/timeseries_rechunked.zarr"
        target_chunks = {"basin": 1, "date": 128, "lead_time": 10}
    else:  # ERA5
        in_path  = base / "ERA5_LAND/timeseries.zarr"
        tmp_path = base / "ERA5_LAND/rechunk_tmp.zarr"
        out_path = base / "ERA5_LAND/timeseries_rechunked.zarr"
        target_chunks = {"basin": 1, "date": 128}

    rechunk_zarr(in_path, out_path, tmp_path, target_chunks)


if __name__ == "__main__":
    main()

