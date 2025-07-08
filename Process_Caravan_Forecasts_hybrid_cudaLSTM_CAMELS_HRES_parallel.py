#!/usr/bin/env python
# coding: utf-8

# preprocess_LSTM_dask.py
# -----------------------------------------------------------
# ❶  Imports & Dask cluster
# -----------------------------------------------------------
import pandas as pd, numpy as np, xarray as xr, geopandas as gpd
import traceback, sys, dask, pickle, torch, zarr, logging
import multiprocessing, psutil, os, tempfile, json, shutil, time, urllib.error
from dask import delayed, compute
from dask.distributed import LocalCluster, Client, performance_report, wait, as_completed
from pathlib import Path
from functools import lru_cache
from collections import Counter, defaultdict
from logging.handlers import RotatingFileHandler


from data_access import era5, hres
from utils import build_sample, samples_to_xarray, build_sample_wrapped, get_usgs_streamflow, get_or_download_streamflows
# from config import FORECAST_BLOCKS, SCRATCH

# ------------------------------------------------------------------
# paths & constants -------------------------------------------------
BASE_OBS  = Path('/Projects/HydroMet/currierw/ERA5_LAND')
BASE_FCST = Path('/Projects/HydroMet/currierw/HRES')
SCRATCH   = Path('/Projects/HydroMet/currierw/HRES_processed_tmp')
FINAL_OUT = Path('/Projects/HydroMet/currierw/HRES_processed')
STREAMFLOW_PATH = Path('/Projects/HydroMet/currierw/HRES_processed')

# 2016       - 2020-09-30, n = 347 w/ 5 day intervals
# 2016-10-01 - 2022-09-30, n = 146 w/ 5 day intervals
# 2022-10-01 - 2024-09-30, n = 147 w/ 5 day intervals
#                              640
# 587 gauges = 375,680 samples each 106 long w/ 5 variables (Precip, Temp, NSRad, Flag, Streamflow)
# Test it on 50 gauges and 10 dates over 3 splits (train/val/test) = 1500 ~.3%
FORECAST_BLOCKS = {
    "train":      pd.date_range('2016-01-01', '2020-09-30', freq='5D'),
    "validation": pd.date_range('2020-10-01', '2022-09-30', freq='5D'),
    "test":       pd.date_range('2022-10-01', '2024-09-30', freq='5D'),
}
EXPECTED_LEN = 106

# Create a handler that rotates logs at 5MB, keeping 3 backups
rotating_handler = RotatingFileHandler(
    "preprocess_LSTM_dask.log",  # Base log file
    maxBytes=5_000_000,          # 5 MB max size per file
    backupCount=3                # Keep up to 3 backup logs
)
# Example log output files:
# preprocess_LSTM_dask.log        <- current file
# preprocess_LSTM_dask.log.1      <- previous
# preprocess_LSTM_dask.log.2      <- older
# preprocess_LSTM_dask.log.3      <- oldest (will be deleted next)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        rotating_handler
    ]
)
logger = logging.getLogger(__name__)

def task_done_callback(future):
    try:
        # This will raise if the task failed
        _ = future.result()
    except Exception as e:
        logger.error(f"Task {future.key} failed with exception: {e}", exc_info=True)
    else:
        logger.debug(f"Task {future.key} completed successfully.")

# ---------------------------- Main driver -----------------------------
if __name__ == "__main__":


    BATCH_SIZE = 500 # This is the number of Dask tasks (i.e., futures) you submit before pausing and waiting for them to finish.
    SAVE_INTERVAL = 500 # This determines how often you write results to disk (as NetCDF) while processing the completed futures.

    # ---- Clear scratch output folder ----
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    # ---- Load gauges ----
    gdf = gpd.read_file('/Projects/HydroMet/currierw/Caravan-Jan25-csv/shapefiles/camels/camels_basin_shapes.shp')
    gauge_ids = gdf["gauge_id"].str.split("_").str[-1].tolist()
    streamflows, skipped = get_or_download_streamflows(gdf, STREAMFLOW_PATH)

    print(f"Loaded {len(streamflows)} gauges. Skipped {len(skipped)}.")

    # ---- Dask cluster setup ----
    module_path = '/home/wcurrier/LSTM_runoff_tests'

    def add_module_path(dask_worker):
        if module_path not in sys.path:
            sys.path.insert(0, module_path)

    cluster = LocalCluster(n_workers=16, threads_per_worker=2, memory_limit="8GB")
    # cluster.adapt(minimum=8, maximum=32)
    client = Client(cluster)
    client.run(add_module_path)
    logger.info("Module path added to workers")

    # ---- Task submission in batches ----

    ds_era5 = era5()
    ds_hres = hres()
    try:
        all_futures = [] # future object is something that will be done in the background. 
                         # We don't have the result yet but we've told Dask to compute this 
                         # with client.submit(...)
        pending = []     # This is just a Python list where you collect all the Future objects 
                         # before submitting them in a batch.
    
        # for gauge_id in list(streamflows.keys())[:50]: # loop over gauges
        for gauge_id in list(streamflows.keys()):
            dfQ = streamflows[gauge_id]
            for split, dates in FORECAST_BLOCKS.items(): # loop over forecast dates and train/val/test
                # for d in dates[:10]: # only a subset of dates for each split
                for d in dates: # only a subset of dates for each split
                    # We submitted 10 forecast dates per gauge-split pair and thus generated 10 results that will be grouped in output unless > SAVE_INTERVAL
                    future = client.submit(
                        build_sample_wrapped, gauge_id, split, d, dfQ, ds_era5, ds_hres, pure=False
                    )
                    future.add_done_callback(task_done_callback)
                    # build_sample_wrapped eventually called build_sample which creates the data we're interested in
                    pending.append(future)
    
                    if len(pending) >= BATCH_SIZE:
                        logger.debug(f"Submitting batch of {len(pending)} tasks...")
                        wait(pending, timeout=None)
                        all_futures.extend(pending) # a complete list of all futures we've launched so far
                        pending = []
    
        # Final pending batch
        if pending:
            logger.debug(f"Submitting final batch of {len(pending)} tasks...")
            wait(pending, timeout=None)
            all_futures.extend(pending)
            
        ###########################################################
        ##### This is the “intermediate write-to-disk” stage. #####
        ###########################################################
    
        # ---- Collect results as they complete ----
        logger.info("Waiting for results and saving intermediate files...")
        start_time = time.time()
        grouped = defaultdict(list)
        counter = 0 # a global count of how many results you've collected from as_completed(...).
                    # - Drives SAVE_INTERVAL: if counter % SAVE_INTERVAL == 0 → triggers disk write.
                    # - Provides a unique suffix to output filenames so you don’t overwrite earlier files.
    
        # This loop will:
        # - Wait for each Future to finish.
        # - Collect the result (the dictionary).
        # - Group them for writing out to disk (as NetCDF).
        for future, result in as_completed(all_futures, with_results=True):
            if result is not None:
                grouped[(result["basin_id"], result["split"])].append(result)
            else:
                logger.error(f"Future {future.key} returned None or failed.")
    
            counter += 1
    
            if counter % SAVE_INTERVAL == 0 and counter > 0:
                # After we collect and group results from as_completed(all_futures, with_results=True):
                # Every SAVE_INTERVAL samples, flush everything to disk and clear memory.
                # - Avoids memory bloat: You don’t hold all samples in RAM
                # - Protects against crashes: If your run dies mid-process, you’ve already saved partial results
                # - Reduces I/O contention: Writing files constantly (e.g., every sample) can slow things down
                
                logger.info(f"[{counter}] Saving intermediate results at {time.time() - start_time:.1f}s")
                for (gauge, split), samples in grouped.items():
                    ds = samples_to_xarray(samples)
                    out = SCRATCH / f"{split}_{gauge}_{counter}.nc"
    
                    if ds.dynamic_inputs.isnull().any() or ds.targets.isnull().any():
                        logger.warning(f"[{split}][{gauge}] contains NaNs — skipping {out.name}")
                        continue
    
                    ds.to_netcdf(out)                
                    # print("Wrote", out)
    
                grouped.clear()
    
        ###########################################################
        ######## This is the “final cleanup write” stage. #########
        ###########################################################
        # ---- Final save ----
        logger.info("Final save of remaining results...")
        summary = []
    
        for (gauge, split), samples in grouped.items():
            ds = samples_to_xarray(samples)
            out = SCRATCH / f"{split}_{gauge}.nc"
    
            if ds.dynamic_inputs.isnull().any() or ds.targets.isnull().any():
                logger.warning(f"[{split}][{gauge}] contains NaNs — skipping final save")
                continue
    
            ds.to_netcdf(out)
            logger.info(f"Wrote {out}")
    
            summary.append({
                "gauge_id": gauge,
                "split": split,
                "n_samples": len(samples),
                "filename": out.name
            })

    except Exception as e:
        logger.error("Error during processing", exc_info=True)
    finally:
        logger.info("Closing client and cluster")
        client.close()
        cluster.close()
    
    # ---- Summary output ----
    pd.DataFrame(summary).to_csv("processed_summary.csv", index=False)
    logger.info(f"Done. Total time: {(time.time() - start_time)/60:.1f} minutes")
    logger.info(f"Results saved to {SCRATCH}")


    # # Concatenate indivudal files from parallel output
    SCRATCH = Path('/Projects/HydroMet/currierw/HRES_processed_tmp')
    FINAL_OUT = Path('/Projects/HydroMet/currierw/HRES_processed')
    # Step 1: List of splits
    splits = ["train", "validation", "test"]
    
    for split in splits:
        files = sorted(SCRATCH.glob(f"{split}_*.nc"))
        valid_datasets = []
    
        print(f"\n🧪 Checking {len(files)} files for '{split}' split...")
    
        for f in files:
            ds = xr.open_dataset(f)
    
            # Check for NaNs
            has_nan = (
                np.isnan(ds["dynamic_inputs"]).any() or
                np.isnan(ds["targets"]).any()
            )
    
            if has_nan:
                print(f"⚠️ Skipping file with NaNs: {f.name}")
                continue
    
            valid_datasets.append(ds)
    
        if not valid_datasets:
            print(f"❌ No valid files found for split: {split}")
            continue
    
        print(f"✅ {len(valid_datasets)} valid files found for '{split}'")
    
        # Step 2: Concatenate along the sample dimension
        combined = xr.concat(valid_datasets, dim="sample")
    
        # Step 3: Write out combined file
        output_path = FINAL_OUT / f"{split}.nc"
        combined.to_netcdf(output_path)
        print(f"📦 Wrote combined dataset for '{split}' → {output_path}")


# # Cleanup scratch files

# In[4]:


# # Optional cleanup
# for f in SCRATCH.glob("*.nc"):
#     f.unlink()  # or f.rename(SCRATCH / "archive" / f.name)

