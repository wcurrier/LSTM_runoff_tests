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
from threading import Lock

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


def estimate_group_memory(samples):
    """Estimate memory use of a list of dict samples (roughly)."""
    total = 0
    for s in samples:
        try:
            total += sys.getsizeof(s)
            for k, v in s.items():
                total += sys.getsizeof(k)
                total += sys.getsizeof(v)
        except Exception:
            pass
    return total / 1024**2  # in MB

def task_done_callback(future):
    try:
        # This will raise if the task failed
        _ = future.result()
    except Exception as e:
        logger.error(f"Task {future.key} failed with exception: {e}", exc_info=True)
    else:
        logger.debug(f"Task {future.key} completed successfully.")

# ---------------------------- Main driver -----------------------------
def main():


    BATCH_SIZE = 50 # This is the number of Dask tasks (i.e., futures) you submit before pausing and waiting for them to finish.
    SAVE_INTERVAL = 50 # This determines how often you write results to disk (as NetCDF) while processing the completed futures.

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
    summary = []

    try:
        grouped = defaultdict(list)
        counter = 0
        start_time = time.time()
        lock = Lock()  # for thread-safe counter + grouped access in callback

        # Callback function to handle the result of each Dask Future after it finishes
        def handle_result(future):
            nonlocal counter, grouped # Declare that we are modifying these outer-scope variables
            try:
                result = future.result() # Try to retrieve the result from the completed Future
                if result is None:
                    logger.warning("Future returned None.")
                    return
    
                with lock: # Acquire the lock to safely modify shared state (grouped results and counter)
                    grouped[(result["basin_id"], result["split"])].append(result) # Group the result using (basin_id, split) as the key for later saving
                    counter += 1

                    # Every SAVE_INTERVAL results, flush results to disk to avoid memory bloat
                    if counter % SAVE_INTERVAL == 0:
                        logger.info(f"[{counter}] Saving intermediate results after {time.time() - start_time:.1f}s")
                        for (gauge, split), samples in grouped.items():
                            n = len(samples)
                            mem_mb = estimate_group_memory(samples)
                            logger.info(f"[{split}][{gauge}] Final save: {n} samples (~{mem_mb:.2f} MB)")
                            
                            ds = samples_to_xarray(samples)
                            out = SCRATCH / f"{split}_{gauge}_{counter}.nc"

                            if ds.dynamic_inputs.isnull().any() or ds.targets.isnull().any():
                                logger.warning(f"[{split}][{gauge}] contains NaNs — skipping {out.name}")
                                continue
    
                            ds.to_netcdf(out)
                            logger.info(f"Wrote {out}")
    
                        grouped.clear()
    
            except Exception as e:
                # If something goes wrong during result processing, log the full traceback
                logger.error(f"Task failed in callback: {e}", exc_info=True)
    
        logger.info("Submitting tasks...")
        # Loop over all gauge IDs for which streamflow data is available
        for gauge_id in list(streamflows.keys()):
            dfQ = streamflows[gauge_id] # Get streamflow data for the current gauge
            for split, dates in FORECAST_BLOCKS.items():
                for d in dates:
                    # Submit the build_sample_wrapped function as a Dask task (Future)
                    future = client.submit(
                        build_sample_wrapped, gauge_id, split, d, dfQ, ds_era5, ds_hres, pure=False, retries=1
                    )
                    # Register the callback to process the result when this task completes
                    future.add_done_callback(handle_result)
                    
    
        ###########################################################
        ######## This is the “final cleanup write” stage. #########
        ###########################################################
        # ---- Final save ----
        logger.info("Final flush of remaining results...")

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


if __name__ == "__main__":
    main()
    
# # Cleanup scratch files

# In[4]:


# # Optional cleanup
# for f in SCRATCH.glob("*.nc"):
#     f.unlink()  # or f.rename(SCRATCH / "archive" / f.name)

