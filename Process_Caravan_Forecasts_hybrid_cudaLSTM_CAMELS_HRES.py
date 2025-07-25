#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
import torch
import zarr
import xarray as xr
import pandas as pd
import numpy as np
import math
from uuid import uuid4
import os, multiprocessing, psutil
from pathlib import Path
from collections import Counter
import glob
import re
import gc
import time
# -----------------------------------
# Setup
# -----------------------------------
# ERA5 = 1950-01-01 - 2024-10-31
# HRES = 2016-01-01 - 2024-09-30
# Train on 2016-01-01 - 2020-09-30, (4 years)
# Validate on 2020-10-01 - 2022-09-30 (2 years)
# Test on 2022-10-01 - 2024-09-30 (2 years)

# 2 meter temperature
# total precipitation
# net solar radiation

"""|<--- 294d (weekly) --->|<-- 60d (daily) -->|<--- 10d (forecast) -->|
       Weekly ERA5           Daily ERA5            HREF Forecast
        (n=42)                (n=60)                  (n=10)
        """


BATCH_SIZE = 200
intermediate_store = []
expected_precip_shape = None       
EXPECTED_LEN = 106
output_dir="/Projects/HydroMet/currierw/HRES_processed/"

# Load in CAMELS Data
df=gpd.read_file('/Projects/HydroMet/currierw/Caravan-Jan25-csv/shapefiles/camels/camels_basin_shapes.shp')
gaugeIDs=[]
for i in range(0,len(df)):
    gaugeIDs.append(df['gauge_id'][i].split('_')[-1])

# Load all pre-saved streamflow data
print("📥 Loading pre-saved streamflow data...")
streamflow_df_dict = pd.read_pickle(output_dir+'streamflows.pkl')

# Load in meterological observations and forecasts
base_obs = '/Projects/HydroMet/currierw/ERA5_LAND/'
base_fcst = '/Projects/HydroMet/currierw/HRES/'
ds_obs_all = xr.open_zarr(base_obs + 'camels_rechunked.zarr').sel(date=slice('2015-01-01', '2024-09-30')).load()
ds_fcst_all   = xr.open_zarr(base_fcst + 'camels_rechunked.zarr',decode_timedelta=True).load()
print('loaded in forecast and obs to memory')

# Set up training validation and test period
forecast_blocks = {
    "train": pd.date_range('2016-01-01', '2020-09-30', freq='5D'),
    "validation": pd.date_range('2020-10-01', '2022-09-30', freq='5D'),
    "test": pd.date_range('2022-10-01', '2024-09-30', freq='5D'),
}

# -----------------------------------
# Functions
# -----------------------------------
def get_last_processed_entry(split, output_dir=output_dir):
    pattern = os.path.join(output_dir, f"{split}_data_batch*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None  # Nothing processed yet
    latest = files[-1]
    try:
        ds = xr.open_dataset(latest)
        basin_ids = ds["basin_id"].values
        forecast_dates = ds["forecast_date"].values
        if len(basin_ids) > 0:
            return basin_ids[-1], forecast_dates[-1]  # last one in the file
        else:
            return None, None
    except Exception as e:
        print(f"⚠️ Failed to read last batch: {latest}: {e}")
        return None, None
    
def get_last_batch_number(split, output_dir=output_dir):
    pattern = os.path.join(output_dir, f"{split}_data_batch*.nc")
    files = glob.glob(pattern)
    if not files:
        return 0
    batch_nums = []
    for f in files:
        match = re.search(rf"{split}_data_batch(\d+)\.nc", os.path.basename(f))
        if match:
            batch_nums.append(int(match.group(1)))
    return max(batch_nums) + 1  # Resume at next batch number

def to_xarray_dataset(samples, EXPECTED_LEN, split=None, standardize=False):
    # Step 1: Count all time lengths (based on 'precip')
    lengths = [s['precip'].shape[0] for s in samples]
    length_counts = Counter(lengths)

    REQUIRED_KEYS = ['precip', 'temp', 'net_solar', 'target', 'flag']

    # Optional: Print samples with mismatched array lengths
    for s in samples:
        lengths = {k: s[k].shape[0] for k in REQUIRED_KEYS}
        # if len(set(lengths.values())) > 1:
            # print(f"⚠️ Mismatched lengths for sample {s['forecast_date']} / {s['basin_id']}: {lengths}")

    # Step 3: Keep only samples where ALL arrays match EXPECTED_LEN
    clean_samples = [
        s for s in samples
        if all(s[k].shape[0] == EXPECTED_LEN for k in REQUIRED_KEYS)
    ]

    dropped = len(samples) - len(clean_samples)
    if dropped > 0:
        print(f"⚠️ [{split}] Dropped {dropped} of {len(samples)} samples due to unexpected time length.")

    if not clean_samples:
        raise ValueError("No valid samples with consistent length")

    n_samples = len(clean_samples)
    n_time = EXPECTED_LEN
    dyn_inputs = np.zeros((n_samples, n_time, 4), dtype=np.float32)
    targets = np.zeros((n_samples, n_time, 1), dtype=np.float32)
    basin_ids = np.empty(n_samples, dtype='U20')
    forecast_dates = np.empty(n_samples, dtype='U20')

    for i, s in enumerate(clean_samples):
        p = s['precip']
        t2 = s['temp']
        ns = s['net_solar']
        f = s['flag']
        t = s['target']

        dyn_inputs[i, :, 0] = p
        dyn_inputs[i, :, 1] = t2
        dyn_inputs[i, :, 2] = ns
        dyn_inputs[i, :, 3] = f
        targets[i, :, 0] = t
        basin_ids[i] = s['basin_id']
        forecast_dates[i] = s['forecast_date']

    return xr.Dataset(
        {
            "dynamic_inputs": (
                ["sample", "time", "feature"],
                dyn_inputs,
                {"feature": ["precip", "temp", "net_solar","flag"]}
            ),
            "targets": (
                ["sample", "time", "target"],
                targets,
                {"target": ["streamflow"]}
            ),
            "basin_id": (["sample"], basin_ids),
            "forecast_date": (["sample"], forecast_dates)
        }
    )
    
def write_batch(split, batch, count, EXPECTED_LEN):
    ds = to_xarray_dataset(batch, EXPECTED_LEN, split=split, standardize=False)
    fname = f"{split}_data_batch{count:05d}.nc"
    path = f"/Projects/HydroMet/currierw/HRES_processed/{fname}"
    ds.to_netcdf(path)
    print(f"✅ Saved batch: {fname}")

# -----------------------------------
# Main Loop
# -----------------------------------
start = time.time()

for split, forecast_dates in forecast_blocks.items():

    last_gauge, last_date = get_last_processed_entry(split)
    resume_flag = False if last_gauge else True

    count = get_last_batch_number(split)
    
    print(f"🔁 Resume is {resume_flag} last gauge is '{last_gauge} last date is '{last_date}'")
    print(f"🔁 Resuming from batch {count:05d} for split '{split}'")
    intermediate_store.clear()

    for gaugeID in gaugeIDs:

        if not resume_flag:
            if gaugeID < last_gauge:
                print(f"⏩ Skipping gauge {gaugeID}")
                continue
            elif gaugeID == last_gauge:
                resume_dates = forecast_dates[forecast_dates > pd.Timestamp(str(last_date))]
            else:
                resume_flag = True
                resume_dates = forecast_dates
        else:
            resume_dates = forecast_dates

        print(f"🔍 Processing Gauge: {gaugeID} , {split}")

        try:
            try:
                dfQ = streamflow_df_dict[gaugeID]
            except KeyError:
                print(f"⚠️ Missing streamflow for gauge {gaugeID}, skipping.")
                continue

            ds_obs = ds_obs_all.sel(basin=f'camels_{gaugeID}')
            ds_obs_p = ds_obs['era5land_total_precipitation']
            ds_obs_t = ds_obs['era5land_temperature_2m']
            ds_obs_s = ds_obs['era5land_surface_net_solar_radiation']

            ds_fcst = ds_fcst_all.sel(basin=f'camels_{gaugeID}')
    
            for fcst_date in forecast_dates:
                print(f"⏱️ Done {gaugeID} {fcst_date} in {time.time() - start:.2f}s", flush=True)
                try:
                    # make weekly window
                    start_weekly = fcst_date - pd.Timedelta(days=305)
                    end_weekly   = fcst_date - pd.Timedelta(days=60) - pd.Timedelta(days=1)
                    
                    # 60 full daily points
                    start_daily  = fcst_date - pd.Timedelta(days=60)
                    end_daily    = fcst_date - pd.Timedelta(days=1)
                    
                    # forecast: 10 days inclusive of fcst_date
                    start_forecast = fcst_date
                    end_forecast   = fcst_date + pd.Timedelta(days=10)

                    # Streamflow
                    q_weekly = dfQ.loc[start_weekly:end_weekly]
                    q_weekly = q_weekly['streamflow_cms'].resample('7D').mean()
                    q_daily = dfQ.loc[start_daily:end_daily]['streamflow_cms']
                    q_forecast = dfQ.loc[start_forecast:end_forecast]['streamflow_cms']
                    q_combined = pd.concat([q_weekly, q_daily, q_forecast]).to_xarray()
                    q_combined.name = 'streamflow'
                    
                    if len(q_combined) != EXPECTED_LEN:
                        print(f"[{gaugeID}] {fcst_date.date()} q_combined length: {len(q_combined)}")
                    
                    ##### Precipitation prep
                    obs_weekly_p = ds_obs_p.sel(date=slice(start_weekly, end_weekly)).resample(date='7D').mean()
                    obs_weekly_t = ds_obs_t.sel(date=slice(start_weekly, end_weekly)).resample(date='7D').mean()
                    obs_weekly_s = ds_obs_s.sel(date=slice(start_weekly, end_weekly)).resample(date='7D').mean()
                    
                    obs_daily_p = ds_obs_p.sel(date=slice(start_daily, end_daily +  pd.Timedelta(days=1)))
                    obs_daily_t = ds_obs_t.sel(date=slice(start_daily, end_daily +  pd.Timedelta(days=1)))
                    obs_daily_s = ds_obs_s.sel(date=slice(start_daily, end_daily +  pd.Timedelta(days=1)))
            
                    tmp = ds_fcst.sel(date=fcst_date, method='nearest')
                    forecast_dates_expanded = pd.Timestamp(tmp.date.values) + pd.to_timedelta(tmp.lead_time)
                    tmp = tmp.assign_coords(date=('lead_time', forecast_dates_expanded))
                    forecast_data = tmp.swap_dims({'lead_time': 'date'}).drop_vars('lead_time').isel(date=slice(0, 10))
                    forecast_data_p = forecast_data['hres_total_precipitation']
                    forecast_data_t = forecast_data['hres_temperature_2m']
                    forecast_data_s = forecast_data['hres_surface_net_solar_radiation']
            
                    precipitation_concat = xr.concat([obs_weekly_p, obs_daily_p, forecast_data_p],dim='date')
                    temperature_concat   = xr.concat([obs_weekly_t, obs_daily_t, forecast_data_t],dim='date')
                    netSrad_concat       = xr.concat([obs_weekly_s, obs_daily_s, forecast_data_s],dim='date')

                    if precipitation_concat.shape[0] != EXPECTED_LEN:
                        print(f"[{gaugeID}] {fcst_date.date()} → shape mismatch: {precipitation_concat.shape[0]} vs {EXPECTED_LEN}")

                    # Flags
                    flags = np.concatenate([
                        np.full(obs_weekly_p.date.size, 0),
                        np.full(obs_daily_p.date.size, 1),
                        np.full(forecast_data_p.date.size, 2)
                    ])
                    

                    sample = {
                        'precip': precipitation_concat.values.astype(np.float32),
                        'temp': temperature_concat.values.astype(np.float32),
                        'net_solar': netSrad_concat.values.astype(np.float32),
                        'flag': flags,
                        'flow': q_combined.values.astype(np.float32),
                        'target': q_combined.values.astype(np.float32),
                        'basin_id': gaugeID,
                        "forecast_date": fcst_date.strftime("%Y-%m-%d")
                    }
                    
                    # ---------- validation ----------
                    # 1.  any NaNs in precip?
                    if np.isnan(sample["precip"]).any():
                        raise ValueError("NaNs found in precip array")

                    if np.isnan(sample["target"]).any():
                        raise ValueError("No Streamflow Data Available")
            
                    # 2.  precip shape drift?
                    if expected_precip_shape is None:
                        expected_precip_shape = sample["precip"].shape
                    elif sample["precip"].shape != expected_precip_shape:
                        raise ValueError(
                            f"Precip shape changed from {expected_precip_shape} "
                            f"to {sample['precip'].shape}"
                        )
                    # ---------------------------------                    
                    # ✅ Validate sample BEFORE adding it
                    valid = True
                    try:
                        if np.isnan(sample["precip"]).any():
                            raise ValueError("NaNs found in precip array")
                        if np.isnan(sample["target"]).any():
                            raise ValueError("No Streamflow Data Available")
                        if sample["precip"].shape != (EXPECTED_LEN,):
                            raise ValueError("Precip shape mismatch")
                        if sample["target"].shape != (EXPECTED_LEN,):
                            raise ValueError("Target shape mismatch")
                    except ValueError as ve:
                        print(f"⚠️ Skipping invalid sample for {gaugeID} {fcst_date.date()}: {ve}")
                        valid = False
                    
                    if valid:
                        intermediate_store.append(sample)
                        print(f"📦 Batch size: {len(intermediate_store)}", flush=True)
                    
                        if len(intermediate_store) == BATCH_SIZE:
                            write_batch(split, intermediate_store, count, EXPECTED_LEN)
                            count += 1
                            intermediate_store.clear()
                                            
                except Exception as e:
                    print(f"Skipping {fcst_date} for {gaugeID} due to error: {e}")
                    continue
                        
            # After processing all forecast dates for one gauge:
            del ds_obs, ds_fcst, ds_obs_p, ds_obs_t, ds_obs_s
            # gc.collect()
            
            print(f"✅ Finished {gaugeID}", flush=True)

        except Exception as e:
            print(f"Failed to process gauge {gaugeID}: {e}")

    # Write final remainder
    if intermediate_store:
        try:
            write_batch(split, intermediate_store, count, EXPECTED_LEN)
            print(f"✅ Final batch written for split '{split}'")
        except ValueError as e:
            print(f"⚠️ Skipping final batch due to invalid samples: {e}")
        intermediate_store.clear()

if False:  # change to True when you're ready
    for split in ["train", "validation", "test"]:
        files = f"/Projects/HydroMet/currierw/HRES_processed/{split}_data_batch*.nc"
        ds_all = xr.open_mfdataset(files, combine='nested', concat_dim='sample')
        ds_all.to_netcdf(f"/Projects/HydroMet/currierw/HRES_processed/{split}_data_combined.nc")


# def get_usgs_streamflow(site, start_date, end_date):
#     """
#     Download daily streamflow data from USGS NWIS for a given site and date range.

#     Parameters:
#         site (str): USGS site number (gauge ID)
#         start_date (str): Start date in 'YYYY-MM-DD'
#         end_date (str): End date in 'YYYY-MM-DD'

#     Returns:
#         pd.DataFrame: DataFrame with datetime and streamflow values in cfs
#     """
#     url = (
#         "https://waterservices.usgs.gov/nwis/dv/"
#         "?format=rdb&sites={site}&startDT={start}&endDT={end}"
#         "&parameterCd=00060&siteStatus=all"
#     ).format(site=site, start=start_date, end=end_date)

#     df = pd.read_csv(url, comment='#', sep='\t', header=1, parse_dates=['20d'])
#     df = df.rename(columns={'14n': 'streamflow_cfs'})
#     df = df.rename(columns={'20d': 'date'})
#     df['streamflow_cfs'] = pd.to_numeric(df['streamflow_cfs'], errors='coerce')

#     # Convert to cubic meters per second (cms)
#     df['streamflow_cms'] = df['streamflow_cfs'] * 0.0283168
#     df = df[['date', 'streamflow_cms']]
#     df = df.set_index('date')

#     return df



# -----------------------------------
# Loop through Gauges and Forecast Dates
# -----------------------------------
