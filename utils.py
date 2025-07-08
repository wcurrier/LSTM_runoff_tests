import pandas as pd, numpy as np, xarray as xr
import traceback, pickle
import json, urllib.error

def load_streamflow_for_gauge(gauge_id):
    path = Path("/path/to/your/streamflow") / f"{gauge_id}.csv"
    return pd.read_csv(path, parse_dates=["date"])

def samples_to_xarray(samples):
    n = len(samples)
    dyn = np.zeros((n, EXPECTED_LEN, 4), np.float32)
    tgt = np.zeros((n, EXPECTED_LEN, 1), np.float32)
    bas = np.empty(n, 'U20')
    fdt = np.empty(n, 'U20')

    for i,s in enumerate(samples):
        dyn[i,:,0] = s["precip"]
        dyn[i,:,1] = s["temp"]
        dyn[i,:,2] = s["net_solar"]
        dyn[i,:,3] = s["flag"].astype(np.float32)
        tgt[i,:,0] = s["target"]
        bas[i] = s["basin_id"]
        fdt[i] = s["forecast_date"]

    ds = xr.Dataset(
        data_vars=dict(
            dynamic_inputs=(["sample","time","feature"], dyn,
                {"feature":["precip","temp","net_solar","flag"]}),
            targets        =(["sample","time","target"], tgt,
                {"target":["streamflow"]}),
        ),
        coords=dict(
            basin_id      =(["sample"], bas),
            forecast_date =(["sample"], fdt),
        ),
        attrs=dict(
            flag_description=json.dumps({"0":"weekly ERA5","1":"daily ERA5","2":"HRES"})
        )
    )
    return ds

def build_sample_from_disk(gauge_id, split, date):
    try:
        dfQ = load_streamflow_for_gauge(gauge_id)
        return build_sample(gauge_id, split, date, dfQ)
    except Exception as e:
        print(f"[{gauge_id}] Failed to build sample for {split} {date}: {e}")
        return None


def build_sample(gauge_id: str, split: str, fcst_date: pd.Timestamp, df_streamflow):
    """
    Returns dict | None.  No huge objects passed around.
    """
    """Return one sample dict or None."""
    # ----------------------------------------------------------------------
    try:
        # print(f"START [{gauge_id}] {fcst_date}")

        # print(f"[{gauge_id}] {fcst_date.date()} loading ERA5")
        ds_obs  = era5().sel(basin=f'camels_{gauge_id}')

        # print(f"[{gauge_id}] {fcst_date.date()} loading HRES")
        ds_fcst = hres().sel(basin=f'camels_{gauge_id}')


        # windows ---------------------------------------------------
        start_w = fcst_date - pd.Timedelta(days=305)
        end_w   = fcst_date - pd.Timedelta(days=60) - pd.Timedelta(days=1)
        start_d = fcst_date - pd.Timedelta(days=60)
        end_d   = fcst_date - pd.Timedelta(days=1)
        start_f = fcst_date
        end_f   = fcst_date + pd.Timedelta(days=10)

        # streamflow ------------------------------------------------
        q_week = df_streamflow.loc[start_w:end_w]['streamflow_cms'].resample('7D').mean()
        q_day  = df_streamflow.loc[start_d:end_d]['streamflow_cms']
        q_fore = df_streamflow.loc[start_f:end_f]['streamflow_cms']
        q_comb = pd.concat([q_week, q_day, q_fore]).to_xarray()

        if len(q_comb) != EXPECTED_LEN:
            # print(f"[{gauge_id}] {fcst_date.date()} q_comb length: {len(q_comb)}")
            return None

        # obs slices (eager) ---------------------------------------
        obs_wp = ds_obs['era5land_total_precipitation'        ].sel(date=slice(start_w, end_w)).resample(date='7D').mean().load()
        obs_wt = ds_obs['era5land_temperature_2m'             ].sel(date=slice(start_w, end_w)).resample(date='7D').mean().load()
        obs_ws = ds_obs['era5land_surface_net_solar_radiation'].sel(date=slice(start_w, end_w)).resample(date='7D').mean().load()

        obs_dp = ds_obs['era5land_total_precipitation'        ].sel(date=slice(start_d, end_d +  pd.Timedelta(days=1))).load()
        obs_dt = ds_obs['era5land_temperature_2m'             ].sel(date=slice(start_d, end_d +  pd.Timedelta(days=1))).load()
        obs_ds = ds_obs['era5land_surface_net_solar_radiation'].sel(date=slice(start_d, end_d +  pd.Timedelta(days=1))).load()

        # forecast slice -------------------------------------------
        tmp  = ds_fcst.sel(date=fcst_date, method='nearest').load()
        tmp  = tmp.assign_coords(date=('lead_time',
                     pd.Timestamp(tmp.date.values) + pd.to_timedelta(tmp.lead_time.values)))
        fcst = tmp.swap_dims({'lead_time':'date'}).drop_vars('lead_time').isel(date=slice(0,10))

        fc_p = fcst['hres_total_precipitation']
        fc_t = fcst['hres_temperature_2m']
        fc_s = fcst['hres_surface_net_solar_radiation']

        print(f"[{gauge_id}] {fcst_date.date()} fcst dates available: {ds_fcst.date.values[:5]} ...")

        # concat ----------------------------------------------------
        precip = xr.concat([obs_wp, obs_dp, fc_p], dim='date')
        temp   = xr.concat([obs_wt, obs_dt, fc_t], dim='date')
        nsrad  = xr.concat([obs_ws, obs_ds, fc_s], dim='date')

        # print(f"[{gauge_id}] {fcst_date.date()} precip shape: {precip.shape}")
        # print(f"[{gauge_id}] {fcst_date.date()} temp shape:   {temp.shape}")
        # print(f"[{gauge_id}] {fcst_date.date()} nsrad shape:  {nsrad.shape}")
        # print(f"[{gauge_id}] {fcst_date.date()} EXPECTED_LEN: {EXPECTED_LEN}")

        if precip.shape[0] != EXPECTED_LEN:
            # print(f"[{gauge_id}] {fcst_date.date()} → shape mismatch: {precip.shape[0]} vs {EXPECTED_LEN}")
            return None

        flags = np.concatenate([
            np.zeros(obs_wp.date.size, np.int8),
            np.ones (obs_dp.date.size, np.int8),
            np.full (fc_p.date.size, 2, np.int8)
        ])

        return {
            "split": split,
            "basin_id": gauge_id,
            "forecast_date": fcst_date.strftime('%Y-%m-%d'),
            "precip": precip.values.astype(np.float32),
            "temp":   temp.values.astype(np.float32),
            "net_solar": nsrad.values.astype(np.float32),
            "flag": flags,
            "flow":   q_comb.values.astype(np.float32),
            "target": q_comb.values.astype(np.float32),
        }
    except Exception as e:
        print(f"[{gauge_id}] {fcst_date.date()} → {e}")
        traceback.print_exc()
        return None

def get_or_download_streamflows(df, STREAMFLOW_PATH, start_date="2015-01-01", end_date="2024-12-31"):
    streamflow_file = STREAMFLOW_PATH / "streamflows.pkl"
    skipped_file    = STREAMFLOW_PATH / "skipped_gauges.txt"

    if streamflow_file.exists() and skipped_file.exists():
        print("🔁 Loading cached streamflows and skipped gauges...")
        with open(streamflow_file, "rb") as f:
            streamflows = pickle.load(f)
        with open(skipped_file, "r") as f:
            skipped_gauges = [line.strip() for line in f]
    else:
        print("⬇️  Downloading streamflows from USGS...")
        streamflows = {}
        skipped_gauges = []
        gauge_ids = df["gauge_id"].str.split("_").str[-1].tolist()

        for g in gauge_ids:
            dfQ = get_usgs_streamflow(g, start_date, end_date)
            if dfQ is None:
                skipped_gauges.append(g)
            else:
                streamflows[g] = dfQ

        # Save results
        STREAMFLOW_PATH.mkdir(exist_ok=True)
        with open(streamflow_file, "wb") as f:
            pickle.dump(streamflows, f)
        with open(skipped_file, "w") as f:
            f.write("\n".join(skipped_gauges))
        print(f"✅ Saved streamflows to {streamflow_file}")
        print(f"❌ Saved skipped gauges to {skipped_file}")

    return streamflows, skipped_gauges

def get_usgs_streamflow(site, start_date, end_date, min_end_date="2024-12-31"):
    """
    Download daily streamflow data from USGS NWIS for a given site and date range.
    Assumes columns '20d' (date) and '14n' (flow in cfs).

    Returns:
        pd.DataFrame or None if download fails or structure is unexpected
    """
    url = (
        "https://waterservices.usgs.gov/nwis/dv/"
        f"?format=rdb&sites={site}&startDT={start_date}&endDT={end_date}"
        "&parameterCd=00060&siteStatus=all"
    )

    try:
        df = pd.read_csv(url, comment="#", sep="\t", header=1, parse_dates=["20d"])
    except Exception as e:
        print(f"[{site}] failed to download: {e}; skipping")
        return None

    if "14n" not in df.columns or "20d" not in df.columns:
        print(f"[{site}] missing expected columns '20d' and '14n'; skipping")
        return None

    df = df.rename(columns={"14n": "streamflow_cfs", "20d": "date"})
    df["streamflow_cfs"] = pd.to_numeric(df["streamflow_cfs"], errors="coerce")

    # Remove rows with NaNs
    df = df.dropna(subset=["streamflow_cfs"])
    if df.empty:
        print(f"[{site}] all streamflow data missing or invalid; skipping")
        return None

    # Check time coverage
    if pd.to_datetime(df["date"].max()) < pd.to_datetime(min_end_date):
        print(f"[{site}] data ends at {df['date'].max()}, < {min_end_date}; skipping")
        return None

    # Convert to cubic meters per second (cms)
    df["streamflow_cms"] = df["streamflow_cfs"] * 0.0283168
    df = df[["date", "streamflow_cms"]].set_index("date").sort_index()

    return df
# # Example usage:
# # site_id = gaugeID  # Example gauge ID
# site_id = '09085000'

# start = '2015-01-01'
# end = '2024-12-31'

# streamflow_data = get_usgs_streamflow(site_id, start, end)
# print(streamflow_data.tail())

