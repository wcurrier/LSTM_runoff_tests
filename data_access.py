# data_access.py

from functools import lru_cache
import xarray as xr
from pathlib import Path

BASE_OBS  = Path('/Projects/HydroMet/currierw/ERA5_LAND')
BASE_FCST = Path('/Projects/HydroMet/currierw/HRES')

@lru_cache(maxsize=None)
def era5():
    return xr.open_zarr(BASE_OBS / 'camels_rechunked.zarr',
                        consolidated=True, chunks={'date': 365})

@lru_cache(maxsize=None)
def hres():
    return xr.open_zarr(BASE_FCST / 'camels_rechunked.zarr',
                        consolidated=True, decode_timedelta=True,
                        chunks={'date': 365})
