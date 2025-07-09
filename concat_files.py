import xarray as xr
import numpy as np
from pathlib import Path

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