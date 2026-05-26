"""Merge per-year fdir NetCDFs into data/interim/era5_spb_v2.zarr.

Reads ``data/raw/era5_v2/era5_spb_v2_fdir_<year>.nc`` for 2014..2024,
concatenates along time, normalizes the ``valid_time`` coordinate to
``time``, validates shape/grid against the existing v2 zarr, and writes
``fdir`` as an additional variable into the same zarr store.

Run after ``scripts/fetch_fdir_per_year.py`` finishes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger("merge_fdir")
ZARR_PATH = Path("data/interim/era5_spb_v2.zarr")
RAW_DIR = Path("data/raw/era5_v2")
YEARS = list(range(2014, 2025))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.info("opening existing v2 zarr: %s", ZARR_PATH)
    base = xr.open_zarr(ZARR_PATH, consolidated=False)
    if "fdir" in base.data_vars:
        LOGGER.info("fdir already present in v2 zarr; nothing to do")
        return 0

    paths = [RAW_DIR / f"era5_spb_v2_fdir_{year}.nc" for year in YEARS]
    missing = [p for p in paths if not p.exists() or p.stat().st_size == 0]
    if missing:
        LOGGER.error("missing fdir files: %s", [str(p) for p in missing])
        return 1

    LOGGER.info("reading %d per-year fdir NetCDFs", len(paths))
    per_year = []
    for path in paths:
        ds = xr.open_dataset(path, engine="netcdf4")
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        if "expver" in ds.dims:
            # 1 = ERA5 reanalysis, 5 = ERA5T — take the merged max across them.
            ds = ds.max("expver", skipna=True)
        ds = ds.drop_vars([v for v in ("number", "expver") if v in ds.coords], errors="ignore")
        per_year.append(ds[["fdir"]].load())

    combined = xr.concat(per_year, dim="time", combine_attrs="drop_conflicts").sortby("time")
    combined = combined.assign_coords(
        time=pd.DatetimeIndex(pd.to_datetime(combined["time"].values)).to_numpy("datetime64[ns]")
    )

    LOGGER.info("combined fdir shape=%s, time=%s..%s",
                combined["fdir"].shape,
                str(combined["time"].values[0])[:13],
                str(combined["time"].values[-1])[:13])

    base_time = pd.DatetimeIndex(pd.to_datetime(base["time"].values))
    fdir_time = pd.DatetimeIndex(pd.to_datetime(combined["time"].values))
    if len(base_time) != len(fdir_time) or not np.array_equal(base_time, fdir_time):
        LOGGER.error("time-coord mismatch: base=%d (%s..%s), fdir=%d (%s..%s)",
                     len(base_time), base_time[0], base_time[-1],
                     len(fdir_time), fdir_time[0], fdir_time[-1])
        return 1
    if not np.array_equal(base["latitude"].values, combined["latitude"].values):
        LOGGER.error("latitude mismatch")
        return 1
    if not np.array_equal(base["longitude"].values, combined["longitude"].values):
        LOGGER.error("longitude mismatch")
        return 1

    fdir = combined["fdir"].astype("float32")
    fdir.attrs.update(
        units="J m**-2",
        long_name="Total sky direct solar radiation at surface",
        era5_cds_variable="total_sky_direct_solar_radiation_at_surface",
        description="J/m² accumulated over the previous hour (divide by 3600 for mean W/m²)",
    )
    out = fdir.to_dataset(name="fdir")

    # Match existing time chunking convention (one month worth) so the
    # appended variable lines up with the others.
    out["fdir"].encoding = {"chunks": (min(31 * 24, out.sizes["time"]), 5, 7)}

    LOGGER.info("appending fdir to %s", ZARR_PATH)
    out.to_zarr(ZARR_PATH, mode="a", zarr_format=2)

    verified = xr.open_zarr(ZARR_PATH, consolidated=False)
    if "fdir" not in verified.data_vars:
        LOGGER.error("verification failed: fdir not present after write")
        return 1
    LOGGER.info("fdir present; sample max (2020-07-15T12:00Z, all cells): %.0f J/m² (~%.0f W/m² mean)",
                float(verified["fdir"].sel(time="2020-07-15T12").max().compute()),
                float(verified["fdir"].sel(time="2020-07-15T12").max().compute()) / 3600.0)
    LOGGER.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
