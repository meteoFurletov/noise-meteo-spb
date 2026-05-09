"""
Step 1: ERA5 fetch and dataset assembly.

Pulls ten years of hourly ERA5 reanalysis data for the Saint Petersburg
domain into a local Zarr cache. The cache is the input that all downstream
steps read from.

This is the only step that touches external data sources. Once it has run,
the rest of the pipeline runs offline.

────────────────────────────────────────────────────────────────────────────
Output contract — data/interim/era5_spb.zarr
────────────────────────────────────────────────────────────────────────────

Dimensions:
    time           ~96,000 hourly timesteps (2014-01-01 00:00 to 2024-12-31 23:00 UTC)
    latitude       ~5 cells (0.25° spacing across 59.5°–60.5°N)
    longitude      ~7 cells (0.25° spacing across 29.5°–31.0°E)
    pressure_level 1 (1000 hPa, used for upper wind in Richardson)

Coordinates:
    time           datetime64[ns], UTC
    latitude       float, decimal degrees N
    longitude      float, decimal degrees E
    pressure_level int, hPa
    local_time     datetime64[ns], computed from time + UTC+3, used for
                   day/evening/night classification downstream

Variables (single-level, all with dims (time, latitude, longitude)):
    u10            10m u-component of wind, m/s
    v10            10m v-component of wind, m/s
    t2m            2m temperature, K
    d2m            2m dewpoint temperature, K
    sp             surface pressure, Pa
    tcc            total cloud cover, fraction 0–1
    blh            boundary layer height, m
    tp             total precipitation in past hour, m

Variables (pressure-level, with dims (time, pressure_level, latitude, longitude)):
    u              u-component of wind, m/s
    v              v-component of wind, m/s
    t              temperature, K

────────────────────────────────────────────────────────────────────────────
Methodological choices
────────────────────────────────────────────────────────────────────────────

ARCO-ERA5 (Google Cloud, Zarr) is the preferred source — it allows lazy
spatial slicing without downloading global files. The CDS API is the
fallback for environments where Google Cloud is unreachable.

The 1000 hPa pressure level is used as the "~110 m AGL" reference for
Richardson's upper height. SPb is essentially at sea level so 1000 hPa
≈ 110 m geopotential height. This is one of the methodological choices
flagged for sensitivity analysis in the paper.

Downloads are cached. Reruns of this step are no-ops if the Zarr cache
exists and covers the requested time/space window; force a refresh by
deleting data/interim/era5_spb.zarr.

────────────────────────────────────────────────────────────────────────────
Implementation hints (for Claude Code)
────────────────────────────────────────────────────────────────────────────

1. Open the ARCO-ERA5 Zarr store (gs://gcp-public-data-arco-era5/...).
2. Slice spatially to the configured bounding box.
3. Slice temporally to the configured time window.
4. Select only the configured variables.
5. Compute and attach `local_time` coordinate.
6. Write to data/interim/era5_spb.zarr with sensible chunking
   (chunk by month in time, by 1 in pressure_level, full domain in space).

The whole step should be ~50 lines of xarray code; complexity is low.
"""

from pathlib import Path

import xarray as xr


def fetch_era5(config: dict) -> xr.Dataset:
    """Fetch ERA5 data per config and return the assembled dataset.

    Implementation goes here. The function should:
    - read config["era5"], config["domain"], config["time"]
    - open ARCO-ERA5 (or fallback to CDS) per config["era5"]["source"]
    - slice spatially and temporally
    - add local_time coordinate
    - write to data/interim/era5_spb.zarr
    - return the opened dataset
    """
    raise NotImplementedError


def open_cached(path: Path = Path("data/interim/era5_spb.zarr")) -> xr.Dataset:
    """Open the cached Zarr dataset for downstream steps."""
    return xr.open_zarr(path)
