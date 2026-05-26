"""Core groupby aggregation for Step 4 v2.

Consumes ``favorable_v2.zarr`` and produces the climatological lookup table
keyed by (latitude, longitude, sector?, season, period).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.aggregate.time_bins import (
    PERIOD_ORDER,
    SEASON_ORDER,
    assign_period,
    assign_season,
)

PROBABILITY_VARS_SECTOR = ("favorable", "favorable_wind")
PROBABILITY_VARS_THERMAL = (
    "favorable_thermal_Ri",
    "favorable_thermal_Ri_strict",
    "favorable_thermal_L_strict",
    "favorable_thermal_L_moderate",
    "favorable_thermal_PG",
)
REQUIRED_VARS = PROBABILITY_VARS_SECTOR + PROBABILITY_VARS_THERMAL


def aggregate_to_climatology(
    favorable_ds: xr.Dataset,
    season_map: dict[str, list[int]],
    period_config: dict,
    tz_offset_hours: int = 3,
) -> xr.Dataset:
    """Reduce per-hour favorable arrays to climatological probabilities.

    Returns a Dataset with seven probability variables plus ``n_samples``,
    indexed by (latitude, longitude, sector?, season, period).
    """
    _validate_input(favorable_ds)

    season = assign_season(favorable_ds["time"], season_map)
    period = assign_period(favorable_ds["time"], period_config, tz_offset_hours)
    sp = pd.MultiIndex.from_arrays(
        [season.values, period.values], names=("season", "period")
    )
    work = favorable_ds.assign_coords(season_period=("time", sp))

    out_vars: dict[str, xr.DataArray] = {}
    for name in PROBABILITY_VARS_SECTOR:
        out_vars[_prob_name(name)] = _group_probability(work[name])
    for name in PROBABILITY_VARS_THERMAL:
        out_vars[_prob_name(name)] = _group_probability(work[name])

    # n_samples — count of hours per (season, period) bin, broadcast over space.
    counts = work["favorable_thermal_Ri"].groupby("season_period").count("time")
    counts = counts.unstack("season_period").reset_coords(drop=True)
    counts = counts.reindex(season=list(SEASON_ORDER), period=list(PERIOD_ORDER))
    counts = counts.transpose("latitude", "longitude", "season", "period")
    out_vars["n_samples"] = counts.astype("int32")

    out = xr.Dataset(out_vars)
    out = out.assign_coords(
        latitude=favorable_ds["latitude"].astype("float32"),
        longitude=favorable_ds["longitude"].astype("float32"),
        sector=favorable_ds["sector"].astype("int8"),
        season=list(SEASON_ORDER),
        period=list(PERIOD_ORDER),
    )
    sector_centers = (np.arange(favorable_ds.sizes["sector"]) * 20).astype("float32")
    out = out.assign_coords(sector_center_iso_deg=("sector", sector_centers))
    return out


def _validate_input(ds: xr.Dataset) -> None:
    missing = [v for v in REQUIRED_VARS if v not in ds.data_vars]
    if missing:
        raise KeyError(f"favorable dataset missing required variables: {missing}")
    for v in PROBABILITY_VARS_SECTOR:
        if set(ds[v].dims) != {"time", "latitude", "longitude", "sector"}:
            raise ValueError(f"{v!r} has unexpected dims {ds[v].dims}")
    for v in PROBABILITY_VARS_THERMAL:
        if set(ds[v].dims) != {"time", "latitude", "longitude"}:
            raise ValueError(f"{v!r} has unexpected dims {ds[v].dims}")


def _prob_name(boolean_var: str) -> str:
    return "p_" + boolean_var


def _group_probability(da: xr.DataArray) -> xr.DataArray:
    grouped = da.astype("float32").groupby("season_period").mean("time")
    grouped = grouped.unstack("season_period").reset_coords(drop=True)
    grouped = grouped.reindex(season=list(SEASON_ORDER), period=list(PERIOD_ORDER))
    spatial_dims = [d for d in ("latitude", "longitude", "sector") if d in grouped.dims]
    order = tuple(spatial_dims) + ("season", "period")
    return grouped.transpose(*order).astype("float32")
