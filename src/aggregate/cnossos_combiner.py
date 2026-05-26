"""CNOSSOS-EU faithful climatology aggregation.

Combines `favorable_wind` with `favorable_thermal_PG` (Pasquill–Turner E+F+G)
per NMPB-Routes-2008 §VI / JRC Reference Report 2012, then groups by
(season, period) to produce the CNOSSOS-faithful companion to
`p_favorable_spb.nc`.
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

REQUIRED_INPUT_VARS = ("favorable_wind", "favorable_thermal_PG")


def build_cnossos_favorable(favorable_ds: xr.Dataset) -> xr.DataArray:
    """Return per-hour CNOSSOS-EU favorable flag: wind OR Pasquill E+F+G."""
    _validate_input(favorable_ds)
    wind = favorable_ds["favorable_wind"]
    pg = favorable_ds["favorable_thermal_PG"]
    return (wind | pg).rename("favorable_cnossos")


def aggregate_cnossos_to_climatology(
    favorable_ds: xr.Dataset,
    season_map: dict[str, list[int]],
    period_config: dict,
    tz_offset_hours: int = 3,
) -> xr.Dataset:
    """Reduce per-hour CNOSSOS-EU favorable flags to climatological probabilities."""
    _validate_input(favorable_ds)

    cnossos = build_cnossos_favorable(favorable_ds)
    season = assign_season(favorable_ds["time"], season_map)
    period = assign_period(favorable_ds["time"], period_config, tz_offset_hours)
    sp = pd.MultiIndex.from_arrays(
        [season.values, period.values], names=("season", "period")
    )

    wind = favorable_ds["favorable_wind"].assign_coords(season_period=("time", sp))
    pg = favorable_ds["favorable_thermal_PG"].assign_coords(season_period=("time", sp))
    cn = cnossos.assign_coords(season_period=("time", sp))

    out_vars: dict[str, xr.DataArray] = {
        "p_favorable_cnossos": _group_probability(cn),
        "p_favorable_wind": _group_probability(wind),
        "p_favorable_thermal_PG": _group_probability(pg),
    }

    counts = pg.groupby("season_period").count("time")
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
    missing = [v for v in REQUIRED_INPUT_VARS if v not in ds.data_vars]
    if missing:
        raise KeyError(f"favorable dataset missing required variables: {missing}")
    if set(ds["favorable_wind"].dims) != {"time", "latitude", "longitude", "sector"}:
        raise ValueError(f"favorable_wind has unexpected dims {ds['favorable_wind'].dims}")
    if set(ds["favorable_thermal_PG"].dims) != {"time", "latitude", "longitude"}:
        raise ValueError(
            f"favorable_thermal_PG has unexpected dims {ds['favorable_thermal_PG'].dims}"
        )


def _group_probability(da: xr.DataArray) -> xr.DataArray:
    grouped = da.astype("float32").groupby("season_period").mean("time")
    grouped = grouped.unstack("season_period").reset_coords(drop=True)
    grouped = grouped.reindex(season=list(SEASON_ORDER), period=list(PERIOD_ORDER))
    spatial_dims = [d for d in ("latitude", "longitude", "sector") if d in grouped.dims]
    order = tuple(spatial_dims) + ("season", "period")
    return grouped.transpose(*order).astype("float32")
