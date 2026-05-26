"""Season and Lden-period assignment for Step 4 v2 aggregation.

Both functions operate on a UTC, tz-naive ``time`` coordinate (as stored in
``favorable_v2.zarr``). Period assignment converts UTC to Saint Petersburg
local time by adding a fixed offset (UTC+3, year-round — Russia does not
observe DST since 2014).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

SEASON_ORDER: tuple[str, ...] = ("DJF", "MAM", "JJA", "SON")
PERIOD_ORDER: tuple[str, ...] = ("day", "evening", "night")


def assign_season(time: xr.DataArray, season_map: dict[str, list[int]]) -> xr.DataArray:
    """Map each UTC timestep to its meteorological season label.

    Month boundaries are within hours of local-time month boundaries (UTC+3,
    no DST), so using UTC month is exact at the season level.
    """
    month = time.dt.month.values
    labels = np.full(month.shape, "", dtype="<U3")
    for season, months in season_map.items():
        labels[np.isin(month, months)] = season
    if np.any(labels == ""):
        missing = sorted(set(month[labels == ""].tolist()))
        raise ValueError(f"No season mapping for month(s): {missing}")
    return xr.DataArray(labels, dims=time.dims, coords={"time": time["time"]}, name="season")


def assign_period(
    time: xr.DataArray,
    period_config: dict[str, dict[str, int] | list[int]],
    tz_offset_hours: int = 3,
) -> xr.DataArray:
    """Map each UTC timestep to its Lden period in *local* time.

    ``period_config`` accepts either ``{period: {"start_hour": h, "end_hour": h}}``
    or ``{period: [start, end]}``. ``end`` is exclusive; ``start > end`` wraps
    midnight (e.g. night = 23..7).
    """
    utc_hour = time.dt.hour.values
    local_hour = (utc_hour + tz_offset_hours) % 24
    labels = np.full(local_hour.shape, "", dtype="<U7")
    assigned = np.zeros(local_hour.shape, dtype=bool)
    for period, bounds in period_config.items():
        if isinstance(bounds, dict):
            start = int(bounds["start_hour"])
            end = int(bounds["end_hour"])
        else:
            start = int(bounds[0])
            end = int(bounds[1])
        if start < end:
            mask = (local_hour >= start) & (local_hour < end)
        elif start > end:
            mask = (local_hour >= start) | (local_hour < end)
        else:
            mask = np.ones(local_hour.shape, dtype=bool)
        if np.any(assigned & mask):
            overlap = sorted(set(local_hour[assigned & mask].tolist()))
            raise ValueError(f"Overlapping period definitions at local hour(s): {overlap}")
        labels[mask] = period
        assigned |= mask
    if np.any(labels == ""):
        missing = sorted(set(local_hour[labels == ""].tolist()))
        raise ValueError(f"No period mapping for local hour(s): {missing}")
    return xr.DataArray(labels, dims=time.dims, coords={"time": time["time"]}, name="period")
