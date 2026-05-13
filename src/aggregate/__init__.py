"""
Step 4: Climatological aggregation.

Collapses the per-hour favorable array from Step 3 into the climatological
lookup table that is the primary scientific output of this project.

THIS STEP PRODUCES THE DELIVERABLE. Everything upstream is plumbing;
everything downstream is visualization. The output of this step is the
file that gets cited in the paper, archived on Zenodo, and consumed by
noise calculation code.

────────────────────────────────────────────────────────────────────────────
Output contract — data/processed/p_favorable_spb.nc
────────────────────────────────────────────────────────────────────────────

Dimensions:
    cell    flat index over the SPb domain ERA5 grid (~50 cells)
    sector  18, indexed 0..17, center azimuths 0°, 20°, ..., 340°
    season  4: DJF, MAM, JJA, SON
    period  3: day, evening, night

Coordinates:
    cell             int, flat index
    cell_lat         float, latitude of cell center
    cell_lon         float, longitude of cell center
    sector           int, 0..17
    sector_az        float, sector center azimuth in degrees
    season           string, "DJF" | "MAM" | "JJA" | "SON"
    period           string, "day" | "evening" | "night"

Variables:
    p_favorable      float32, range 0–1, fraction of hours that were favorable
                     dims: (cell, sector, season, period)
    p_favorable_wind float32, range 0–1, wind-component fraction
                     dims: (cell, sector, season, period)
    p_favorable_thermal
                     float32, range 0–1, stable-stratification fraction
                     dims: (cell, season, period)
    n_samples        int32, number of hourly samples in this bin
                     dims: (cell, season, period)

Global attributes (CF conventions):
    title, institution, references, conventions, history, source.

────────────────────────────────────────────────────────────────────────────
Aggregation specification
────────────────────────────────────────────────────────────────────────────

Season assignment: by month per configs/spb_default.yaml seasons map.

Period assignment: by LOCAL HOUR (not UTC). Use the local_time coordinate
attached in Step 1. Note that "night" wraps midnight (≥23:00 OR <07:00),
which requires a careful boolean expression.

The aggregation is conceptually:

    p_favorable[c, s, season, period] = mean(favorable[t, c, s] over t
                                             where t in season and t in period)

Implemented as xarray.groupby on a synthetic (season, period) coordinate.

────────────────────────────────────────────────────────────────────────────
Quality flags and sample counts
────────────────────────────────────────────────────────────────────────────

n_samples MUST be carried through. Some bins (rare wind directions in
specific seasons) will have low sample counts. The paper reports both
p_favorable and a confidence interval derived from n_samples, treating
each hour as a Bernoulli sample. Estimates with n_samples < 100 should
be flagged in visualizations and discussed in the paper.

A reasonable expectation: 96,000 hours / (4 seasons × 3 periods) ~ 8,000
hours per (season, period) per cell. With 18 sectors, the average sector
gets all 8,000 (because thermal favorability is omnidirectional) but the
*wind-only* favorable count varies by sector. For diagnostic plots, the
sector-resolved sample count for the wind component is what matters.

────────────────────────────────────────────────────────────────────────────
Implementation hints (for Claude Code)
────────────────────────────────────────────────────────────────────────────

1. Open data/interim/favorable.zarr.
2. Add season and period coordinates from local_time.
3. Group by (season, period) and compute mean and count over time.
4. Stack/flatten lat,lon into cell dimension if not already.
5. Write to NetCDF with CF-compliant attributes.

The whole step is ~50 lines including metadata.

────────────────────────────────────────────────────────────────────────────
Reading downstream
────────────────────────────────────────────────────────────────────────────

The file is small enough to ship with the paper as supplementary data.
A consumer accessing it does:

    ds = xr.open_dataset("p_favorable_spb.nc")
    p = ds.p_favorable.sel(season="DJF", period="night").isel(cell=12)
    # 18-element array of nightly winter favorability for cell 12 by sector

This direct access pattern is the user-facing API of the project.
"""

from datetime import UTC, datetime
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import xarray as xr


SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
PERIOD_ORDER = ("day", "evening", "night")


def aggregate_to_climatology(favorable_ds: xr.Dataset, config: dict) -> xr.Dataset:
    """Aggregate per-hour favorable flags into the climatology lookup table.

    Args:
        favorable_ds: Dataset from Step 3 with favorable[time, cell, sector].
        config: Project configuration.

    Returns:
        Dataset with p_favorable, p_favorable_wind, p_favorable_thermal, and n_samples.
    """
    _validate_input(favorable_ds)

    season = assign_season(favorable_ds["local_time"], config["seasons"])
    period = assign_period(favorable_ds["local_time"], config["periods"])
    season_period = pd.MultiIndex.from_arrays(
        [season.values, period.values],
        names=("season", "period"),
    )

    work = favorable_ds.assign_coords(season_period=("time", season_period))

    p_favorable = _grouped_probability(work["favorable"], "p_favorable")
    p_favorable_wind = _grouped_probability(work["favorable_wind"], "p_favorable_wind")
    p_favorable_thermal = _grouped_probability(
        work["favorable_thermal"], "p_favorable_thermal"
    )
    n_samples = _grouped_count(work["favorable_thermal"], "n_samples")

    out = xr.Dataset(
        {
            "p_favorable": p_favorable,
            "p_favorable_wind": p_favorable_wind,
            "p_favorable_thermal": p_favorable_thermal,
            "n_samples": n_samples,
        }
    )

    out = out.assign_coords(
        cell=np.arange(favorable_ds.sizes["cell"], dtype=np.int32),
        season=list(SEASON_ORDER),
        period=list(PERIOD_ORDER),
    )
    if "sector" in favorable_ds.coords:
        out = out.assign_coords(sector=("sector", favorable_ds["sector"].values.astype(np.int16)))
    else:
        out = out.assign_coords(sector=np.arange(favorable_ds.sizes["sector"], dtype=np.int16))

    out = _attach_auxiliary_coordinates(out, favorable_ds)
    out = _attach_variable_attrs(out)
    return out


def assign_season(local_time: xr.DataArray, season_map: dict) -> xr.DataArray:
    """Map each timestep to its season label."""
    month = local_time.dt.month.values
    labels = np.full(month.shape, "", dtype="<U3")

    for season, months in season_map.items():
        labels[np.isin(month, months)] = season

    if np.any(labels == ""):
        missing = sorted(set(month[labels == ""].tolist()))
        raise ValueError(f"No season mapping for month(s): {missing}")

    return xr.DataArray(
        labels,
        dims=local_time.dims,
        coords=_time_coords(local_time),
        name="season",
        attrs={"long_name": "climatological season"},
    )


def assign_period(local_time: xr.DataArray, period_config: dict) -> xr.DataArray:
    """Map each timestep to its Lden period label.

    Note: night wraps midnight; use boolean OR.
    """
    hour = local_time.dt.hour.values
    labels = np.full(hour.shape, "", dtype="<U7")
    assigned = np.zeros(hour.shape, dtype=bool)

    for period, bounds in period_config.items():
        start = int(bounds["start_hour"])
        end = int(bounds["end_hour"])
        if start == end:
            mask = np.ones(hour.shape, dtype=bool)
        elif start < end:
            mask = (hour >= start) & (hour < end)
        else:
            mask = (hour >= start) | (hour < end)

        if np.any(assigned & mask):
            overlap_hours = sorted(set(hour[assigned & mask].tolist()))
            raise ValueError(f"Overlapping period definitions for hour(s): {overlap_hours}")

        labels[mask] = period
        assigned |= mask

    if np.any(labels == ""):
        missing_hours = sorted(set(hour[labels == ""].tolist()))
        raise ValueError(f"No period mapping for local hour(s): {missing_hours}")

    return xr.DataArray(
        labels,
        dims=local_time.dims,
        coords=_time_coords(local_time),
        name="period",
        attrs={"long_name": "Lden time-of-day period"},
    )


def write_netcdf(ds: xr.Dataset, path: str, metadata: dict) -> None:
    """Write the climatology dataset to NetCDF with CF-compliant attributes."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attrs = dict(metadata or {})
    attrs.update(
        {
            "title": "Climatological probability of favorable noise propagation conditions, "
            "Saint Petersburg, 2014-2024",
            "institution": "RSHU (Russian State Hydrometeorological University)",
            "source": "ERA5 hourly reanalysis, Copernicus Climate Data Store",
            "references": "GOST 31295.2-2005; ISO 9613-2:1996; Hersbach et al. 2020 (ERA5)",
            "Conventions": "CF-1.10",
            "conventions": "CF-1.10",
            "history": _history_attr(),
        }
    )
    attrs["Conventions"] = "CF-1.10"
    attrs["conventions"] = "CF-1.10"
    ds = _attach_variable_attrs(ds.copy())
    ds.attrs.update(attrs)

    encoding = {
        "p_favorable": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "p_favorable_wind": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "p_favorable_thermal": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "n_samples": {"zlib": True, "complevel": 4, "dtype": "int32"},
    }
    ds.to_netcdf(output_path, engine="netcdf4", encoding=encoding)


def _validate_input(ds: xr.Dataset) -> None:
    required_vars = {"favorable", "favorable_wind", "favorable_thermal"}
    required_coords = {"local_time"}
    missing_vars = required_vars.difference(ds.data_vars)
    missing_coords = required_coords.difference(ds.coords)
    if missing_vars or missing_coords:
        raise ValueError(
            "Step 3 favorable dataset is missing required content: "
            f"vars={sorted(missing_vars)}, coords={sorted(missing_coords)}"
        )

    expected_dims = {
        "favorable": ("time", "cell", "sector"),
        "favorable_wind": ("time", "cell", "sector"),
        "favorable_thermal": ("time", "cell"),
    }
    for name, dims in expected_dims.items():
        if ds[name].dims != dims:
            raise ValueError(f"{name!r} has dims {ds[name].dims}, expected {dims}")


def _time_coords(local_time: xr.DataArray) -> dict[str, xr.DataArray]:
    return {"time": local_time["time"]} if "time" in local_time.coords else {}


def _grouped_probability(da: xr.DataArray, name: str) -> xr.DataArray:
    grouped = da.groupby("season_period").mean("time").unstack("season_period")
    grouped = grouped.reset_coords(drop=True)
    grouped = grouped.reindex(season=list(SEASON_ORDER), period=list(PERIOD_ORDER))
    dim_order = tuple(dim for dim in ("cell", "sector", "season", "period") if dim in grouped.dims)
    return grouped.transpose(*dim_order).astype("float32").rename(name)


def _grouped_count(da: xr.DataArray, name: str) -> xr.DataArray:
    grouped = da.groupby("season_period").count("time").unstack("season_period")
    grouped = grouped.reset_coords(drop=True)
    grouped = grouped.reindex(season=list(SEASON_ORDER), period=list(PERIOD_ORDER))
    return grouped.transpose("cell", "season", "period").astype("int32").rename(name)


def _attach_auxiliary_coordinates(out: xr.Dataset, source: xr.Dataset) -> xr.Dataset:
    if "latitude" in source.coords:
        out = out.assign_coords(cell_lat=("cell", source["latitude"].values))
        out["cell_lat"].attrs.update(
            {
                "units": "degrees_north",
                "standard_name": "latitude",
                "long_name": "latitude of ERA5 grid-cell center",
            }
        )
    if "longitude" in source.coords:
        out = out.assign_coords(cell_lon=("cell", source["longitude"].values))
        out["cell_lon"].attrs.update(
            {
                "units": "degrees_east",
                "standard_name": "longitude",
                "long_name": "longitude of ERA5 grid-cell center",
            }
        )
    if "sector_azimuth_deg" in source.coords:
        out = out.assign_coords(sector_az=("sector", source["sector_azimuth_deg"].values))
    else:
        out = out.assign_coords(sector_az=("sector", (out["sector"].values * 20).astype("float32")))
    out["sector_az"].attrs.update(
        {
            "units": "degrees",
            "long_name": "sector center azimuth",
            "convention": "clockwise from north, direction toward which sound propagates",
        }
    )
    out["cell"].attrs.update({"long_name": "ERA5 grid-cell index"})
    out["sector"].attrs.update({"long_name": "azimuth sector index"})
    out["season"].attrs.update({"long_name": "climatological season"})
    out["period"].attrs.update({"long_name": "Lden time-of-day period"})
    return out


def _attach_variable_attrs(ds: xr.Dataset) -> xr.Dataset:
    ds["p_favorable"].attrs.update(
        {
            "units": "1",
            "long_name": "Probability of favorable noise propagation conditions",
            "criterion": "favorable_wind OR favorable_thermal",
        }
    )
    ds["p_favorable_wind"].attrs.update(
        {
            "units": "1",
            "long_name": "Probability of favorable wind-component propagation conditions",
        }
    )
    ds["p_favorable_thermal"].attrs.update(
        {
            "units": "1",
            "long_name": "Probability of favorable stable-stratification conditions",
        }
    )
    ds["n_samples"].attrs.update(
        {
            "units": "1",
            "long_name": "Number of hourly observations in bin",
        }
    )
    return ds


def _history_attr() -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit = _git_commit_hash()
    if commit:
        return f"Created {timestamp} by src.aggregate.write_netcdf; git_commit={commit}"
    return f"Created {timestamp} by src.aggregate.write_netcdf; git_commit=unavailable"


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
