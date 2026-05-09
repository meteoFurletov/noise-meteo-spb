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
    n_samples        int32, number of hourly samples in this bin
                     dims: (cell, sector, season, period)
                     (does NOT vary with sector — included with sector dim
                      for convenience; or alternatively store as separate
                      array of shape (cell, season, period))

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

import xarray as xr


def aggregate_to_climatology(favorable_ds: xr.Dataset, config: dict) -> xr.Dataset:
    """Aggregate per-hour favorable flags into the climatology lookup table.

    Args:
        favorable_ds: Dataset from Step 3 with favorable[time, cell, sector].
        config: Project configuration.

    Returns:
        Dataset with p_favorable and n_samples on (cell, sector, season, period).
    """
    raise NotImplementedError


def assign_season(local_time: xr.DataArray, season_map: dict) -> xr.DataArray:
    """Map each timestep to its season label."""
    raise NotImplementedError


def assign_period(local_time: xr.DataArray, period_config: dict) -> xr.DataArray:
    """Map each timestep to its Lden period label.

    Note: night wraps midnight; use boolean OR.
    """
    raise NotImplementedError


def write_netcdf(ds: xr.Dataset, path: str, metadata: dict) -> None:
    """Write the climatology dataset to NetCDF with CF-compliant attributes."""
    raise NotImplementedError
