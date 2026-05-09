"""
Step 3: Per-sector favorable-propagation flag.

For every hour, every grid cell, and every of 18 azimuth sectors, computes
a binary flag indicating whether atmospheric conditions favor sound
propagation in that direction.

This step implements the criterion from ISO 9613-2 / ГОСТ 31295.2 strictly.
The criterion is the methodological foundation of the entire project; the
paper's normative framing depends on this being correct.

────────────────────────────────────────────────────────────────────────────
Favorable-condition criterion (ISO 9613-2)
────────────────────────────────────────────────────────────────────────────

A sector-hour is FAVORABLE if EITHER of these is true:

    (a) Wind component along the sector ≥ wind_threshold_ms.
        Default threshold: 2.0 m/s (CNOSSOS-EU practice).

    (b) Atmospheric stratification is stable (Pasquill class E, F, or G,
        equivalently bulk Richardson > ~0.1).
        When stable, ALL 18 sectors are flagged favorable for that hour,
        regardless of wind, because downward refraction from temperature
        is omnidirectional.

────────────────────────────────────────────────────────────────────────────
Output contract — data/interim/favorable.zarr
────────────────────────────────────────────────────────────────────────────

Dimensions:
    time              ~96,000 hourly timesteps
    cell              flat index over latitude × longitude (matches Step 1)
    sector            18, indexed 0..17

Variables:
    favorable         bool, dims (time, cell, sector)
    favorable_wind    bool, dims (time, cell, sector) — only the wind criterion
    favorable_thermal bool, dims (time, cell) — only the thermal criterion

The two component flags are retained because the relative contribution of
wind vs. thermal mechanisms to total favorability is a paper result.

────────────────────────────────────────────────────────────────────────────
Sector convention (be careful!)
────────────────────────────────────────────────────────────────────────────

Sector k is centered on azimuth angle k × 20°, measured FROM north,
clockwise, indicating the direction propagation goes TOWARD.

So sector 0 = "propagation toward north," and a north-going wind
(positive v, negative direction-from-which) is favorable for sector 0.

For each sector with center azimuth θ_k:
    sector_unit_vector = (sin(θ_k), cos(θ_k))    # (east, north)
    wind_component = u·sin(θ_k) + v·cos(θ_k)

This wind_component is the projection of the wind velocity onto the
direction propagation is going. Positive means wind helps propagation;
favorable = (wind_component ≥ wind_threshold_ms).

Common mistake: confusing "wind direction" (the direction wind comes FROM,
meteorological convention) with the propagation direction. Wind from the
WEST has positive u, which favors propagation TOWARD the EAST (sector 90°).

────────────────────────────────────────────────────────────────────────────
Implementation hints (for Claude Code)
────────────────────────────────────────────────────────────────────────────

The naive implementation creates an array of shape (time, cell, sector),
which for full SPb domain × 10 years is ~96,000 × ~50 × 18 = 86M booleans
= ~86 MB. Manageable.

A vectorized xarray approach:
- Create a 1D coordinate array of sector center azimuths (0, 20, 40, ..., 340).
- Broadcast wind components against sector unit vectors.
- Apply threshold.
- OR with the broadcast thermal flag (which has no sector dim).

The whole step is ~30 lines of xarray/numpy.

────────────────────────────────────────────────────────────────────────────
Sensitivity
────────────────────────────────────────────────────────────────────────────

The wind_threshold_ms parameter is one of the methodological choices flagged
for sensitivity analysis. The pipeline should be runnable with thresholds
of 1.0, 2.0, and 3.0 m/s and the resulting lookup tables compared. This
comparison is reported in the paper.
"""

import xarray as xr


def compute_favorable(ds: xr.Dataset, stability: xr.Dataset, config: dict) -> xr.Dataset:
    """Compute per-sector favorable flag for every hour.

    Args:
        ds: ERA5 dataset with u10, v10 wind components.
        stability: Stability dataset from Step 2 (only stability_pasquill
                   or stability_ri is consulted, depending on config).
        config: Project configuration.

    Returns:
        Dataset with favorable, favorable_wind, favorable_thermal variables.
    """
    raise NotImplementedError


def sector_azimuths(n_sectors: int) -> xr.DataArray:
    """Return the sector center azimuths in degrees, as a coordinate array."""
    raise NotImplementedError


def wind_component_along_sector(
    u: xr.DataArray, v: xr.DataArray, sector_az_deg: xr.DataArray
) -> xr.DataArray:
    """Project (u, v) wind onto sector direction. Returns dims of u plus sector."""
    raise NotImplementedError
