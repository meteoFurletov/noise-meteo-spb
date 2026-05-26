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

from pathlib import Path
import shutil

import numpy as np
import xarray as xr

DEFAULT_OUTPUT_PATH = Path("data/interim/favorable.zarr")
CLASS_LABELS = tuple("ABCDEFG")
CLASS_TO_INT = {name: i for i, name in enumerate(CLASS_LABELS)}


def compute_favorable(ds: xr.Dataset, stability: xr.Dataset, config: dict) -> xr.Dataset:
    """Compute per-sector favorable flag for every hour.

    The thermal term is selected from ``stability_<primary>`` according to
    ``config["stability"]["primary"]``. The default SPb config sets
    ``primary: richardson``, so Step 3 uses ``stability_ri`` rather than
    ``stability_pasquill`` for the ISO 9613-2 stable-stratification flag.

    Args:
        ds: ERA5 dataset with u10, v10 wind components.
        stability: Stability dataset from Step 2 (only stability_pasquill
                   or stability_ri is consulted, depending on config).
        config: Project configuration.

    Returns:
        Dataset with favorable, favorable_wind, favorable_thermal variables.
    """
    favorable_config = config["favorable"]
    wind_threshold = float(favorable_config["wind_threshold_ms"])

    u = _flatten_spatial(ds["u10"])
    v = _flatten_spatial(ds["v10"])
    stability_class = _flatten_spatial(stability[_stability_variable_name(config)])
    u, v, stability_class = xr.align(u, v, stability_class, join="exact")

    sector_az = sector_azimuths(int(config["sectors"]["count"]))
    wind_component = wind_component_along_sector(u, v, sector_az)
    favorable_wind = (wind_component >= wind_threshold).rename("favorable_wind")

    stable_codes = _stable_class_codes(favorable_config["stable_classes"])
    favorable_thermal = stability_class.isin(stable_codes).rename("favorable_thermal")
    favorable = (favorable_wind | favorable_thermal).rename("favorable")

    out = xr.Dataset(
        {
            "favorable": favorable.transpose("time", "cell", "sector"),
            "favorable_wind": favorable_wind.transpose("time", "cell", "sector"),
            "favorable_thermal": favorable_thermal.transpose("time", "cell"),
        },
        coords={"sector_azimuth_deg": sector_az},
        attrs={
            "criterion": "favorable_wind OR favorable_thermal",
            "wind_threshold_ms": wind_threshold,
            "stable_classes": ",".join(favorable_config["stable_classes"]),
            "stability_primary": config["stability"]["primary"],
            "sector_convention": "azimuth degrees clockwise from north, propagation toward",
        },
    )
    out["favorable"].attrs["long_name"] = "combined favorable propagation flag"
    out["favorable_wind"].attrs["long_name"] = "wind-component favorable propagation flag"
    out["favorable_thermal"].attrs["long_name"] = "stable-stratification favorable flag"

    output_path = Path(favorable_config.get("output_path", DEFAULT_OUTPUT_PATH))
    _write_zarr(out, output_path)
    return xr.open_zarr(output_path, consolidated=False)


def sector_azimuths(n_sectors: int) -> xr.DataArray:
    """Return the sector center azimuths in degrees, as a coordinate array."""
    if n_sectors <= 0:
        raise ValueError("n_sectors must be positive")
    width = 360.0 / n_sectors
    azimuths = np.arange(n_sectors, dtype=np.float32) * width
    return xr.DataArray(
        azimuths,
        dims=("sector",),
        coords={"sector": np.arange(n_sectors, dtype=np.int16)},
        name="sector_azimuth_deg",
        attrs={
            "long_name": "sector center azimuth",
            "units": "degrees",
            "convention": "clockwise from north, propagation toward",
        },
    )


def wind_component_along_sector(
    u: xr.DataArray, v: xr.DataArray, sector_az_deg: xr.DataArray
) -> xr.DataArray:
    """Project (u, v) wind onto sector direction. Returns dims of u plus sector."""
    theta = np.deg2rad(sector_az_deg)
    component = u * np.sin(theta) + v * np.cos(theta)
    component = component.rename("wind_component_along_sector")
    component.attrs.update(
        long_name="wind component along propagation sector",
        units=u.attrs.get("units", "m s-1"),
        formula="u10*sin(theta) + v10*cos(theta)",
    )
    return component


def _flatten_spatial(da: xr.DataArray) -> xr.DataArray:
    if "cell" in da.dims:
        return da
    if "latitude" not in da.dims or "longitude" not in da.dims:
        return da
    return da.stack(cell=("latitude", "longitude")).reset_index("cell")


def _stability_variable_name(config: dict) -> str:
    primary = config["stability"]["primary"]
    if primary == "richardson":
        return "stability_ri"
    if primary == "pasquill":
        return "stability_pasquill"
    raise ValueError(f"Unsupported stability.primary: {primary!r}")


def _stable_class_codes(stable_classes: list[str | int]) -> list[int]:
    codes: list[int] = []
    for class_name in stable_classes:
        if isinstance(class_name, str):
            codes.append(CLASS_TO_INT[class_name])
        else:
            codes.append(int(class_name))
    return codes


def _write_zarr(ds: xr.Dataset, output_path: Path) -> None:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(tmp_path, mode="w", zarr_format=2)

    if output_path.exists():
        shutil.rmtree(output_path)
    tmp_path.rename(output_path)
