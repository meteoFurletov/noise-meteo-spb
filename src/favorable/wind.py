"""Wind component along sector and wind-favorable flag (ISO 9613-2)."""

from __future__ import annotations

import numpy as np
import xarray as xr


def sector_azimuths_iso_deg(n_sectors: int = 18) -> xr.DataArray:
    """Sector center azimuths in degrees, ISO 9613-2 convention.

    Azimuth is measured clockwise from north and points TOWARD the receiver
    (the direction noise propagates). Sector k spans
    [(k - 0.5) * (360/n), (k + 0.5) * (360/n)) and is centered at k * (360/n).
    """
    if n_sectors <= 0:
        raise ValueError("n_sectors must be positive")
    width = 360.0 / n_sectors
    az = np.arange(n_sectors, dtype=np.float32) * width
    return xr.DataArray(
        az,
        dims=("sector",),
        coords={"sector": np.arange(n_sectors, dtype=np.int16)},
        name="sector_azimuth_iso_deg",
        attrs={
            "long_name": "sector center azimuth (ISO 9613-2)",
            "units": "degrees",
            "convention": "clockwise from north; azimuth = direction noise propagates TOWARD",
        },
    )


def wind_component_along_sector(
    u10: xr.DataArray, v10: xr.DataArray, sector_az_deg: xr.DataArray
) -> xr.DataArray:
    """Project the (u10, v10) wind vector onto each sector's unit vector.

    For sector k with center azimuth θ_k (ISO convention, measured clockwise
    from north toward the receiver), the unit vector is
    n̂_k = (sin θ_k, cos θ_k) in (east, north) coords. The projection is

        u_parallel(k) = u10 · sin θ_k + v10 · cos θ_k.

    Positive u_parallel means wind blows toward the receiver in sector k.
    """
    theta = np.deg2rad(sector_az_deg).astype(np.float32)
    sin_t = np.sin(theta).astype(np.float32)
    cos_t = np.cos(theta).astype(np.float32)
    component = u10 * sin_t + v10 * cos_t
    component.name = "wind_component_along_sector"
    component.attrs.update(
        long_name="wind component along propagation sector",
        units="m s-1",
        formula="u10 * sin(theta_iso) + v10 * cos(theta_iso)",
    )
    return component


def compute_favorable_wind(
    u10: xr.DataArray,
    v10: xr.DataArray,
    *,
    wind_threshold_ms: float = 2.0,
    n_sectors: int = 18,
) -> xr.DataArray:
    """Boolean wind-favorable flag, dims (time, latitude, longitude, sector)."""
    az = sector_azimuths_iso_deg(n_sectors)
    u_parallel = wind_component_along_sector(u10, v10, az)
    flag = (u_parallel >= np.float32(wind_threshold_ms)).rename("favorable_wind")
    flag.attrs.update(
        long_name="wind-component favorable propagation flag (ISO 9613-2)",
        wind_threshold_ms=float(wind_threshold_ms),
        sector_convention="iso_9613_2_to",
    )
    return flag
