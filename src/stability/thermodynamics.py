"""Shared thermodynamic primitives for Step 2 v2.

Pure xarray functions (no module-level state, no I/O). Every routine
operates on full ``xarray.DataArray`` objects so the runner can express
the whole pipeline as broadcasted operations on the v2 cache.

The constants are intentionally explicit at the call site (passed via the
config dict) rather than baked in here. The defaults documented in the
docstrings match those in ``configs/spb_default.yaml`` under
``stability.monin_obukhov`` and ``stability.t100_interpolation``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

GRAVITY_M_PER_S2 = 9.80665
REFERENCE_PRESSURE_PA = 100000.0
KAPPA_DRY = 0.286  # R_d / c_p for dry air


def saturation_vapor_pressure_magnus(temperature_K: xr.DataArray) -> xr.DataArray:
    """Magnus formula for saturation vapor pressure in Pa.

    Evaluating at the dewpoint temperature yields the actual vapor pressure.
    """
    t_c = temperature_K - 273.15
    return 611.2 * np.exp((17.62 * t_c) / (243.12 + t_c))


def specific_humidity(dewpoint_K: xr.DataArray, pressure_Pa: xr.DataArray) -> xr.DataArray:
    """Specific humidity ``q`` from dewpoint and pressure.

    ``q = 0.622·e / (p - 0.378·e)`` where ``e`` is the actual vapor pressure
    obtained from the Magnus formula evaluated at the dewpoint.
    """
    e = saturation_vapor_pressure_magnus(dewpoint_K)
    return 0.622 * e / (pressure_Pa - 0.378 * e)


def virtual_temperature(temperature_K: xr.DataArray, specific_humidity_q: xr.DataArray) -> xr.DataArray:
    """Virtual temperature ``T_v = T · (1 + 0.61·q)``."""
    return temperature_K * (1.0 + 0.61 * specific_humidity_q)


def potential_temperature(
    temperature_K: xr.DataArray,
    pressure_Pa: xr.DataArray,
    reference_pressure_Pa: float = REFERENCE_PRESSURE_PA,
    kappa: float = KAPPA_DRY,
) -> xr.DataArray:
    """Potential temperature ``θ = T · (p_ref/p)^κ`` (dry exponent)."""
    return temperature_K * (reference_pressure_Pa / pressure_Pa) ** kappa


def air_density(
    pressure_Pa: xr.DataArray,
    virtual_temperature_K: xr.DataArray,
    Rd: float = 287.05,
) -> xr.DataArray:
    """Ideal-gas density ``ρ = p / (R_d · T_v)``."""
    return pressure_Pa / (Rd * virtual_temperature_K)


def barometric_pressure(
    surface_pressure_Pa: xr.DataArray,
    height_m: float,
    virtual_temperature_K: xr.DataArray,
    Rd: float = 287.05,
    gravity: float = GRAVITY_M_PER_S2,
) -> xr.DataArray:
    """Pressure at ``height_m`` above the surface via the barometric formula.

    Uses the (constant) lower-level virtual temperature as the scale-height
    temperature. Adequate for the lowest ~100 m where the sub-meter
    pressure correction is negligible.
    """
    return surface_pressure_Pa * np.exp(-gravity * height_m / (Rd * virtual_temperature_K))


def height_agl_from_pressure_geopotential(
    pressure_level_geopotential_m2_s2: xr.DataArray,
    surface_geopotential_m2_s2: xr.DataArray,
    gravity: float = GRAVITY_M_PER_S2,
) -> xr.DataArray:
    """Convert pressure-level geopotential to height AGL in metres.

    Both inputs are in m²/s² (ERA5's ``z`` on pressure levels and the static
    surface ``z_surface``). The result is per (time, latitude, longitude)
    for a given pressure level — pressure-level geopotential height varies
    with time and location.
    """
    return pressure_level_geopotential_m2_s2 / gravity - surface_geopotential_m2_s2 / gravity


def interpolate_t100(
    t2m: xr.DataArray,
    t_pressure: xr.DataArray,
    z_pressure_agl: xr.DataArray,
    target_height_m: float = 100.0,
    lower_reference_height_m: float = 2.0,
    dry_adiabatic_lapse_K_per_m: float = 0.00977,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Interpolate temperature at ``target_height_m`` AGL with a fallback hierarchy.

    Parameters
    ----------
    t2m
        2 m temperature, dims (time, latitude, longitude).
    t_pressure
        Pressure-level temperature, dims (time, level, latitude, longitude).
        The ``level`` coordinate carries the pressure values in hPa (e.g.
        1000 / 975 / 950).
    z_pressure_agl
        Pressure-level height above the surface in metres, dims matching
        ``t_pressure``. Computed by
        :func:`height_agl_from_pressure_geopotential`.
    target_height_m
        The AGL target height (default 100 m).
    lower_reference_height_m
        The reference height at which ``t2m`` is valid (default 2 m). Used by
        the case B fallback when the lowest pressure level is itself above
        ``target_height_m``.
    dry_adiabatic_lapse_K_per_m
        Lapse rate (K/m) used by the case C fallback when the target height
        is above every available pressure level (extreme low-surface case).

    Returns
    -------
    T_100 : xr.DataArray
        Interpolated temperature at ``target_height_m`` AGL,
        dims (time, latitude, longitude).
    qc_fallback : xr.DataArray
        Boolean, True where the dry-adiabatic fallback (case C) was used.

    Notes
    -----
    The three cases follow the v2 plan §4.3:

        A. ``target_height_m`` is bracketed by an adjacent pressure-level
           pair → linear-in-height interpolation between them.
        B. All pressure levels are above the target (very low surface) →
           linear interpolation between ``t2m`` and the lowest pressure
           level.
        C. All pressure levels are below the target (extreme low surface) →
           dry-adiabatic lapse applied to ``t2m``.

    Cases A and B are principled; only case C is flagged in ``qc_fallback``.
    """
    if "level" not in t_pressure.dims or "level" not in z_pressure_agl.dims:
        raise KeyError("t_pressure and z_pressure_agl must have a 'level' dimension.")

    # Sort levels by ascending height (i.e. descending pressure) for predictable
    # adjacent-pair iteration. The level coordinate stays a regular xarray dim.
    t_sorted = t_pressure.sortby("level", ascending=False)
    z_sorted = z_pressure_agl.sortby("level", ascending=False)
    levels = list(t_sorted["level"].values)

    # Start with the case-C fallback (dry adiabatic from 2 m); cases A and B
    # overwrite where they apply.
    T_100 = t2m - dry_adiabatic_lapse_K_per_m * (target_height_m - lower_reference_height_m)
    qc_fallback = xr.ones_like(t2m, dtype=bool)

    # Case B: target is between t2m and the lowest pressure level above the
    # surface that is also above the target.
    z_lowest = z_sorted.isel(level=0)
    t_lowest = t_sorted.isel(level=0)
    case_B_mask = z_lowest > target_height_m
    alpha_B = (target_height_m - lower_reference_height_m) / (z_lowest - lower_reference_height_m)
    T_100_B = t2m + alpha_B * (t_lowest - t2m)
    T_100 = xr.where(case_B_mask, T_100_B, T_100)
    qc_fallback = xr.where(case_B_mask, False, qc_fallback)

    # Case A: target is bracketed by adjacent pressure levels.
    for lo_idx in range(len(levels) - 1):
        hi_idx = lo_idx + 1
        z_lo = z_sorted.isel(level=lo_idx)  # lower altitude (higher pressure)
        z_hi = z_sorted.isel(level=hi_idx)  # higher altitude (lower pressure)
        t_lo = t_sorted.isel(level=lo_idx)
        t_hi = t_sorted.isel(level=hi_idx)
        bracket_mask = (z_lo <= target_height_m) & (z_hi > target_height_m)
        alpha_A = (target_height_m - z_lo) / (z_hi - z_lo)
        T_100_A = t_lo + alpha_A * (t_hi - t_lo)
        T_100 = xr.where(bracket_mask, T_100_A, T_100)
        qc_fallback = xr.where(bracket_mask, False, qc_fallback)

    T_100 = T_100.astype("float32")
    qc_fallback = qc_fallback.astype(bool)
    T_100.name = "T_100"
    qc_fallback.name = "qc_t100_fallback"
    return T_100, qc_fallback


def virtual_potential_temperature_2m(
    t2m: xr.DataArray,
    d2m: xr.DataArray,
    sp: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Virtual potential temperature at 2 m. Returns ``(theta_v_2m, q_2m)``.

    The companion specific humidity is returned so callers can reuse it as
    the moisture estimate at higher levels (constant-q approximation in the
    lowest ~100 m) without recomputing.
    """
    q2 = specific_humidity(d2m, sp)
    Tv2 = virtual_temperature(t2m, q2)
    theta_v2 = potential_temperature(Tv2, sp)
    return theta_v2, q2


def virtual_potential_temperature_at_height(
    T_height_K: xr.DataArray,
    specific_humidity_q: xr.DataArray,
    pressure_Pa_at_height: xr.DataArray,
) -> xr.DataArray:
    """Virtual potential temperature at the given level."""
    Tv = virtual_temperature(T_height_K, specific_humidity_q)
    return potential_temperature(Tv, pressure_Pa_at_height)
