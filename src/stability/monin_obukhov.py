"""Inverse Monin–Obukhov length 1/L from ERA5 instantaneous fluxes.

Reference: §5 of the Step 2 v2 specification. The recipe follows the
ECMWF Confluence guide *"ERA5: How to calculate Obukhov length"*.

    u_* = sqrt(|τ| / ρ)
    Q_H = −instantaneous_surface_sensible_heat_flux                 [W/m²]
    Q_E = −L_v · instantaneous_moisture_flux                        [W/m²]
    B   = Q_H / (ρ c_p) + 0.61 · T2m · Q_E / (ρ L_v)               [K·m/s]
    1/L = −κ g B / (T_v · u_*³)                                     [1/m]

Sign convention:
    1/L > 0  ↔ stable
    1/L < 0  ↔ unstable
ECMWF instantaneous surface heat-flux convention is positive *downward*
(into the surface); the explicit negation above flips it to the
meteorological "atmospheric heating from below" convention required by
the standard Obukhov formula.

The module is pure: it consumes an ``xarray.Dataset`` and the configuration
dict, returns an ``xarray.Dataset`` of diagnostics. No I/O.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from src.stability.thermodynamics import (
    GRAVITY_M_PER_S2,
    air_density,
    specific_humidity,
    virtual_temperature,
)


def compute(ds: xr.Dataset, cfg: dict) -> xr.Dataset:
    """Compute inverse Obukhov length and companion diagnostics.

    Returns variables ``inv_L``, ``u_star``, ``stable_L``, ``qc_low_ustar``
    over (time, latitude, longitude).
    """
    mo_cfg = cfg["stability"]["monin_obukhov"]
    threshold = float(mo_cfg["threshold_inv_L"])
    min_ustar = float(mo_cfg["min_ustar"])
    kappa = float(mo_cfg["kappa"])
    cp = float(mo_cfg["cp"])
    Lv = float(mo_cfg["Lv"])
    Rd = float(mo_cfg["Rd"])

    # Lower-level virtual temperature and density at 2 m / surface pressure.
    q2 = specific_humidity(ds["d2m"], ds["sp"])
    Tv2 = virtual_temperature(ds["t2m"], q2)
    rho = air_density(ds["sp"], Tv2, Rd=Rd)

    # Friction velocity from the two stress components. Sign of τ_x / τ_y is
    # irrelevant once magnitude is taken.
    tau_mag = np.hypot(ds["iews"], ds["inss"])
    u_star = np.sqrt(tau_mag / rho)
    qc_low_ustar = (u_star < min_ustar).astype(bool)
    qc_low_ustar.name = "qc_low_ustar"

    # Surface upward sensible and latent heat fluxes from ECMWF instantaneous
    # downward-positive fluxes (negate to convert).
    Q_H = -ds["ishf"]                          # W/m², upward positive
    Q_E = -Lv * ds["ie"]                       # W/m², upward positive

    # Buoyancy flux (K·m/s).
    B = Q_H / (rho * cp) + 0.61 * ds["t2m"] * Q_E / (rho * Lv)

    # Avoid divide-by-zero where u_star == 0 by masking below the QC threshold.
    safe_u_star = u_star.where(u_star >= min_ustar)
    inv_L = -(kappa * GRAVITY_M_PER_S2 * B) / (Tv2 * safe_u_star ** 3)

    inv_L = inv_L.astype("float32")
    inv_L.name = "inv_L"
    inv_L.attrs.update(
        long_name="inverse Monin-Obukhov length",
        units="m**-1",
        method="ECMWF instantaneous surface fluxes; sign>0 = stable",
        qc_dependencies="qc_low_ustar",
    )

    u_star = u_star.astype("float32")
    u_star.name = "u_star"
    u_star.attrs.update(
        long_name="friction velocity from surface stress",
        units="m s**-1",
        method="u* = sqrt(|tau|/rho) with rho computed from 2 m T_v and surface pressure",
    )

    stable_L = ((inv_L >= threshold) & inv_L.notnull()).astype(bool)
    stable_L.name = "stable_L"
    stable_L.attrs.update(
        long_name=f"1/L ≥ {threshold:g} m**-1",
        qc_dependencies="qc_low_ustar",
    )

    qc_low_ustar.attrs.update(
        long_name=f"u_* < {min_ustar:g} m/s; inv_L undefined",
    )

    return xr.Dataset(
        {
            "inv_L": inv_L,
            "u_star": u_star,
            "stable_L": stable_L,
            "qc_low_ustar": qc_low_ustar,
        }
    )


def verify_sign_convention(
    inv_L: xr.DataArray,
    local_time: xr.DataArray | None = None,
    reference_latitude: float = 59.75,
    reference_longitude: float = 30.25,
) -> dict:
    """Sign-sanity check on ``inv_L``.

    Computes regional means over a reference inland cell for:

      * JJA-day  : months ∈ {6,7,8}, UTC hour ∈ {9..14} (local 12–17, SPb)
        expected: < −0.005 m⁻¹ (convective unstable)
      * DJF-night: months ∈ {12,1,2}, UTC hour ∈ {0..3, 21..23} (local 00–06)
        expected: > +0.003 m⁻¹ (nocturnal stable)

    The default reference cell ``(59.75, 30.25)`` is over the Saint
    Petersburg city centre (z_surface ≈ 54 m). The Step 2 v2 spec
    originally specified ``(60.0, 30.0)``, but that cell sits over Neva
    Bay (z_surface ≈ 18 m, water) and shows winter-night upward heat flux
    that inverts the expected DJF-night sign without any sign-convention
    bug — see docs/step2_v2_report.md §4.

    Returns a dict with the two means, the pass/fail flags, and a combined
    ``passed`` flag. The runner aborts with a diagnostic when ``passed`` is
    False rather than silently flipping the sign.
    """
    time = inv_L["time"]
    months = time.dt.month
    hours = time.dt.hour

    jja_day_mask = months.isin([6, 7, 8]) & hours.isin([9, 10, 11, 12, 13, 14])
    djf_night_mask = months.isin([12, 1, 2]) & hours.isin([0, 1, 2, 3, 21, 22, 23])

    central = inv_L.sel(
        latitude=reference_latitude, longitude=reference_longitude, method="nearest"
    )
    jja_day_mean = float(central.where(jja_day_mask).mean(skipna=True).compute())
    djf_night_mean = float(central.where(djf_night_mask).mean(skipna=True).compute())

    jja_ok = jja_day_mean < -0.005
    djf_ok = djf_night_mean > 0.003

    return {
        "jja_day_mean": jja_day_mean,
        "djf_night_mean": djf_night_mean,
        "jja_day_passed": jja_ok,
        "djf_night_passed": djf_ok,
        "passed": bool(jja_ok and djf_ok),
    }
