"""Bulk Richardson number Ri_b on the 2 m → 100 m AGL layer.

Reference: §4 of the Step 2 v2 specification.

    Ri_b = (g / θ̄_v) · (θ_{v,2} − θ_{v,1}) · (z_2 − z_1) / ((u_2 − u_1)² + (v_2 − v_1)²)

with θ̄_v the arithmetic mean of the lower (2 m) and upper (100 m AGL)
virtual potential temperatures. The temperature at 100 m AGL comes from
``interpolate_t100`` (case A/B principled, case C dry-adiabatic flagged).

The module returns a self-contained xarray.Dataset of the diagnostic
fields so the runner can write them directly. It does no I/O.
"""

from __future__ import annotations

import xarray as xr

from src.stability.thermodynamics import (
    GRAVITY_M_PER_S2,
    barometric_pressure,
    height_agl_from_pressure_geopotential,
    interpolate_t100,
    virtual_potential_temperature_2m,
    virtual_potential_temperature_at_height,
)


def compute(ds: xr.Dataset, cfg: dict) -> xr.Dataset:
    """Compute Ri_b and companion diagnostics from the v2 cache.

    Parameters
    ----------
    ds
        ERA5 v2 dataset. Must contain at minimum: ``t2m``, ``d2m``, ``sp``,
        ``u10``, ``v10``, ``u100``, ``v100`` (single level) and ``t``, ``z``
        on the ``level`` dimension (pressure levels in hPa), plus the
        static ``z_surface`` (m²/s²).
    cfg
        The top-level configuration dict (``configs/spb_default.yaml``
        loaded as YAML). Reads ``stability.Ri_b`` and
        ``stability.t100_interpolation``.

    Returns
    -------
    xr.Dataset
        Variables ``Ri_b``, ``T_100``, ``stable_Ri``,
        ``qc_subterranean_1000hPa``, ``qc_t100_fallback``, all over
        (time, latitude, longitude).
    """
    ri_cfg = cfg["stability"]["Ri_b"]
    t100_cfg = cfg["stability"]["t100_interpolation"]
    z_lower = float(ri_cfg["z_lower_m"])
    z_upper = float(ri_cfg["z_upper_m"])
    threshold = float(ri_cfg["threshold"])
    min_shear_sq = float(ri_cfg["min_shear_sq"])
    lapse = float(t100_cfg["dry_adiabatic_lapse_K_per_m"])

    # Heights AGL of each pressure level, per (time, lat, lon).
    z_agl = height_agl_from_pressure_geopotential(ds["z"], ds["z_surface"])

    # qc_subterranean_1000hPa: pressure-level "below ground" diagnostic.
    qc_subterranean = (z_agl.sel(level=1000) < 0).astype(bool)
    qc_subterranean.name = "qc_subterranean_1000hPa"

    # T at 100 m AGL with the documented fallback hierarchy.
    T_100, qc_t100_fallback = interpolate_t100(
        t2m=ds["t2m"],
        t_pressure=ds["t"],
        z_pressure_agl=z_agl,
        target_height_m=z_upper,
        lower_reference_height_m=z_lower,
        dry_adiabatic_lapse_K_per_m=lapse,
    )

    # Virtual potential temperature at the lower (2 m) level. The companion
    # specific humidity is reused at the upper level (constant-q assumption).
    theta_v_lower, q_lower = virtual_potential_temperature_2m(
        t2m=ds["t2m"], d2m=ds["d2m"], sp=ds["sp"]
    )

    # Virtual temperature at the lower level — needed for the barometric
    # pressure at 100 m.
    Tv_lower = ds["t2m"] * (1.0 + 0.61 * q_lower)

    # Pressure at 100 m via the barometric formula with the constant
    # lower-level virtual temperature as the scale-height temperature.
    p_upper = barometric_pressure(
        surface_pressure_Pa=ds["sp"],
        height_m=z_upper - z_lower,
        virtual_temperature_K=Tv_lower,
    )

    theta_v_upper = virtual_potential_temperature_at_height(
        T_height_K=T_100, specific_humidity_q=q_lower, pressure_Pa_at_height=p_upper
    )

    # Wind shear between the 10 m vector and the 100 m vector. We accept the
    # small approximation of using 10 m wind in place of 2 m wind (the
    # spec's documented choice — surface roughness makes "true" 2 m wind
    # essentially zero and would dominate Ri_b spuriously).
    du = ds["u100"] - ds["u10"]
    dv = ds["v100"] - ds["v10"]
    shear_sq = du * du + dv * dv

    theta_v_mean = 0.5 * (theta_v_lower + theta_v_upper)
    numerator = GRAVITY_M_PER_S2 * (theta_v_upper - theta_v_lower) * (z_upper - z_lower)
    ri_b = numerator / (theta_v_mean * shear_sq)

    # Where the wind field is essentially uniform (vanishing shear), Ri_b is
    # not defined; set to NaN rather than letting it explode.
    ri_b = ri_b.where(shear_sq >= min_shear_sq)
    ri_b = ri_b.astype("float32")
    ri_b.name = "Ri_b"
    ri_b.attrs.update(
        long_name="bulk Richardson number on 2-100 m AGL layer",
        units="1",
        method="theta_v formulation; T_100 interpolated in geopotential height",
    )

    stable_Ri = ((ri_b >= threshold) & ri_b.notnull()).astype(bool)
    stable_Ri.name = "stable_Ri"
    stable_Ri.attrs.update(
        long_name=f"Ri_b ≥ {threshold:g}",
        method="bulk Richardson threshold",
        qc_dependencies="qc_t100_fallback",
    )

    T_100.attrs.update(
        long_name="temperature at 100 m AGL",
        units="K",
        method="linear interpolation in geopotential height across ERA5 pressure levels",
        qc_dependencies="qc_t100_fallback",
    )
    qc_t100_fallback.attrs.update(
        long_name="t100 interpolation used the dry-adiabatic fallback (case C)",
    )
    qc_subterranean.attrs.update(
        long_name="1000 hPa pressure level is below the surface",
        method="z(1000 hPa) < z_surface from ERA5 static field",
    )

    return xr.Dataset(
        {
            "Ri_b": ri_b,
            "T_100": T_100,
            "stable_Ri": stable_Ri,
            "qc_subterranean_1000hPa": qc_subterranean,
            "qc_t100_fallback": qc_t100_fallback,
        }
    )
