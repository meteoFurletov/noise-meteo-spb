"""Unit tests for the Step 2 v2 stability module.

Each classifier gets a normal-case test and an edge-case test, per §11 of the
specification. Tests run on tiny synthetic datasets — no I/O, no v2 cache
dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import yaml

from src.stability import monin_obukhov, pasquill_turner, richardson, thermodynamics


# ─── Helpers ────────────────────────────────────────────────────────────────

DIMS_3D = ("time", "latitude", "longitude")


@pytest.fixture
def cfg() -> dict:
    return yaml.safe_load(open("configs/spb_default.yaml"))


def _scalar_ds(
    *,
    t2m: float,
    d2m: float,
    sp: float,
    u10: float,
    v10: float,
    u100: float,
    v100: float,
    t_profile_K: list[float],
    z_agl_profile_m: list[float],
    z_surface_m: float = 20.0,
    ishf: float = 0.0,
    ie: float = 0.0,
    iews: float = 0.05,
    inss: float = 0.05,
    fdir: float = 0.0,
    tcc: float = 0.0,
) -> xr.Dataset:
    """One time step, one grid cell, three pressure levels."""
    levels = [1000, 975, 950]
    g = thermodynamics.GRAVITY_M_PER_S2
    # Convert AGL heights (m) → geopotential (m²/s²) given surface elevation.
    z_pl = np.array(z_agl_profile_m) + z_surface_m
    z_geopot = (z_pl * g).reshape(1, 3, 1, 1)
    t_pl = np.array(t_profile_K).reshape(1, 3, 1, 1)

    return xr.Dataset(
        {
            "t2m":  (DIMS_3D, [[[t2m]]]),
            "d2m":  (DIMS_3D, [[[d2m]]]),
            "sp":   (DIMS_3D, [[[sp]]]),
            "u10":  (DIMS_3D, [[[u10]]]),
            "v10":  (DIMS_3D, [[[v10]]]),
            "u100": (DIMS_3D, [[[u100]]]),
            "v100": (DIMS_3D, [[[v100]]]),
            "ishf": (DIMS_3D, [[[ishf]]]),
            "ie":   (DIMS_3D, [[[ie]]]),
            "iews": (DIMS_3D, [[[iews]]]),
            "inss": (DIMS_3D, [[[inss]]]),
            "fdir": (DIMS_3D, [[[fdir]]]),
            "tcc":  (DIMS_3D, [[[tcc]]]),
            "t":    (("time", "level", "latitude", "longitude"), t_pl),
            "z":    (("time", "level", "latitude", "longitude"), z_geopot),
            "z_surface": (("latitude", "longitude"), [[z_surface_m * g]]),
        },
        coords={
            "time": np.array(["2020-07-15T12:00"], dtype="datetime64[ns]"),
            "level": levels,
            "latitude": [60.0],
            "longitude": [30.0],
        },
    )


# ─── Thermodynamics quick smoke ─────────────────────────────────────────────


def test_thermodynamics_magnus_e_at_10C() -> None:
    # Magnus (WMO 17.62/243.12) at T_d = 283.15 K → ~1228 Pa. We use the
    # 17.62/243.12 form which gives 1213.8 Pa — accept ±2 %.
    e = float(thermodynamics.saturation_vapor_pressure_magnus(xr.DataArray(283.15)).values)
    assert 1190.0 <= e <= 1250.0


def test_thermodynamics_t100_three_cases() -> None:
    """t100 interpolation: case A (bracket), B (low pressure level), C (extreme)."""
    t2m = xr.DataArray([288.0], dims=("time",))
    t_press = xr.DataArray(
        [[287.0, 285.0, 283.0]], dims=("time", "level"),
        coords={"level": [1000, 975, 950]},
    )
    # Case A: 100 m AGL is bracketed by 1000 hPa (50 m) and 975 hPa (250 m).
    z_A = xr.DataArray([[50.0, 250.0, 450.0]], dims=("time", "level"),
                       coords={"level": [1000, 975, 950]})
    T_A, qc_A = thermodynamics.interpolate_t100(t2m, t_press, z_A)
    assert qc_A.values[0] == False  # principled interp
    assert abs(float(T_A.values[0]) - 286.5) < 0.01

    # Case B: lowest pressure level already above 100 m AGL.
    z_B = xr.DataArray([[200.0, 400.0, 600.0]], dims=("time", "level"),
                       coords={"level": [1000, 975, 950]})
    T_B, qc_B = thermodynamics.interpolate_t100(t2m, t_press, z_B)
    assert qc_B.values[0] == False  # case B is principled too
    # alpha = (100-2)/(200-2) = 0.4949; T = 288 + 0.4949*(287-288) = 287.51
    assert abs(float(T_B.values[0]) - 287.51) < 0.02

    # Case C: every pressure level is below 100 m AGL (extreme low surface).
    z_C = xr.DataArray([[10.0, 30.0, 50.0]], dims=("time", "level"),
                       coords={"level": [1000, 975, 950]})
    T_C, qc_C = thermodynamics.interpolate_t100(t2m, t_press, z_C)
    assert qc_C.values[0] == True
    # T = 288 - 0.00977 * 98 = 287.043
    assert abs(float(T_C.values[0]) - 287.04) < 0.01


# ─── Richardson ─────────────────────────────────────────────────────────────


def test_richardson_stable_inversion_yields_positive_ri(cfg) -> None:
    """Nocturnal inversion: temperature *increases* with height → Ri_b > 0."""
    ds = _scalar_ds(
        t2m=270.0, d2m=268.0, sp=101300.0,
        u10=1.0, v10=0.5, u100=3.0, v100=1.5,
        # T at 1000/975/950 hPa with AGL of 50/250/450 m → warms upward
        t_profile_K=[272.0, 280.0, 285.0],
        z_agl_profile_m=[50.0, 250.0, 450.0],
        z_surface_m=20.0,
    )
    out = richardson.compute(ds, cfg)
    ri_value = float(out["Ri_b"].values.flatten()[0])
    assert ri_value > cfg["stability"]["Ri_b"]["threshold"]
    assert bool(out["stable_Ri"].values.flatten()[0]) is True
    # Diagnostic propagation
    assert not bool(out["qc_t100_fallback"].values.flatten()[0])
    assert "T_100" in out


def test_richardson_weak_shear_yields_nan(cfg) -> None:
    """Wind ~uniform in height → shear² below floor → Ri_b is NaN."""
    ds = _scalar_ds(
        t2m=288.0, d2m=283.0, sp=101325.0,
        u10=2.0, v10=1.0, u100=2.05, v100=1.02,   # Δu²+Δv² ≈ 0.0029 < 1.0 floor
        t_profile_K=[287.0, 285.0, 283.0],
        z_agl_profile_m=[50.0, 250.0, 450.0],
    )
    out = richardson.compute(ds, cfg)
    ri_value = float(out["Ri_b"].values.flatten()[0])
    assert np.isnan(ri_value)
    # Stable flag must be False when Ri_b is NaN (not None / not propagated).
    assert bool(out["stable_Ri"].values.flatten()[0]) is False


# ─── Monin–Obukhov ──────────────────────────────────────────────────────────


def test_monin_obukhov_sign_convention(cfg) -> None:
    """Upward sensible heat flux (sunny afternoon) → inv_L < 0 (unstable).
    Downward sensible heat flux (clear night)     → inv_L > 0 (stable).
    """
    # Daytime: ECMWF instantaneous_surface_sensible_heat_flux is *downward
    # positive*. A sunny afternoon over land has upward turbulent heat, so
    # in ECMWF's sign convention the value is *negative* (≈ -200 W/m²).
    day = _scalar_ds(
        t2m=298.0, d2m=290.0, sp=101000.0,
        u10=3.0, v10=2.0, u100=6.0, v100=4.0,
        t_profile_K=[297.0, 295.0, 293.0],
        z_agl_profile_m=[50.0, 250.0, 450.0],
        ishf=-200.0, ie=-1.0e-4,
        iews=0.10, inss=0.05,
    )
    night = _scalar_ds(
        t2m=268.0, d2m=266.0, sp=101800.0,
        u10=1.5, v10=0.8, u100=3.0, v100=1.6,
        t_profile_K=[270.0, 271.0, 272.0],
        z_agl_profile_m=[50.0, 250.0, 450.0],
        ishf=30.0, ie=1.0e-5,
        iews=0.02, inss=0.01,
    )
    day_out = monin_obukhov.compute(day, cfg)
    night_out = monin_obukhov.compute(night, cfg)

    inv_L_day = float(day_out["inv_L"].values.flatten()[0])
    inv_L_night = float(night_out["inv_L"].values.flatten()[0])

    assert inv_L_day < 0, f"day inv_L should be negative (unstable), got {inv_L_day}"
    assert inv_L_night > 0, f"night inv_L should be positive (stable), got {inv_L_night}"
    assert not bool(day_out["qc_low_ustar"].values.flatten()[0])
    assert not bool(night_out["qc_low_ustar"].values.flatten()[0])


def test_monin_obukhov_low_ustar_yields_nan_and_qc(cfg) -> None:
    """Near-zero surface stress → u_* < min_ustar → inv_L = NaN, QC flag set."""
    ds = _scalar_ds(
        t2m=288.0, d2m=283.0, sp=101325.0,
        u10=0.1, v10=0.1, u100=0.2, v100=0.1,
        t_profile_K=[287.0, 285.0, 283.0],
        z_agl_profile_m=[50.0, 250.0, 450.0],
        ishf=-5.0, ie=-1.0e-6,
        iews=1e-6, inss=1e-6,   # essentially zero stress
    )
    out = monin_obukhov.compute(ds, cfg)
    assert bool(out["qc_low_ustar"].values.flatten()[0]) is True
    assert np.isnan(float(out["inv_L"].values.flatten()[0]))
    assert bool(out["stable_L"].values.flatten()[0]) is False


# ─── Pasquill–Turner ────────────────────────────────────────────────────────


def test_pasquill_strong_sun_calm_wind_is_class_A(cfg) -> None:
    """Strong direct beam (≥ 700 W/m²) with < 2 m/s wind → class A (=1)."""
    # 700 W/m² × 3600 s = 2.52 MJ/m² accumulated over the previous hour.
    ds = _scalar_ds(
        t2m=298.0, d2m=290.0, sp=101000.0,
        u10=0.8, v10=0.2, u100=2.0, v100=1.0,
        t_profile_K=[297.0, 295.0, 293.0],
        z_agl_profile_m=[50.0, 250.0, 450.0],
        fdir=750.0 * 3600.0, tcc=0.2,
    )
    out = pasquill_turner.compute(ds, cfg)
    pg = int(out["pasquill_class"].values.flatten()[0])
    assert pg == 1, f"expected class A (1), got {pg}"
    assert bool(out["stable_PG"].values.flatten()[0]) is False


def test_pasquill_calm_clear_night_is_class_G(cfg) -> None:
    """Clear sky (tcc < 0.5) at night (fdir ≈ 0) with < 2 m/s wind → class G."""
    ds = _scalar_ds(
        t2m=268.0, d2m=266.0, sp=101800.0,
        u10=0.5, v10=0.2, u100=1.2, v100=0.6,
        t_profile_K=[270.0, 271.0, 272.0],
        z_agl_profile_m=[50.0, 250.0, 450.0],
        fdir=0.0, tcc=0.1,
    )
    out = pasquill_turner.compute(ds, cfg)
    pg = int(out["pasquill_class"].values.flatten()[0])
    assert pg == 7, f"expected class G (7), got {pg}"
    assert bool(out["stable_PG"].values.flatten()[0]) is True
