"""Unit tests for Step 3 v2 favorable-propagation module.

Pins the methodologically critical behaviour:
- Sector convention (ISO 9613-2: azimuth measured clockwise from north,
  pointing TOWARD the receiver).
- Wind component sign (positive = wind blows toward receiver).
- Thermal flag is per-cell (broadcasts over sectors when OR'd into
  the combined `favorable`).
- Cascade logic (Ri NaN -> PG, never -> 1/L).

If any of these tests fail the science is broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.favorable.thermal import compute_thermal_variants
from src.favorable.wind import (
    compute_favorable_wind,
    sector_azimuths_iso_deg,
    wind_component_along_sector,
)


def _wind_da(u_vals, v_vals):
    n = len(u_vals)
    time = pd.date_range("2020-01-01", periods=n, freq="h")
    u = xr.DataArray(
        np.array(u_vals, dtype=np.float32).reshape(n, 1, 1),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [60.0], "longitude": [30.0]},
    )
    v = xr.DataArray(
        np.array(v_vals, dtype=np.float32).reshape(n, 1, 1),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [60.0], "longitude": [30.0]},
    )
    return u, v


# ─── sector convention ────────────────────────────────────────────────────


def test_sector_zero_is_north():
    az = sector_azimuths_iso_deg(18)
    assert az.dims == ("sector",)
    assert az["sector"].values.tolist() == list(range(18))
    assert az.values.tolist() == [float(20 * k) for k in range(18)]


def test_west_wind_favors_eastward_propagation():
    """u10 > 0 means wind blows toward the east (azimuth 90°)."""
    u = xr.DataArray([3.0], dims=("time",))
    v = xr.DataArray([0.0], dims=("time",))
    az = xr.DataArray([0.0, 90.0, 180.0, 270.0], dims=("sector",))
    comp = wind_component_along_sector(u, v, az)
    assert float(comp.sel(sector=1).item()) == pytest.approx(3.0)
    assert float(comp.sel(sector=3).item()) == pytest.approx(-3.0)
    assert float(comp.sel(sector=0).item()) == pytest.approx(0.0, abs=1e-6)
    assert float(comp.sel(sector=2).item()) == pytest.approx(0.0, abs=1e-6)


def test_compute_favorable_wind_threshold():
    u, v = _wind_da([3.0, 1.5], [0.0, 0.0])
    fw = compute_favorable_wind(u, v, wind_threshold_ms=2.0, n_sectors=18)
    # Hour 0: u=3, sector az=80° (k=4) has u_par = 3*sin80 ≈ 2.95 >= 2 -> True.
    assert bool(fw.isel(time=0).sel(sector=4).item()) is True
    # Hour 0: sector az=260° (k=13) has u_par = 3*sin260 ≈ -2.95 -> False.
    assert bool(fw.isel(time=0).sel(sector=13).item()) is False
    # Hour 1: u=1.5 < 2, all sectors should be False at threshold 2.
    assert not bool(fw.isel(time=1).any().item())


def test_threshold_is_configurable():
    u, v = _wind_da([1.5], [0.0])
    fw_low = compute_favorable_wind(u, v, wind_threshold_ms=1.0)
    fw_high = compute_favorable_wind(u, v, wind_threshold_ms=2.0)
    assert bool(fw_low.sel(sector=4).any().item()) is True
    assert bool(fw_high.sel(sector=4).any().item()) is False


# ─── thermal cascade ──────────────────────────────────────────────────────


def _stab_ds(*, Ri_b, inv_L, stable_Ri, stable_PG, qc_low_ustar):
    n = len(Ri_b)
    time = pd.date_range("2020-01-01", periods=n, freq="h")
    coords = {"time": time, "latitude": [60.0], "longitude": [30.0]}
    shape = (n, 1, 1)
    return xr.Dataset(
        {
            "Ri_b": (("time", "latitude", "longitude"),
                     np.array(Ri_b, dtype=np.float32).reshape(shape)),
            "inv_L": (("time", "latitude", "longitude"),
                      np.array(inv_L, dtype=np.float32).reshape(shape)),
            "stable_Ri": (("time", "latitude", "longitude"),
                          np.array(stable_Ri, dtype=bool).reshape(shape)),
            "stable_PG": (("time", "latitude", "longitude"),
                          np.array(stable_PG, dtype=bool).reshape(shape)),
            "qc_low_ustar": (("time", "latitude", "longitude"),
                             np.array(qc_low_ustar, dtype=bool).reshape(shape)),
        },
        coords=coords,
    )


def test_cascade_substitutes_PG_only_where_Ri_is_NaN():
    stab = _stab_ds(
        Ri_b=[0.2, np.nan, np.nan, 0.05],
        inv_L=[0.06, 0.06, np.nan, 0.0],
        stable_Ri=[True, False, False, False],   # False at NaN per Phase A encoding
        stable_PG=[False, True, False, True],
        qc_low_ustar=[False, False, True, False],
    )
    out = compute_thermal_variants(stab)
    primary = out["favorable_thermal_Ri"].values.flatten().tolist()
    # Hour 0: Ri defined and True -> True.
    # Hour 1: Ri NaN, PG True -> True (cascade).
    # Hour 2: Ri NaN, PG False -> False.
    # Hour 3: Ri defined and False -> False (does NOT promote to True via PG).
    assert primary == [True, True, False, False]

    cascade = out["cascade_used"].values.flatten().tolist()
    assert cascade == [False, True, True, False]


def test_strict_ri_does_not_cascade():
    stab = _stab_ds(
        Ri_b=[np.nan],
        inv_L=[0.06],
        stable_Ri=[False],
        stable_PG=[True],
        qc_low_ustar=[False],
    )
    out = compute_thermal_variants(stab)
    assert bool(out["favorable_thermal_Ri_strict"].item()) is False
    assert bool(out["favorable_thermal_Ri"].item()) is True


def test_L_strict_and_moderate_thresholds():
    stab = _stab_ds(
        Ri_b=[0.2, 0.2, 0.2],
        inv_L=[0.03, 0.07, 0.005],
        stable_Ri=[True, True, True],
        stable_PG=[False, False, False],
        qc_low_ustar=[False, False, False],
    )
    out = compute_thermal_variants(stab, L_strict_inv_m=0.05, L_moderate_inv_m=0.01)
    L_strict = out["favorable_thermal_L_strict"].values.flatten().tolist()
    L_mod = out["favorable_thermal_L_moderate"].values.flatten().tolist()
    assert L_strict == [False, True, False]
    assert L_mod == [True, True, False]


def test_L_masked_by_low_ustar():
    stab = _stab_ds(
        Ri_b=[0.2],
        inv_L=[0.5],       # would be very stable
        stable_Ri=[True],
        stable_PG=[False],
        qc_low_ustar=[True],   # masked
    )
    out = compute_thermal_variants(stab)
    assert bool(out["favorable_thermal_L_strict"].item()) is False
    assert bool(out["favorable_thermal_L_moderate"].item()) is False


# ─── combined flag broadcast ──────────────────────────────────────────────


def test_thermal_true_marks_all_sectors_favorable():
    u, v = _wind_da([0.0], [0.0])
    fw = compute_favorable_wind(u, v, wind_threshold_ms=2.0)
    thermal = xr.DataArray(
        np.array([[[True]]], dtype=bool),
        dims=("time", "latitude", "longitude"),
        coords=fw.drop_vars("sector").coords if "sector" in fw.coords else None,
    )
    combined = fw | thermal
    assert bool(combined.all().item()) is True
    assert combined.sizes["sector"] == 18


def test_combined_criterion_is_or_not_and():
    """Strong wind in one sector + unstable atmosphere => that sector is favorable."""
    u, v = _wind_da([3.0], [0.0])
    fw = compute_favorable_wind(u, v, wind_threshold_ms=2.0)
    thermal = xr.DataArray(
        np.array([[[False]]], dtype=bool),
        dims=("time", "latitude", "longitude"),
    )
    combined = fw | thermal
    assert bool(combined.isel(time=0).sel(sector=4).item()) is True   # 80°
    assert bool(combined.isel(time=0).sel(sector=13).item()) is False  # 260°


# ─── sign-convention regression ───────────────────────────────────────────


def test_sign_convention_north_wind_negative_v():
    """Wind from the north (v < 0) favors southward (azimuth 180°) propagation."""
    u, v = _wind_da([0.0], [-3.0])   # v negative => wind blows toward south
    fw = compute_favorable_wind(u, v, wind_threshold_ms=2.0)
    assert bool(fw.isel(time=0).sel(sector=9).item()) is True   # az 180°
    assert bool(fw.isel(time=0).sel(sector=0).item()) is False  # az 0° (north)
