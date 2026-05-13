"""
Unit tests for the favorable-propagation criterion.

These tests pin down the methodologically critical behavior:
- Sector convention (north = 0°, clockwise, propagation TOWARD)
- Wind component sign (positive = favorable for that sector)
- Thermal flag is omnidirectional
- Combined criterion is OR (not AND)

If any of these tests fail, the science is broken. They should be the
first thing implemented and the last thing changed.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.favorable import compute_favorable, sector_azimuths, wind_component_along_sector


CLASS_TO_INT = {name: i for i, name in enumerate("ABCDEFG")}


def _config(tmp_path, threshold: float = 2.0) -> dict:
    return {
        "sectors": {"count": 18},
        "stability": {"primary": "richardson"},
        "favorable": {
            "wind_threshold_ms": threshold,
            "stable_classes": ["E", "F", "G"],
            "output_path": str(tmp_path / "favorable.zarr"),
        },
    }


def _single_cell_ds(u: float, v: float) -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=1, freq="h")
    return xr.Dataset(
        {
            "u10": (("time", "cell"), np.array([[u]], dtype=np.float32)),
            "v10": (("time", "cell"), np.array([[v]], dtype=np.float32)),
        },
        coords={"time": time, "cell": [0]},
    )


def _single_cell_stability(class_name: str) -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=1, freq="h")
    return xr.Dataset(
        {
            "stability_ri": (
                ("time", "cell"),
                np.array([[CLASS_TO_INT[class_name]]], dtype=np.int8),
            )
        },
        coords={"time": time, "cell": [0]},
    )


def test_sector_zero_is_north():
    """Sector 0 should be centered on azimuth 0° (north)."""
    sectors = sector_azimuths(18)

    assert sectors.dims == ("sector",)
    assert sectors["sector"].values.tolist() == list(range(18))
    assert sectors.values.tolist() == list(range(0, 360, 20))


def test_west_wind_favors_eastward_propagation():
    """Wind from the west (positive u) should mark sector 90° (east) as favorable."""
    u = xr.DataArray([3.0], dims=("time",), coords={"time": [0]})
    v = xr.DataArray([0.0], dims=("time",), coords={"time": [0]})
    azimuths = xr.DataArray(
        [0.0, 90.0, 180.0, 270.0],
        dims=("sector",),
        coords={"sector": [0, 1, 2, 3]},
    )

    component = wind_component_along_sector(u, v, azimuths)

    assert float(component.sel(sector=1).item()) == pytest.approx(3.0)
    assert float(component.sel(sector=3).item()) == pytest.approx(-3.0)
    assert float(component.sel(sector=0).item()) == pytest.approx(0.0, abs=1e-12)
    assert float(component.sel(sector=2).item()) == pytest.approx(0.0, abs=1e-12)


def test_below_threshold_wind_not_favorable_unless_stable(tmp_path):
    """Light wind (below threshold) should be wind-unfavorable, but stable
    stratification still flags all sectors as thermally favorable."""
    out = compute_favorable(
        _single_cell_ds(u=1.0, v=0.0),
        _single_cell_stability("E"),
        _config(tmp_path),
    )

    assert not bool(out["favorable_wind"].any())
    assert bool(out["favorable_thermal"].item())
    assert bool(out["favorable"].all())


def test_stable_stratification_favors_all_sectors(tmp_path):
    """When atmosphere is stable, all 18 sectors should be favorable
    regardless of wind direction."""
    out = compute_favorable(
        _single_cell_ds(u=0.0, v=0.0),
        _single_cell_stability("F"),
        _config(tmp_path),
    )

    assert bool(out["favorable_thermal"].item())
    assert out["favorable"].sizes["sector"] == 18
    assert bool(out["favorable"].all())


def test_combined_criterion_is_or_not_and(tmp_path):
    """An hour with strong wind in one sector should mark that sector as
    favorable even if stratification is unstable (lapse)."""
    out = compute_favorable(
        _single_cell_ds(u=3.0, v=0.0),
        _single_cell_stability("D"),
        _config(tmp_path),
    )

    assert not bool(out["favorable_thermal"].item())
    assert bool(out["favorable"].sel(sector=4).item())  # 80°, near east
    assert bool(out["favorable"].sel(sector=5).item())  # 100°, near east
    assert not bool(out["favorable"].sel(sector=0).item())
    assert not bool(out["favorable"].sel(sector=9).item())


def test_wind_threshold_is_configurable(tmp_path):
    """Changing the threshold parameter should change which sectors are flagged."""
    ds = _single_cell_ds(u=1.5, v=0.0)
    stability = _single_cell_stability("D")

    low_threshold = compute_favorable(ds, stability, _config(tmp_path / "low", threshold=1.0))
    high_threshold = compute_favorable(ds, stability, _config(tmp_path / "high", threshold=2.0))

    assert bool(low_threshold["favorable_wind"].sel(sector=4).item())
    assert not bool(high_threshold["favorable_wind"].sel(sector=4).item())
