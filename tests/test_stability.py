"""
Unit tests for stability classification.

These tests pin down:
- Richardson formula uses correct sign convention (positive Ri = stable)
- Class boundaries match config exactly
- Pasquill table lookup matches Stull (1988) Table 9.4 reference values
- Solar zenith calculation matches metpy / known reference values

These tests catch silent sign flips and off-by-one bin errors that would
quietly corrupt the climatology.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from src.config import load_config
from src.stability import (
    _classify_richardson_values,
    pasquill_class,
    richardson_bulk,
    solar_zenith_angle,
)


def _config() -> dict:
    return load_config("configs/spb_default.yaml")


def _richardson_ds(t2m: float, t_upper: float, u10: float = 1.0, u_upper: float = 6.0) -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=1, freq="h")
    return xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), np.full((1, 1, 1), t2m)),
            "u10": (("time", "latitude", "longitude"), np.full((1, 1, 1), u10)),
            "v10": (("time", "latitude", "longitude"), np.zeros((1, 1, 1))),
            "t": (
                ("time", "pressure_level", "latitude", "longitude"),
                np.full((1, 1, 1, 1), t_upper),
            ),
            "u": (
                ("time", "pressure_level", "latitude", "longitude"),
                np.full((1, 1, 1, 1), u_upper),
            ),
            "v": (
                ("time", "pressure_level", "latitude", "longitude"),
                np.zeros((1, 1, 1, 1)),
            ),
        },
        coords={
            "time": time,
            "latitude": [60.0],
            "longitude": [30.0],
            "pressure_level": [1000.0],
        },
    )


def _pasquill_ds(time: str, wind_speed: float, cloud_cover: float) -> xr.Dataset:
    return xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), np.array([[[wind_speed]]])),
            "v10": (("time", "latitude", "longitude"), np.zeros((1, 1, 1))),
            "tcc": (("time", "latitude", "longitude"), np.array([[[cloud_cover]]])),
        },
        coords={
            "time": [np.datetime64(time)],
            "latitude": [60.0],
            "longitude": [30.0],
        },
    )


def test_positive_richardson_is_stable():
    """Positive bulk Ri must map to stability classes E/F/G."""
    ri = richardson_bulk(_richardson_ds(t2m=280.0, t_upper=281.0), _config())
    stability = _classify_richardson_values(ri, _config())

    assert int(stability.item()) >= 4


def test_negative_richardson_is_unstable():
    """Negative bulk Ri must map to A/B/C."""
    ri = richardson_bulk(_richardson_ds(t2m=281.0, t_upper=280.0), _config())
    stability = _classify_richardson_values(ri, _config())

    assert int(stability.item()) <= 2


def test_richardson_class_boundaries_match_config():
    """Class assignment must respect the boundaries in spb_default.yaml."""
    config = _config()
    boundaries = [entry["ri_max"] for entry in config["stability"]["richardson"]["classes"]]
    da = xr.DataArray(boundaries, dims=("sample",))

    classes = _classify_richardson_values(da, config)

    assert classes.values.tolist() == list(range(7))


def test_pasquill_high_wind_clear_day_is_class_C_or_D():
    """Reference case from Stull (1988) Table 9.4: 6 m/s wind, strong
    insolation, clear sky — should give class C (slightly unstable)."""
    ds = _pasquill_ds("2020-06-21T09:00:00", wind_speed=6.0, cloud_cover=0.0)
    stability = pasquill_class(ds, _config())

    assert int(stability.item()) in {2, 3}


def test_pasquill_calm_clear_night_is_class_F_or_G():
    """Reference case: calm wind, clear night — should give class F or G."""
    ds = _pasquill_ds("2020-12-21T22:00:00", wind_speed=0.5, cloud_cover=0.0)
    stability = pasquill_class(ds, _config())

    assert int(stability.item()) in {5, 6}


def test_solar_zenith_at_pulkovo_summer_solstice_noon():
    """Solar zenith at Pulkovo (~60°N) at summer solstice noon should be
    about 90° - (90° - 60° + 23.4°) ≈ 36.6°."""
    ds = xr.Dataset(
        coords={
            "time": [np.datetime64("2020-06-21T10:00:00")],
            "latitude": [59.79],
            "longitude": [30.27],
        }
    )
    zenith = solar_zenith_angle(ds)

    assert float(zenith.item()) == pytest.approx(36.6, abs=2.0)
