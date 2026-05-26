"""Tests for Step 4 v2 aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.aggregate import aggregate_to_climatology, assign_period, assign_season


SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

PERIODS = {
    "day": [7, 19],
    "evening": [19, 23],
    "night": [23, 7],
}


def _utc_hours(n_days: int = 30) -> xr.DataArray:
    times = pd.date_range("2020-01-01T00:00", periods=n_days * 24, freq="h")
    return xr.DataArray(times, dims=("time",), coords={"time": times})


def test_assign_season_maps_months():
    time = xr.DataArray(
        pd.to_datetime(["2020-01-15", "2020-04-15", "2020-07-15", "2020-10-15"]),
        dims=("time",),
    )
    season = assign_season(time, SEASONS)
    assert season.values.tolist() == ["DJF", "MAM", "JJA", "SON"]


def test_assign_period_applies_local_offset_and_wraps_midnight():
    # UTC hours that, with +3 offset, hit each Lden period edge in local time.
    utc_times = pd.to_datetime(
        [
            "2020-01-01 04:00",  # local 07 -> day
            "2020-01-01 15:00",  # local 18 -> day
            "2020-01-01 16:00",  # local 19 -> evening
            "2020-01-01 19:00",  # local 22 -> evening
            "2020-01-01 20:00",  # local 23 -> night
            "2020-01-01 23:00",  # local 02 next-day -> night
            "2020-01-02 03:00",  # local 06 -> night
        ]
    )
    time = xr.DataArray(utc_times, dims=("time",), coords={"time": utc_times})
    period = assign_period(time, PERIODS, tz_offset_hours=3)
    assert period.values.tolist() == [
        "day",
        "day",
        "evening",
        "evening",
        "night",
        "night",
        "night",
    ]


def test_period_coverage_over_a_day_matches_lden_durations():
    time = _utc_hours(n_days=1)
    period = assign_period(time, PERIODS, tz_offset_hours=3)
    counts = pd.Series(period.values).value_counts().to_dict()
    assert counts["day"] == 12
    assert counts["evening"] == 4
    assert counts["night"] == 8


def test_aggregate_to_climatology_schema_and_values():
    # Synthetic 1-month dataset (Jan 2020, 24 h × 30 d = 720 h),
    # 2 × 3 grid, 4 sectors — small but realistic shape.
    time = pd.date_range("2020-01-01T00:00", periods=24 * 30, freq="h")
    lat = np.array([60.0, 60.25], dtype="float64")
    lon = np.array([29.5, 30.0, 30.5], dtype="float64")
    sector = np.arange(4, dtype="int16")
    shape_sector = (time.size, lat.size, lon.size, sector.size)
    shape_thermal = (time.size, lat.size, lon.size)

    wind = np.zeros(shape_sector, dtype=bool)
    wind[:, :, :, 0] = True  # sector 0 always favorable
    thermal_ri = np.zeros(shape_thermal, dtype=bool)
    thermal_ri[((time.hour + 3) % 24 >= 23) | ((time.hour + 3) % 24 < 7), :, :] = True
    favorable = wind | thermal_ri[..., np.newaxis]

    ds = xr.Dataset(
        {
            "favorable": (("time", "latitude", "longitude", "sector"), favorable),
            "favorable_wind": (("time", "latitude", "longitude", "sector"), wind),
            "favorable_thermal_Ri": (("time", "latitude", "longitude"), thermal_ri),
            "favorable_thermal_Ri_strict": (("time", "latitude", "longitude"),
                                            np.zeros(shape_thermal, dtype=bool)),
            "favorable_thermal_L_strict": (("time", "latitude", "longitude"),
                                           np.zeros(shape_thermal, dtype=bool)),
            "favorable_thermal_L_moderate": (("time", "latitude", "longitude"),
                                             np.zeros(shape_thermal, dtype=bool)),
            "favorable_thermal_PG": (("time", "latitude", "longitude"),
                                     np.zeros(shape_thermal, dtype=bool)),
        },
        coords={
            "time": time,
            "latitude": lat,
            "longitude": lon,
            "sector": sector,
        },
    )

    out = aggregate_to_climatology(ds, SEASONS, PERIODS, tz_offset_hours=3)

    # Schema.
    assert out["p_favorable"].dims == ("latitude", "longitude", "sector", "season", "period")
    assert out["p_favorable_wind"].dims == ("latitude", "longitude", "sector", "season", "period")
    assert out["p_favorable_thermal_Ri"].dims == ("latitude", "longitude", "season", "period")
    assert out["n_samples"].dims == ("latitude", "longitude", "season", "period")
    assert out["n_samples"].dtype == np.int32
    assert out["p_favorable"].dtype == np.float32

    # Wind favorable in sector 0 should be 1.0 in DJF for every period; 0 elsewhere.
    p_wind = out["p_favorable_wind"].sel(season="DJF")
    assert float(p_wind.sel(period="day").isel(latitude=0, longitude=0, sector=0)) == 1.0
    assert float(p_wind.sel(period="day").isel(latitude=0, longitude=0, sector=1)) == 0.0

    # Thermal-Ri active 23:00–07:00 local: p == 1 in 'night', 0 in 'day'/'evening'.
    p_th = out["p_favorable_thermal_Ri"].sel(season="DJF")
    assert float(p_th.sel(period="night").isel(latitude=0, longitude=0)) == pytest.approx(1.0)
    assert float(p_th.sel(period="day").isel(latitude=0, longitude=0)) == pytest.approx(0.0)
    assert float(p_th.sel(period="evening").isel(latitude=0, longitude=0)) == pytest.approx(0.0)

    # n_samples for DJF/night should equal the number of local-night hours in Jan.
    expected_night = int(
        (((time.hour + 3) % 24 >= 23) | ((time.hour + 3) % 24 < 7)).sum()
    )
    assert int(out["n_samples"].sel(season="DJF", period="night").isel(latitude=0, longitude=0)) == expected_night

    # Values in [0, 1].
    for name in out.data_vars:
        if name.startswith("p_"):
            assert float(out[name].min()) >= 0.0
            assert float(out[name].max()) <= 1.0
