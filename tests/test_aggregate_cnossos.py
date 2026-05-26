"""Tests for Phase G — CNOSSOS-EU faithful aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.aggregate import (
    aggregate_cnossos_to_climatology,
    aggregate_to_climatology,
    build_cnossos_favorable,
)

SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
PERIODS = {"day": [7, 19], "evening": [19, 23], "night": [23, 7]}


def _synthetic_ds(n_days: int = 30) -> xr.Dataset:
    time = pd.date_range("2020-01-01T00:00", periods=24 * n_days, freq="h")
    lat = np.array([60.0, 60.25], dtype="float64")
    lon = np.array([29.5, 30.0, 30.5], dtype="float64")
    sector = np.arange(4, dtype="int16")
    shape_sector = (time.size, lat.size, lon.size, sector.size)
    shape_thermal = (time.size, lat.size, lon.size)

    rng = np.random.default_rng(0)
    wind = rng.random(shape_sector) < 0.4
    pg = rng.random(shape_thermal) < 0.3
    ri = rng.random(shape_thermal) < 0.5
    fav = wind | ri[..., np.newaxis]

    return xr.Dataset(
        {
            "favorable": (("time", "latitude", "longitude", "sector"), fav),
            "favorable_wind": (("time", "latitude", "longitude", "sector"), wind),
            "favorable_thermal_Ri": (("time", "latitude", "longitude"), ri),
            "favorable_thermal_Ri_strict": (("time", "latitude", "longitude"),
                                            np.zeros(shape_thermal, dtype=bool)),
            "favorable_thermal_L_strict": (("time", "latitude", "longitude"),
                                           np.zeros(shape_thermal, dtype=bool)),
            "favorable_thermal_L_moderate": (("time", "latitude", "longitude"),
                                             np.zeros(shape_thermal, dtype=bool)),
            "favorable_thermal_PG": (("time", "latitude", "longitude"), pg),
        },
        coords={"time": time, "latitude": lat, "longitude": lon, "sector": sector},
    )


def test_build_cnossos_favorable_matches_wind_or_pg():
    ds = _synthetic_ds(n_days=5)
    cn = build_cnossos_favorable(ds)
    expected = ds["favorable_wind"] | ds["favorable_thermal_PG"]
    assert cn.dims == ("time", "latitude", "longitude", "sector")
    assert bool((cn == expected).all().item())


def test_aggregate_cnossos_schema_and_no_nan():
    ds = _synthetic_ds(n_days=30)
    out = aggregate_cnossos_to_climatology(ds, SEASONS, PERIODS, tz_offset_hours=3)

    assert out["p_favorable_cnossos"].dims == (
        "latitude", "longitude", "sector", "season", "period",
    )
    assert out["p_favorable_wind"].dims == (
        "latitude", "longitude", "sector", "season", "period",
    )
    assert out["p_favorable_thermal_PG"].dims == (
        "latitude", "longitude", "season", "period",
    )
    assert out["n_samples"].dims == ("latitude", "longitude", "season", "period")
    assert out["n_samples"].dtype == np.int32
    assert out["p_favorable_cnossos"].dtype == np.float32

    # Synthetic input covers only DJF; non-DJF (season, period) bins are
    # legitimately empty (NaN). Check DJF bins are populated and in [0,1].
    for name in out.data_vars:
        if not name.startswith("p_"):
            continue
        djf = out[name].sel(season="DJF").values
        assert not np.isnan(djf).any(), f"{name} has NaN in DJF (populated) bins"
        assert float(djf.min()) >= 0.0
        assert float(djf.max()) <= 1.0


def test_cnossos_wind_component_matches_primary():
    """Wind probability must be identical between primary and CNOSSOS paths."""
    ds = _synthetic_ds(n_days=30)
    primary = aggregate_to_climatology(ds, SEASONS, PERIODS, tz_offset_hours=3)
    cnossos = aggregate_cnossos_to_climatology(ds, SEASONS, PERIODS, tz_offset_hours=3)
    p = primary["p_favorable_wind"].sel(season="DJF").values
    c = cnossos["p_favorable_wind"].sel(season="DJF").values
    assert float(np.abs(p - c).max()) < 1e-6
