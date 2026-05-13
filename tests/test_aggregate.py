import numpy as np
import pandas as pd
import xarray as xr

from src.aggregate import aggregate_to_climatology, assign_period, assign_season, write_netcdf


SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

PERIODS = {
    "day": {"start_hour": 7, "end_hour": 19},
    "evening": {"start_hour": 19, "end_hour": 23},
    "night": {"start_hour": 23, "end_hour": 7},
}


def test_assign_period_handles_midnight_wrap():
    local_time = xr.DataArray(
        pd.to_datetime(
            [
                "2020-01-01 22:00",
                "2020-01-01 23:00",
                "2020-01-02 00:00",
                "2020-01-02 06:00",
                "2020-01-02 07:00",
            ]
        ),
        dims=("time",),
    )

    period = assign_period(local_time, PERIODS)

    assert period.values.tolist() == ["evening", "night", "night", "night", "day"]


def test_assign_season_maps_months():
    local_time = xr.DataArray(
        pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]),
        dims=("time",),
    )

    season = assign_season(local_time, SEASONS)

    assert season.values.tolist() == ["DJF", "MAM", "JJA", "SON"]


def test_aggregate_to_climatology_shapes_and_counts():
    local_times = []
    for month in [1, 4, 7, 10]:
        for hour in [7, 22, 23]:
            local_times.append(pd.Timestamp(year=2020, month=month, day=1, hour=hour))
    time = pd.DatetimeIndex(local_times)

    wind = np.zeros((len(time), 1, 2), dtype=bool)
    thermal = np.zeros((len(time), 1), dtype=bool)
    wind[:, :, 0] = True
    thermal[2::3, :] = True
    favorable = wind | thermal[:, :, np.newaxis]

    ds = xr.Dataset(
        {
            "favorable": (("time", "cell", "sector"), favorable),
            "favorable_wind": (("time", "cell", "sector"), wind),
            "favorable_thermal": (("time", "cell"), thermal),
        },
        coords={
            "time": time,
            "local_time": ("time", time),
            "latitude": ("cell", [60.0]),
            "longitude": ("cell", [30.0]),
            "sector": [0, 1],
            "sector_azimuth_deg": ("sector", [0.0, 20.0]),
        },
    )

    out = aggregate_to_climatology(ds, {"seasons": SEASONS, "periods": PERIODS})

    assert out["p_favorable"].dims == ("cell", "sector", "season", "period")
    assert out["p_favorable_wind"].dims == ("cell", "sector", "season", "period")
    assert out["p_favorable_thermal"].dims == ("cell", "season", "period")
    assert out["n_samples"].dims == ("cell", "season", "period")
    assert out["n_samples"].dtype == np.int32
    assert bool((out["n_samples"] == 1).all())
    assert float(out["p_favorable_wind"].sel(season="DJF", period="day", sector=0).isel(cell=0)) == 1.0
    assert float(out["p_favorable_thermal"].sel(season="DJF", period="night").isel(cell=0)) == 1.0


def test_write_netcdf_uses_public_metadata(tmp_path):
    ds = xr.Dataset(
        {
            "p_favorable": (
                ("cell", "sector", "season", "period"),
                np.zeros((1, 1, 1, 1), dtype=np.float32),
            ),
            "p_favorable_wind": (
                ("cell", "sector", "season", "period"),
                np.zeros((1, 1, 1, 1), dtype=np.float32),
            ),
            "p_favorable_thermal": (
                ("cell", "season", "period"),
                np.zeros((1, 1, 1), dtype=np.float32),
            ),
            "n_samples": (("cell", "season", "period"), np.ones((1, 1, 1), dtype=np.int32)),
        },
        coords={"cell": [0], "sector": [0], "season": ["DJF"], "period": ["night"]},
    )

    path = tmp_path / "p_favorable_spb.nc"
    write_netcdf(ds, path, {"title": "old title", "institution": "old institution"})

    reopened = xr.open_dataset(path)
    try:
        assert reopened.attrs["title"].startswith("Climatological probability")
        assert reopened.attrs["institution"] == "RSHU (Russian State Hydrometeorological University)"
        assert reopened.attrs["Conventions"] == "CF-1.10"
        assert reopened["p_favorable"].attrs["units"] == "1"
    finally:
        reopened.close()
