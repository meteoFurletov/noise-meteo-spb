import numpy as np
import xarray as xr

from src.stability.validation import agreement_statistics, find_nearest_era5_cell


def test_find_nearest_era5_cell_returns_coordinate_values():
    ds = xr.Dataset(coords={"latitude": [59.5, 60.0], "longitude": [30.0, 30.5]})

    cell = find_nearest_era5_cell(ds, 59.95, 30.45)

    assert cell == {"latitude": 60.0, "longitude": 30.5}


def test_agreement_statistics_confusion_and_rates():
    paired = xr.Dataset(
        {
            "ri_sounding": (("time",), [-0.1, 0.2, 0.3, 0.0, np.nan]),
            "ri_era5": (("time",), [-0.2, 0.4, 0.0, 0.2, 0.3]),
        },
        coords={"time": np.arange("2020-01", "2020-06", dtype="datetime64[M]")},
    )

    stats = agreement_statistics(paired, stable_threshold=0.1)

    assert stats["n"] == 4
    assert stats["confusion_matrix"].tolist() == [[1, 1], [1, 1]]
    assert stats["hit_rate"] == 0.5
    assert stats["false_alarm_rate"] == 0.5
    assert "DJF" in stats["seasonal"]
