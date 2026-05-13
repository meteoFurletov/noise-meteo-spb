import numpy as np
import xarray as xr

from src.case_study.grid import make_receiver_grid
from src.case_study.propagation import (
    compute_cmet,
    compute_levels_three_modes,
    derive_c0_from_pfav,
)
from src.case_study.road import define_road_segment
from src.viz import to_display_angle


def test_to_display_angle_flips_propagation_to_meteorological_from_direction():
    assert to_display_angle(60.0) == 240.0
    assert np.allclose(to_display_angle(np.asarray([0.0, 180.0, 340.0])), [180.0, 0.0, 160.0])


def test_cmet_formula_threshold_and_far_field():
    distances = np.asarray([10.0, 20.0, 100.0])
    cmet = compute_cmet(distances, hs=0.5, hr=1.5, c0_db=2.0)

    assert np.allclose(cmet, [0.0, 0.0, 1.6])


def test_derive_c0_from_pfav_two_state_endpoints():
    c0 = derive_c0_from_pfav(np.asarray([1.0, 0.0]), distance_m=500.0)

    assert np.allclose(c0, [0.0, 9.0])


def test_road_and_receiver_grid_return_distances_and_azimuths():
    road = define_road_segment(59.999980, 30.476457, 90.0, 40.0)
    grid_points, distances, azimuths = make_receiver_grid(road, half_width_m=20.0, spacing_m=20.0)

    assert grid_points.shape == (9, 2)
    assert distances.shape == (9,)
    assert azimuths.shape == (9,)
    assert np.isclose(distances.min(), 0.0)
    assert np.all((azimuths >= 0.0) & (azimuths < 360.0))


def test_compute_levels_three_modes_uses_sector_climatology():
    ds = xr.Dataset(
        {
            "p_favorable": (
                ("cell", "sector", "season", "period"),
                np.asarray([[[[1.0]], [[0.5]], [[0.0]], [[0.25]]]], dtype=np.float32),
            ),
            "n_samples": (("cell", "season", "period"), np.asarray([[[10]]], dtype=np.int32)),
        },
        coords={
            "cell": [0],
            "sector": [0, 1, 2, 3],
            "sector_az": ("sector", [0.0, 90.0, 180.0, 270.0]),
            "season": ["DJF"],
            "period": ["day"],
            "cell_lat": ("cell", [60.0]),
            "cell_lon": ("cell", [30.5]),
        },
    )

    levels = compute_levels_three_modes(
        np.asarray([100.0, 100.0]),
        np.asarray([0.0, 180.0]),
        0,
        ds,
        {"intensity_vph": 1000, "hgv_fraction": 0.4, "speed_kmh": 60, "surface": "asphalt"},
        {"ground_factor": 0.5, "freq_hz": 1000, "T_celsius": 10, "RH_pct": 75, "p_kPa": 101.3},
    )

    assert np.allclose(levels["static"], levels["conservative"])
    assert levels["climatology"][0] > levels["climatology"][1]
    assert levels["climatology"][0] == levels["static"][0]
