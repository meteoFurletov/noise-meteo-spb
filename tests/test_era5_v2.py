from pathlib import Path

from src.data.era5 import _build_request, _load_v2_config, _request_filename, describe_dry_run


def _cfg() -> dict:
    return _load_v2_config("configs/spb_default.yaml")


def test_request_filename_static_ignores_year(tmp_path):
    assert _request_filename("static", 2014, tmp_path) == tmp_path / "era5_spb_v2_static.nc"
    assert _request_filename("static", 2024, tmp_path) == tmp_path / "era5_spb_v2_static.nc"


def test_build_pressure_level_request_for_full_year():
    request = _build_request("pressure_levels", 2014, _cfg())

    assert request["variable"] == [
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "geopotential",
    ]
    assert request["year"] == ["2014"]
    assert request["month"][0] == "01"
    assert request["month"][-1] == "12"
    assert request["day"][0] == "01"
    assert request["day"][-1] == "31"
    assert request["time"][0] == "00:00"
    assert request["time"][-1] == "23:00"
    assert request["pressure_level"] == ["1000", "975", "950"]
    assert request["area"] == [60.5, 29.5, 59.5, 31.0]
    assert request["data_format"] == "netcdf"
    assert request["download_format"] == "unarchived"


def test_dry_run_single_year_has_three_requests():
    lines = describe_dry_run(_cfg(), [2014])

    assert len(lines) == 3
    assert lines[0].startswith("static:")
    assert "single_levels" in lines[1]
    assert "pressure_levels" in lines[2]
    assert all(str(Path("data/raw/era5_v2")) in line for line in lines)
