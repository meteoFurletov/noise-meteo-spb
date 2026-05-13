import numpy as np
import pandas as pd
import xarray as xr

import src.data.soundings as soundings_module
from src.data.soundings import (
    _monthly_list_profiles,
    _sample_launches,
    build_url,
    build_monthly_list_url,
    compute_sounding_richardson,
    fetch_sounding,
    parse_sounding,
    station_id_for,
)


def test_build_url_encodes_datetime_space():
    url = build_url("2020-01-01", 12, 26075)

    assert "datetime=2020-01-01%2012:00:00" in url
    assert "id=26075" in url
    assert "type=TEXT:CSV" in url


def test_build_monthly_list_url_uses_classic_endpoint():
    url = build_monthly_list_url(2020, 2, 26075)

    assert "cgi-bin/sounding" in url
    assert "TYPE=TEXT%3ALIST" in url
    assert "YEAR=2020" in url
    assert "MONTH=02" in url
    assert "TO=2912" in url
    assert "STNM=26075" in url


def test_monthly_list_profiles_split_html_tables():
    text = """
<HTML><H2>26075 ULLI Observations at 00Z 01 Mar 2020</H2>
<PRE>
-----------------------------------------------------------------------------
   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV
    hPa     m      C      C      %    g/kg    deg   knot     K      K      K
-----------------------------------------------------------------------------
 1000.0      2
  991.0     72   -1.9   -3.5     89   2.99    220     10  271.9  280.2  272.4
</PRE>
<H2>26075 ULLI Observations at 12Z 01 Mar 2020</H2>
<PRE>
-----------------------------------------------------------------------------
   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV
    hPa     m      C      C      %    g/kg    deg   knot     K      K      K
-----------------------------------------------------------------------------
 1006.0     72    0.1   -1.0     92   3.40    200      8  273.1  282.0  273.5
</PRE></HTML>
"""

    profiles = list(_monthly_list_profiles(text))

    assert [timestamp.isoformat() for timestamp, _ in profiles] == [
        "2020-03-01T00:00:00",
        "2020-03-01T12:00:00",
    ]
    assert "PRES" in profiles[0][1]
    assert "991.0" in profiles[0][1]


def test_station_id_defaults_to_old_without_cached_cutover(tmp_path, monkeypatch):
    monkeypatch.setattr(soundings_module, "DEFAULT_CUTOVER_CACHE", tmp_path / "cutover.json")

    assert station_id_for("2018-01-01") == 26063


def test_fetch_sounding_falls_forward_to_new_id_and_caches_cutover(tmp_path, monkeypatch):
    cutover_cache = tmp_path / "station_id_cutover.json"
    monkeypatch.setattr(soundings_module, "DEFAULT_CUTOVER_CACHE", cutover_cache)

    def fake_download(url, retries=3):
        if "id=26063" in url:
            return "Unable to retrieve the data for 26063 at 2018-01-01 00:00:00."
        return "\n".join(
            [
                "pressure_hPa,geopotential height_m,temperature_C,relative humidity_%,"
                "wind direction_degree,wind speed_m/s",
                "1007.0,72,-14.3,82,0,0.0",
                "1000.0,128,-13.1,91,10,3.0",
                "992.0,190,-12.3,94,25,6.0",
            ]
        )

    monkeypatch.setattr(soundings_module, "_download_text", fake_download)

    path = fetch_sounding("2018-01-01", 0, tmp_path / "raw")

    assert path is not None
    assert path.read_text(encoding="utf-8").startswith("pressure_hPa")
    assert path.with_suffix(".station_id").read_text(encoding="utf-8").strip() == "26075"
    assert station_id_for("2018-01-01") == 26075


def test_fetch_sounding_tries_fm35_and_retries_old_missing_marker(tmp_path, monkeypatch):
    cutover_cache = tmp_path / "station_id_cutover.json"
    monkeypatch.setattr(soundings_module, "DEFAULT_CUTOVER_CACHE", cutover_cache)
    cutover_cache.write_text('{"cutover_date": "2019-01-01"}', encoding="utf-8")
    raw_dir = tmp_path / "raw"
    stale_marker = raw_dir / "2020" / "07" / "20200701_00.missing"
    stale_marker.parent.mkdir(parents=True)
    stale_marker.write_text("missing\n", encoding="utf-8")
    attempted = []

    def fake_download(url, retries=3):
        attempted.append(url)
        if "id=26075" in url and "src=FM35" in url:
            return "\n".join(
                [
                    "pressure_hPa,geopotential height_m,temperature_C,relative humidity_%,"
                    "wind direction_degree,wind speed_m/s",
                    "1007.0,72,-14.3,82,0,0.0",
                    "1000.0,128,-13.1,91,10,3.0",
                    "992.0,190,-12.3,94,25,6.0",
                ]
            )
        return "Unable to retrieve the data."

    monkeypatch.setattr(soundings_module, "_download_text", fake_download)

    path = fetch_sounding("2020-07-01", 0, raw_dir)

    assert path is not None
    assert "src=UNKNOWN" in attempted[0]
    assert "src=FM35" in attempted[1]
    assert path.with_suffix(".source").read_text(encoding="utf-8").strip() == "FM35"
    assert not stale_marker.exists()


def test_parse_legacy_wyoming_text_profile(tmp_path):
    path = tmp_path / "sounding.txt"
    path.write_text(
        """
-----------------------------------------------------------------------------
   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV
    hPa     m      C      C      %    g/kg    deg   knot     K      K      K
-----------------------------------------------------------------------------
 1010.0     80   -2.0   -3.0     92   3.00     10      2  270.0  280.0  270.2
 1000.0    110   -1.5   -3.0     89   2.90     20      4  271.0  281.0  271.2
  990.0    230    0.0   -2.0     86   2.80     30      8  273.0  283.0  273.2
Description of the columns
""",
        encoding="utf-8",
    )

    parsed = parse_sounding(path)

    assert parsed["height_agl_m"].tolist() == [0.0, 30.0, 150.0]
    assert parsed["temperature_k"].iloc[0] == 271.15
    assert parsed["wind_speed_ms"].iloc[1] == 2.058


def test_parse_wyoming_wsgi_csv_units_header(tmp_path):
    path = tmp_path / "sounding.csv"
    path.write_text(
        "\n".join(
            [
                "pressure_hPa,geopotential height_m,temperature_C,dew point temperature_C,"
                "relative humidity_%,wind direction_degree,wind speed_m/s",
                "1007.0,72,-14.3,-16.7,82,0,0.0",
                "1000.0,128,-13.1,-14.3,91,10,3.0",
                "992.0,190,-12.3,-13.1,94,25,6.0",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_sounding(path)

    assert parsed["pressure_hpa"].tolist() == [1007.0, 1000.0, 992.0]
    assert parsed["height_agl_m"].tolist() == [0.0, 56.0, 118.0]
    assert parsed["temperature_k"].tolist() == [258.85, 260.05, 260.85]
    assert parsed["wind_direction_deg"].tolist() == [0.0, 10.0, 25.0]
    assert parsed["wind_speed_ms"].tolist() == [0.0, 3.0, 6.0]


def test_parse_normalized_monthly_cache_csv(tmp_path):
    path = tmp_path / "normalized.csv"
    path.write_text(
        "\n".join(
            [
                "pressure_hpa,height_agl_m,temperature_k,wind_speed_ms,wind_direction_deg,relative_humidity_pct",
                "1000.0,0.0,266.25,, ,",
                "994.0,47.0,266.25,2.057776,150.0,90.0",
                "981.0,149.0,265.45,3.086664,95.0,91.0",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_sounding(path)

    assert parsed["height_agl_m"].tolist() == [0.0, 47.0, 149.0]
    assert parsed["temperature_k"].iloc[0] == 266.25
    assert parsed["wind_speed_ms"].iloc[1] == 2.058
    assert parsed["relative_humidity_pct"].iloc[1] == 90.0


def test_seasonal_sampling_uses_first_week_of_seasonal_months():
    launches = _sample_launches("2020-01-01", "2020-12-31", "seasonal")

    assert len(launches) == 4 * 7 * 2
    assert set(pd.DatetimeIndex(launches).month) == {1, 4, 7, 10}
    assert set(pd.DatetimeIndex(launches).hour) == {0, 12}


def test_compute_sounding_richardson_positive_for_inversion():
    ds = xr.Dataset(
        {
            "height_agl_m": (("time", "level"), [[0.0, 2.0, 110.0]]),
            "pressure_hpa": (("time", "level"), [[1010.0, 1009.0, 1000.0]]),
            "temperature_k": (("time", "level"), [[270.0, 270.0, 272.0]]),
            "wind_speed_ms": (("time", "level"), [[1.0, 1.0, 5.0]]),
            "wind_direction_deg": (("time", "level"), [[0.0, 0.0, 0.0]]),
            "relative_humidity_pct": (("time", "level"), [[90.0, 90.0, 80.0]]),
            "valid": (("time",), [True]),
        },
        coords={"time": [np.datetime64("2020-01-01T00:00:00")], "level": [0, 1, 2]},
    )

    out = compute_sounding_richardson(ds)

    assert float(out["ri_sounding"].item()) > 0.0
    assert bool(out["valid"].item())
