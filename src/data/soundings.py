"""
Radiosonde validation data loader for the Saint Petersburg stability pipeline.

This module promotes the University of Wyoming Atmospheric Soundings archive
workflow from an exploratory notebook into a reproducible data source. It
downloads Voeikovo radiosonde profiles, keeps the slow-to-rebuild raw CSV cache
under ``data/raw/soundings/``, parses each launch into a consistent profile
schema, and assembles a validation-ready xarray dataset in
``data/interim/soundings_spb.zarr``.

Output contract
---------------
The assembled dataset has one row per requested launch time. Profile variables
use dimensions ``(time, level)`` because radiosondes report a variable number
of vertical levels:

``pressure_hpa``, ``height_agl_m``, ``temperature_k``, ``wind_speed_ms``,
``wind_direction_deg``, ``relative_humidity_pct``.

Per-launch variables include ``valid`` and ``station_id``. Invalid, missing, or
malformed launches remain present with ``valid=False`` so the sampling window is
auditable. ``compute_sounding_richardson`` adds ``ri_sounding(time)`` using the
same bulk Richardson convention as ``src.stability``.

Station-ID switchover
---------------------
Voeikovo changed WMO station IDs during the validation period. The loader does
not need an expensive pre-scan to find the transition. It reads
chronologically, starts with ID 26063, and when Wyoming returns a normal
``Unable to retrieve...`` response it tries ID 26075 for that launch. The first
successful fallback is cached in ``data/external/station_id_cutover.json`` so
later runs can go straight to the later ID.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
from io import StringIO
import json
import logging
from pathlib import Path
import random
import re
import shutil
import socket
import time
import calendar
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)

OLD_STATION_ID = 26063
NEW_STATION_ID = 26075
KNOWN_STATION_IDS = (OLD_STATION_ID, NEW_STATION_ID)

DEFAULT_CUTOVER_CACHE = Path("data/external/station_id_cutover.json")
DEFAULT_RAW_CACHE = Path("data/raw/soundings")
DEFAULT_OUTPUT_PATH = Path("data/interim/soundings_spb.zarr")
WYOMING_URL = "https://weather.uwyo.edu/wsgi/sounding"
WYOMING_MONTHLY_URL = "https://weather.uwyo.edu/cgi-bin/sounding"
USER_AGENT = "noise-meteo-spb/0.1 (research validation; polite cached loader)"
WYOMING_SOURCES = ("UNKNOWN", "FM35")
TRANSIENT_FAILURE = "__TRANSIENT_WYOMING_FAILURE__"
MISSING_MARKER_VERSION = "soundings-loader-v2"
MONTH_ABBREVIATIONS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

PROFILE_COLUMNS = (
    "pressure_hpa",
    "height_agl_m",
    "temperature_k",
    "wind_speed_ms",
    "wind_direction_deg",
    "relative_humidity_pct",
)

PROFILE_ROUNDING = {
    "pressure_hpa": 1,
    "height_agl_m": 1,
    "temperature_k": 2,
    "wind_speed_ms": 3,
    "wind_direction_deg": 1,
    "relative_humidity_pct": 1,
}


@dataclass(frozen=True)
class ProbeResult:
    """Availability result for one station/date pair."""

    date: str
    station_id: int
    valid: bool


def discover_station_id_cutover(
    year_range: tuple[int, int] | range | Iterable[int],
    probe_dates: Iterable[str | pd.Timestamp] | None = None,
) -> pd.Timestamp:
    """Return the cached station-ID cutover, or infer it by fallback loading.

    This function is retained for callers that want a cutover value. The normal
    loader path does not run a separate probing pass: it discovers the switch
    naturally when an old-ID fetch fails and a new-ID fetch succeeds.
    """
    if DEFAULT_CUTOVER_CACHE.exists():
        cached = _read_cutover_cache(DEFAULT_CUTOVER_CACHE)
        LOGGER.info("Using cached Voeikovo station-ID cutover: %s", cached.date())
        return cached

    years = _normalize_year_range(year_range)
    dates = _normalize_probe_dates(probe_dates, years)
    for date in dates:
        for hour in (0, 12):
            fetch_sounding(date, hour, DEFAULT_RAW_CACHE)
            if DEFAULT_CUTOVER_CACHE.exists():
                return _read_cutover_cache(DEFAULT_CUTOVER_CACHE)
    raise RuntimeError("Station-ID cutover is not known yet; run assemble_soundings over the period")


def station_id_for(date: str | pd.Timestamp) -> int:
    """Return the correct Voeikovo WMO station ID for a UTC launch date."""
    if not DEFAULT_CUTOVER_CACHE.exists():
        return OLD_STATION_ID
    cutover = _read_cutover_cache(DEFAULT_CUTOVER_CACHE)
    launch_date = pd.Timestamp(date).normalize()
    return NEW_STATION_ID if launch_date >= cutover else OLD_STATION_ID


def build_url(
    date: str | pd.Timestamp,
    hour: int,
    station_id: int,
    src: str = "UNKNOWN",
) -> str:
    """Construct the University of Wyoming WSGI URL for one sounding CSV."""
    timestamp = pd.Timestamp(date).replace(hour=int(hour), minute=0, second=0, microsecond=0)
    datetime_value = quote(f"{timestamp:%Y-%m-%d %H:00:00}", safe=":")
    return (
        f"{WYOMING_URL}?datetime={datetime_value}"
        f"&id={int(station_id)}&src={quote(src)}&type=TEXT:CSV"
    )


def build_monthly_list_url(year: int, month: int, station_id: int) -> str:
    """Construct the Wyoming monthly classic sounding-list URL."""
    last_day = calendar.monthrange(int(year), int(month))[1]
    return (
        f"{WYOMING_MONTHLY_URL}?region=europe&TYPE=TEXT%3ALIST"
        f"&YEAR={int(year):04d}&MONTH={int(month):02d}"
        f"&FROM=0100&TO={last_day:02d}12&STNM={int(station_id)}"
    )


def fetch_sounding(
    date: str | pd.Timestamp,
    hour: int,
    cache_dir: Path | str = DEFAULT_RAW_CACHE,
) -> Path | None:
    """Download one sounding CSV to the raw cache and return its path.

    Missing launches, empty responses, and 404 responses are represented by a
    sidecar ``.missing`` marker and return ``None``. Existing cache files are
    never re-downloaded.
    """
    cache_path = _raw_cache_path(pd.Timestamp(date), int(hour), Path(cache_dir))
    missing_marker = cache_path.with_suffix(".missing")
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    if missing_marker.exists() and _missing_marker_is_current(missing_marker):
        return None

    launch_date = pd.Timestamp(date)
    misses: list[ProbeResult] = []
    saw_transient_failure = False
    for station_id in _candidate_station_ids(launch_date):
        for src in WYOMING_SOURCES:
            url = build_url(launch_date, hour, station_id, src=src)
            text = _download_text(url)
            if text == TRANSIENT_FAILURE:
                saw_transient_failure = True
                LOGGER.warning(
                    "Transient Wyoming failure; will retry later: %s %02dZ station %s source %s",
                    launch_date.date(),
                    hour,
                    station_id,
                    src,
                )
                continue
            if text is None or _looks_missing(text):
                misses.append(ProbeResult(str(launch_date.date()), station_id, False))
                continue

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
            tmp_path.write_text(text, encoding="utf-8")
            tmp_path.rename(cache_path)
            _write_station_id_sidecar(cache_path, station_id)
            _write_source_sidecar(cache_path, src)
            if station_id == NEW_STATION_ID:
                _record_fallback_cutover(launch_date, misses)
            if missing_marker.exists():
                missing_marker.unlink()
            return cache_path

    if misses and not saw_transient_failure:
        _mark_missing(missing_marker, launch_date, hour)
    return None


def parse_sounding(csv_path: Path | str | None) -> pd.DataFrame:
    """Parse a Wyoming sounding response into the validation profile schema."""
    empty = _empty_profile_frame()
    if csv_path is None:
        return empty

    path = Path(csv_path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        LOGGER.warning("Could not read sounding file %s: %s", path, exc)
        return empty

    if _looks_missing(text):
        return empty

    try:
        raw = _parse_csv_or_text_table(text)
        return _normalize_profile_frame(raw)
    except Exception as exc:  # noqa: BLE001 - malformed observational files are expected.
        LOGGER.warning("Could not parse sounding file %s: %s", path, exc)
        return empty


def assemble_soundings(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    sample_strategy: str = "seasonal",
) -> xr.Dataset:
    """Assemble cached/downloaded Voeikovo soundings into one xarray dataset."""
    output_path = DEFAULT_OUTPUT_PATH
    cache_attrs = {
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
        "sample_strategy": sample_strategy,
    }
    if output_path.exists() and _zarr_attrs_match(output_path, cache_attrs):
        return xr.open_zarr(output_path, consolidated=False)

    launches = _sample_launches(start_date, end_date, sample_strategy)
    _prefetch_monthly_soundings(launches, DEFAULT_RAW_CACHE)
    profiles: list[pd.DataFrame] = []
    station_ids: list[int] = []
    valid_flags: list[bool] = []
    valid_count = 0

    for index, launch_time in enumerate(launches, start=1):
        path = fetch_sounding(launch_time, int(launch_time.hour), DEFAULT_RAW_CACHE)
        station_ids.append(_station_id_for_cached_launch(launch_time, path))
        profile = parse_sounding(path)
        is_valid = _profile_supports_ri(profile)
        profiles.append(profile)
        valid_flags.append(is_valid)
        valid_count += int(is_valid)
        if index == 1 or index % 100 == 0 or index == len(launches):
            LOGGER.warning(
                "Sounding progress: %s/%s launches through %s; valid=%s invalid_or_missing=%s",
                index,
                len(launches),
                launch_time.strftime("%Y-%m-%d %HZ"),
                valid_count,
                index - valid_count,
            )

    invalid_count = len(launches) - valid_count
    LOGGER.info(
        "Parsed Voeikovo soundings: %s valid, %s invalid/missing, %s requested",
        valid_count,
        invalid_count,
        len(launches),
    )

    ds = _profiles_to_dataset(launches, profiles, station_ids, valid_flags)
    ds.attrs.update(cache_attrs)
    ds.attrs["source"] = "University of Wyoming Atmospheric Soundings archive"
    if DEFAULT_CUTOVER_CACHE.exists():
        ds.attrs["station_id_cutover"] = str(_read_cutover_cache(DEFAULT_CUTOVER_CACHE).date())
    else:
        ds.attrs["station_id_cutover"] = "not reached in requested sample"
    ds.attrs["output_path"] = str(output_path)
    _write_zarr(ds, output_path)
    return xr.open_zarr(output_path, consolidated=False)


def assemble_cached_soundings(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    sample_strategy: str = "all",
    cache_dir: Path | str = DEFAULT_RAW_CACHE,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> xr.Dataset:
    """Assemble ``soundings_spb.zarr`` from the existing raw cache only.

    This does not touch the network. Missing launch files remain represented in
    the output with ``valid=False`` so the validation can proceed once the raw
    cache is good enough for analysis.
    """
    launches = _sample_launches(start_date, end_date, sample_strategy)
    cache_dir = Path(cache_dir)
    profiles: list[pd.DataFrame] = []
    station_ids: list[int] = []
    valid_flags: list[bool] = []
    valid_count = 0

    for launch_time in launches:
        path = _raw_cache_path(launch_time, int(launch_time.hour), cache_dir)
        if not path.exists() or path.stat().st_size == 0:
            path_or_none = None
        else:
            path_or_none = path
        profile = parse_sounding(path_or_none)
        is_valid = _profile_supports_ri(profile)
        profiles.append(profile)
        valid_flags.append(is_valid)
        station_ids.append(_station_id_for_cached_launch(launch_time, path_or_none))
        valid_count += int(is_valid)

    ds = _profiles_to_dataset(launches, profiles, station_ids, valid_flags)
    ds.attrs.update(
        {
            "start_date": str(pd.Timestamp(start_date).date()),
            "end_date": str(pd.Timestamp(end_date).date()),
            "sample_strategy": sample_strategy,
            "source": "University of Wyoming Atmospheric Soundings archive raw cache",
            "assembly_mode": "cache_only",
            "raw_cache_csv_count": int(len(list(cache_dir.glob("*/*/*.csv")))),
            "valid_launch_count": int(valid_count),
            "invalid_or_missing_launch_count": int(len(launches) - valid_count),
        }
    )
    if DEFAULT_CUTOVER_CACHE.exists():
        ds.attrs["station_id_cutover"] = str(_read_cutover_cache(DEFAULT_CUTOVER_CACHE).date())

    output_path = Path(output_path)
    _write_zarr(ds, output_path)
    return xr.open_zarr(output_path, consolidated=False)


def compute_sounding_richardson(
    soundings_ds: xr.Dataset,
    lower_height_m: float = 2.0,
    upper_height_m: float = 110.0,
) -> xr.Dataset:
    """Compute bulk Richardson number from each sounding profile."""
    ri_values = []
    for i in range(soundings_ds.sizes.get("time", 0)):
        profile = {
            name: soundings_ds[name].isel(time=i).values
            for name in PROFILE_COLUMNS
            if name in soundings_ds
        }
        ri_values.append(_richardson_from_profile(profile, lower_height_m, upper_height_m))

    ri = xr.DataArray(
        np.asarray(ri_values, dtype="float32"),
        dims=("time",),
        coords={"time": soundings_ds["time"]},
        name="ri_sounding",
    )
    ri.attrs.update(
        long_name="bulk Richardson number from Voeikovo radiosonde profile",
        formula="(g / T_mean) * (T_upper - T_lower) * dz / ((du)^2 + (dv)^2)",
        lower_height_agl_m=float(lower_height_m),
        upper_height_agl_m=float(upper_height_m),
        units="1",
    )
    out = soundings_ds.assign(ri_sounding=ri)
    out["valid"] = out["valid"] & np.isfinite(out["ri_sounding"])
    return out


def write_soundings_dataset(ds: xr.Dataset, output_path: Path | str = DEFAULT_OUTPUT_PATH) -> None:
    """Write a soundings dataset to Zarr using the project atomic-write pattern."""
    _write_zarr(ds, Path(output_path))


def rewrite_cached_soundings(cache_dir: Path | str = DEFAULT_RAW_CACHE) -> tuple[int, int]:
    """Rewrite cached raw sounding CSVs through the normalized rounded schema.

    Returns ``(rewritten, skipped)``. Sidecar metadata is left untouched.
    """
    rewritten = 0
    skipped = 0
    for path in sorted(Path(cache_dir).glob("*/*/*.csv")):
        profile = parse_sounding(path)
        if profile.empty:
            skipped += 1
            continue
        profile.to_csv(path, index=False)
        rewritten += 1
    return rewritten, skipped


def _normalize_year_range(year_range: tuple[int, int] | range | Iterable[int]) -> list[int]:
    if isinstance(year_range, range):
        return list(year_range)
    if isinstance(year_range, tuple) and len(year_range) == 2:
        return list(range(int(year_range[0]), int(year_range[1]) + 1))
    years = [int(year) for year in year_range]
    if not years:
        raise ValueError("year_range must contain at least one year")
    return years


def _prefetch_monthly_soundings(launches: pd.DatetimeIndex, cache_dir: Path) -> None:
    """Fetch monthly classic LIST pages and split them into per-launch cache files."""
    months = pd.DatetimeIndex(launches).to_period("M").unique()
    for month_index, month in enumerate(months, start=1):
        month_launches = launches[launches.to_period("M") == month]
        if _month_cache_is_complete(month_launches, cache_dir):
            continue

        LOGGER.warning(
            "Monthly sounding prefetch: %s/%s %04d-%02d",
            month_index,
            len(months),
            month.year,
            month.month,
        )
        fetched_any = False
        for station_id in _candidate_station_ids(month.start_time):
            text = _download_text(
                build_monthly_list_url(month.year, month.month, station_id),
                retries=2,
            )
            if text == TRANSIENT_FAILURE:
                continue
            if text is None or _looks_missing(text):
                continue
            written = _write_monthly_list_response(text, station_id, cache_dir)
            fetched_any = fetched_any or written > 0
            if written:
                break
        if not fetched_any:
            LOGGER.warning("No monthly LIST soundings found for %04d-%02d", month.year, month.month)


def _month_cache_is_complete(launches: pd.DatetimeIndex, cache_dir: Path) -> bool:
    for launch in launches:
        cache_path = _raw_cache_path(launch, int(launch.hour), cache_dir)
        missing_marker = cache_path.with_suffix(".missing")
        if cache_path.exists() and cache_path.stat().st_size > 0:
            continue
        if missing_marker.exists() and _missing_marker_is_current(missing_marker):
            continue
        return False
    return True


def _write_monthly_list_response(text: str, station_id: int, cache_dir: Path) -> int:
    count = 0
    for launch_time, table_text in _monthly_list_profiles(text):
        cache_path = _raw_cache_path(launch_time, int(launch_time.hour), cache_dir)
        missing_marker = cache_path.with_suffix(".missing")
        profile = _normalize_profile_frame(_parse_csv_or_text_table(table_text))
        if profile.empty:
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        profile.to_csv(cache_path, index=False)
        _write_station_id_sidecar(cache_path, station_id)
        _write_source_sidecar(cache_path, "MONTHLY_LIST")
        if missing_marker.exists():
            missing_marker.unlink()
        count += 1
    return count


def _monthly_list_profiles(text: str) -> Iterable[tuple[pd.Timestamp, str]]:
    pattern = re.compile(
        r"<H2>.*?Observations at\s+(\d{2})Z\s+(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})</H2>\s*<PRE>(.*?)</PRE>",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        hour = int(match.group(1))
        day = int(match.group(2))
        month = MONTH_ABBREVIATIONS[match.group(3).title()]
        year = int(match.group(4))
        table_text = html.unescape(match.group(5)).strip() + "\n"
        yield pd.Timestamp(year=year, month=month, day=day, hour=hour), table_text


def _normalize_probe_dates(
    probe_dates: Iterable[str | pd.Timestamp] | None, years: list[int]
) -> list[pd.Timestamp]:
    if probe_dates is not None:
        return sorted({pd.Timestamp(date).normalize() for date in probe_dates})

    dates = []
    for year in years:
        for month in (1, 4, 7, 10):
            dates.append(pd.Timestamp(year=year, month=month, day=1))
    return dates


def _probe_dates_for_ids(
    dates: Iterable[pd.Timestamp], station_ids: Iterable[int]
) -> list[ProbeResult]:
    results = []
    for date in dates:
        for station_id in station_ids:
            results.append(ProbeResult(str(pd.Timestamp(date).date()), station_id, _probe_station(date, station_id)))
    return results


def _probe_station(date: pd.Timestamp, station_id: int) -> bool:
    for hour in (0, 12):
        for src in WYOMING_SOURCES:
            text = _download_text(build_url(date, hour, station_id, src=src), retries=1)
            if text is not None and text != TRANSIENT_FAILURE and not _looks_missing(text):
                try:
                    parsed = _normalize_profile_frame(_parse_csv_or_text_table(text))
                except Exception as exc:  # noqa: BLE001 - probe responses can be malformed.
                    LOGGER.debug(
                        "Probe parse failed for %s %sZ station %s source %s: %s",
                        date.date(),
                        hour,
                        station_id,
                        src,
                        exc,
                    )
                    continue
                if _profile_supports_ri(parsed):
                    return True
    return False


def _cutover_bracket(results: list[ProbeResult]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    by_date = _availability_by_date(results)
    dates = sorted(by_date)
    previous_old_date = None
    for date in dates:
        old_valid = by_date[date].get(OLD_STATION_ID, False)
        new_valid = by_date[date].get(NEW_STATION_ID, False)
        if old_valid and not new_valid:
            previous_old_date = pd.Timestamp(date)
        if new_valid and not old_valid and previous_old_date is not None:
            return previous_old_date, pd.Timestamp(date)
    return None


def _infer_cutover_from_observations(results: list[ProbeResult]) -> pd.Timestamp:
    by_date = _availability_by_date(results)
    for date in sorted(by_date):
        old_valid = by_date[date].get(OLD_STATION_ID, False)
        new_valid = by_date[date].get(NEW_STATION_ID, False)
        if new_valid and not old_valid:
            return pd.Timestamp(date)

    new_dates = [
        pd.Timestamp(result.date)
        for result in results
        if result.station_id == NEW_STATION_ID and result.valid
    ]
    if new_dates:
        return min(new_dates)
    raise RuntimeError("Could not discover Voeikovo station-ID cutover from probes")


def _availability_by_date(results: list[ProbeResult]) -> dict[str, dict[int, bool]]:
    by_date: dict[str, dict[int, bool]] = {}
    for result in results:
        by_date.setdefault(result.date, {})[result.station_id] = result.valid
    return by_date


def _read_cutover_cache(path: Path) -> pd.Timestamp:
    with path.open() as f:
        payload = json.load(f)
    return pd.Timestamp(payload["cutover_date"]).normalize()


def _write_cutover_cache(cutover: pd.Timestamp, observations: list[ProbeResult]) -> None:
    DEFAULT_CUTOVER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cutover_date": str(cutover.date()),
        "old_station_id": OLD_STATION_ID,
        "new_station_id": NEW_STATION_ID,
        "rule": f"date < cutover uses {OLD_STATION_ID}; date >= cutover uses {NEW_STATION_ID}",
        "observations": [result.__dict__ for result in observations],
    }
    DEFAULT_CUTOVER_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _candidate_station_ids(date: pd.Timestamp) -> list[int]:
    primary = station_id_for(date)
    fallback = NEW_STATION_ID if primary == OLD_STATION_ID else OLD_STATION_ID
    return [primary, fallback]


def _record_fallback_cutover(date: pd.Timestamp, misses: list[ProbeResult]) -> None:
    cutover = date.normalize()
    observations = misses + [ProbeResult(str(cutover.date()), NEW_STATION_ID, True)]
    if DEFAULT_CUTOVER_CACHE.exists():
        existing = _read_cutover_cache(DEFAULT_CUTOVER_CACHE)
        if existing <= cutover:
            return
        LOGGER.info("Updating Voeikovo station-ID cutover from %s to %s", existing.date(), cutover.date())
    else:
        LOGGER.info(
            "Discovered Voeikovo station-ID cutover by fallback: %s; "
            "dates before use %s, dates on/after use %s",
            cutover.date(),
            OLD_STATION_ID,
            NEW_STATION_ID,
        )
    _write_cutover_cache(cutover, observations)


def _station_id_sidecar(path: Path) -> Path:
    return path.with_suffix(".station_id")


def _write_station_id_sidecar(path: Path, station_id: int) -> None:
    _station_id_sidecar(path).write_text(f"{int(station_id)}\n", encoding="utf-8")


def _source_sidecar(path: Path) -> Path:
    return path.with_suffix(".source")


def _write_source_sidecar(path: Path, src: str) -> None:
    _source_sidecar(path).write_text(f"{src}\n", encoding="utf-8")


def _station_id_for_cached_launch(date: pd.Timestamp, path: Path | None) -> int:
    if path is not None:
        sidecar = _station_id_sidecar(path)
        if sidecar.exists():
            try:
                return int(sidecar.read_text(encoding="utf-8").strip())
            except ValueError:
                LOGGER.warning("Malformed station-id sidecar: %s", sidecar)
    return station_id_for(date)


def _download_text(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries + 1):
        time.sleep(random.uniform(0.5, 1.0))
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS data source.
                if response.status == 404:
                    return None
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code == 400:
                body = exc.read().decode("utf-8", errors="replace")
                if _looks_missing(body):
                    return None
            LOGGER.warning("Wyoming request failed with HTTP %s: %s", exc.code, url)
        except URLError as exc:
            LOGGER.warning("Wyoming request failed: %s", exc)
        except TimeoutError as exc:
            LOGGER.warning("Wyoming request timed out: %s", exc)
        except socket.timeout as exc:
            LOGGER.warning("Wyoming socket timed out: %s", exc)
        if attempt < retries:
            time.sleep(2.0 * (attempt + 1))
    return TRANSIENT_FAILURE


def _looks_missing(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 50:
        return True
    lowered = stripped.lower()
    missing_markers = (
        "can't get",
        "cannot get",
        "no data",
        "not found",
        "no observations",
        "unable to retrieve",
        "503 service unavailable",
    )
    return any(marker in lowered for marker in missing_markers)


def _raw_cache_path(date: pd.Timestamp, hour: int, cache_dir: Path) -> Path:
    timestamp = date.replace(hour=hour, minute=0, second=0, microsecond=0)
    return cache_dir / f"{timestamp:%Y}" / f"{timestamp:%m}" / f"{timestamp:%Y%m%d}_{hour:02d}.csv"


def _missing_marker_is_current(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return MISSING_MARKER_VERSION in text and all(f"src={src}" in text for src in WYOMING_SOURCES)


def _mark_missing(path: Path, date: pd.Timestamp, hour: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    attempted = ", ".join(
        f"id={station_id} src={src}"
        for station_id in _candidate_station_ids(date)
        for src in WYOMING_SOURCES
    )
    path.write_text(
        f"missing\n"
        f"loader_version={MISSING_MARKER_VERSION}\n"
        f"datetime={date:%Y-%m-%d} {hour:02d}:00:00\n"
        f"attempted={attempted}\n",
        encoding="utf-8",
    )


def _parse_csv_or_text_table(text: str) -> pd.DataFrame:
    csv_candidate = pd.read_csv(StringIO(text), comment="#")
    if _has_profile_columns(csv_candidate.columns):
        return csv_candidate

    lines = text.splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if "PRES" in line.upper() and "HGHT" in line.upper()),
        None,
    )
    if header_index is None:
        raise ValueError("No recognizable sounding header")

    data_lines = []
    for line in lines[header_index + 2 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or "Description of" in stripped:
            continue
        if not stripped[0].isdigit():
            if data_lines:
                break
            continue
        data_lines.append(stripped)
    if not data_lines:
        raise ValueError("No sounding data rows")

    columns = lines[header_index].split()
    return pd.read_csv(StringIO("\n".join(data_lines)), sep=r"\s+", names=columns)


def _has_profile_columns(columns: Iterable[str]) -> bool:
    normalized = {_normalize_column_name(column) for column in columns}
    pressure_names = {"pres", "pressure", "pressure_hpa"}
    height_names = {"hght", "height", "height_m", "height_agl_m", "geopotential_height_m"}
    return bool(pressure_names & normalized) and bool(height_names & normalized)


def _normalize_profile_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return _empty_profile_frame()

    normalized_columns = {_normalize_column_name(column): column for column in raw.columns}

    def column(*names: str) -> pd.Series:
        for name in names:
            if name in normalized_columns:
                return pd.to_numeric(raw[normalized_columns[name]], errors="coerce")
        return pd.Series(np.nan, index=raw.index, dtype=float)

    if "height_agl_m" in normalized_columns:
        height_agl = column("height_agl_m")
    else:
        height_msl = column("hght", "height", "height_m", "geopotential_height_m")
        surface_height = float(height_msl.dropna().min()) if height_msl.notna().any() else np.nan
        height_agl = height_msl - surface_height

    if "temperature_k" in normalized_columns:
        temperature = column("temperature_k")
    else:
        temperature = column("temp", "temperature", "temperature_c") + 273.15

    wind_speed = column(
        "wspd",
        "sped",
        "speed",
        "wind_speed",
        "wind_speed_ms",
        "wind_speed_m_s",
    )
    if "sknt" in normalized_columns:
        wind_speed = column("sknt") * 0.514444

    out = pd.DataFrame(
        {
            "pressure_hpa": column("pres", "pressure", "pressure_hpa"),
            "height_agl_m": height_agl,
            "temperature_k": temperature,
            "wind_speed_ms": wind_speed,
            "wind_direction_deg": column(
                "drct",
                "direction",
                "wind_direction",
                "wind_direction_deg",
                "wind_direction_degree",
            ),
            "relative_humidity_pct": column(
                "relh",
                "rh",
                "relative_humidity",
                "relative_humidity_pct",
                "relative_humidity_%",
            ),
        }
    )
    out = out.replace({-9999: np.nan, -999: np.nan, 99999: np.nan})
    out = out.dropna(how="all")
    out = out.sort_values("height_agl_m").drop_duplicates("height_agl_m", keep="first")
    return _round_profile_frame(out.reset_index(drop=True))


def _round_profile_frame(profile: pd.DataFrame) -> pd.DataFrame:
    rounded = profile.copy()
    for column, decimals in PROFILE_ROUNDING.items():
        if column in rounded:
            rounded[column] = rounded[column].round(decimals)
    return rounded


def _normalize_column_name(column: object) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def _empty_profile_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=float) for column in PROFILE_COLUMNS})


def _profile_supports_ri(profile: pd.DataFrame) -> bool:
    if profile.empty:
        return False
    needed = ["height_agl_m", "temperature_k", "wind_speed_ms", "wind_direction_deg"]
    subset = profile[needed].dropna()
    if len(subset) < 2:
        return False
    heights = subset["height_agl_m"].to_numpy()
    return np.nanmin(heights) <= 2.0 and np.nanmax(heights) >= 110.0


def _sample_launches(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    sample_strategy: str,
) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must be <= end_date")

    strategy = sample_strategy.lower()
    if strategy == "all":
        days = pd.date_range(start, end, freq="D")
    elif strategy == "monthly":
        days = _first_week_months(start, end, months=range(1, 13))
    elif strategy == "seasonal":
        days = _first_week_months(start, end, months=(1, 4, 7, 10))
    else:
        raise ValueError("sample_strategy must be one of: all, monthly, seasonal")

    launches = []
    for day in days:
        for hour in (0, 12):
            launch = day + pd.Timedelta(hours=hour)
            if start <= launch.normalize() <= end:
                launches.append(launch)
    return pd.DatetimeIndex(launches, name="time")


def _first_week_months(
    start: pd.Timestamp,
    end: pd.Timestamp,
    months: Iterable[int],
) -> pd.DatetimeIndex:
    selected = []
    allowed = set(months)
    for month in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"):
        if month.month not in allowed:
            continue
        first = month.start_time
        for offset in range(7):
            day = first + pd.Timedelta(days=offset)
            if start <= day <= end:
                selected.append(day)
    return pd.DatetimeIndex(selected)


def _profiles_to_dataset(
    launches: pd.DatetimeIndex,
    profiles: list[pd.DataFrame],
    station_ids: list[int],
    valid_flags: list[bool],
) -> xr.Dataset:
    max_levels = max((len(profile) for profile in profiles), default=0)
    max_levels = max(max_levels, 1)
    data_vars = {}
    for column in PROFILE_COLUMNS:
        values = np.full((len(launches), max_levels), np.nan, dtype="float32")
        for i, profile in enumerate(profiles):
            if column not in profile:
                continue
            series = profile[column].to_numpy(dtype=float)
            values[i, : len(series)] = series.astype("float32")
        data_vars[column] = (("time", "level"), values)

    ds = xr.Dataset(
        data_vars={
            **data_vars,
            "valid": (("time",), np.asarray(valid_flags, dtype=bool)),
            "station_id": (("time",), np.asarray(station_ids, dtype="int32")),
        },
        coords={
            "time": launches.to_numpy(dtype="datetime64[ns]"),
            "level": np.arange(max_levels, dtype="int32"),
        },
    )
    ds["height_agl_m"].attrs.update(long_name="height above launch surface", units="m")
    ds["pressure_hpa"].attrs.update(long_name="pressure", units="hPa")
    ds["temperature_k"].attrs.update(long_name="air temperature", units="K")
    ds["wind_speed_ms"].attrs.update(long_name="wind speed", units="m s-1")
    ds["wind_direction_deg"].attrs.update(long_name="wind direction from north", units="degree")
    ds["relative_humidity_pct"].attrs.update(long_name="relative humidity", units="%")
    ds["valid"].attrs.update(long_name="profile has enough levels to compute 2-110 m bulk Ri")
    return ds


def _richardson_from_profile(
    profile: dict[str, np.ndarray],
    lower_height_m: float,
    upper_height_m: float,
) -> float:
    height = np.asarray(profile.get("height_agl_m", []), dtype=float)
    temperature = np.asarray(profile.get("temperature_k", []), dtype=float)
    speed = np.asarray(profile.get("wind_speed_ms", []), dtype=float)
    direction = np.asarray(profile.get("wind_direction_deg", []), dtype=float)
    valid = (
        np.isfinite(height)
        & np.isfinite(temperature)
        & np.isfinite(speed)
        & np.isfinite(direction)
    )
    if valid.sum() < 2:
        return np.nan

    height = height[valid]
    order = np.argsort(height)
    height = height[order]
    temperature = temperature[valid][order]
    speed = speed[valid][order]
    direction = direction[valid][order]

    unique_height, unique_index = np.unique(height, return_index=True)
    if len(unique_height) < 2 or unique_height[0] > lower_height_m or unique_height[-1] < upper_height_m:
        return np.nan

    temperature = temperature[unique_index]
    speed = speed[unique_index]
    direction = direction[unique_index]
    radians = np.deg2rad(direction)
    u = -speed * np.sin(radians)
    v = -speed * np.cos(radians)

    t_lower = np.interp(lower_height_m, unique_height, temperature)
    t_upper = np.interp(upper_height_m, unique_height, temperature)
    u_lower = np.interp(lower_height_m, unique_height, u)
    u_upper = np.interp(upper_height_m, unique_height, u)
    v_lower = np.interp(lower_height_m, unique_height, v)
    v_upper = np.interp(upper_height_m, unique_height, v)

    shear_squared = (u_upper - u_lower) ** 2 + (v_upper - v_lower) ** 2
    if shear_squared <= 0 or not np.isfinite(shear_squared):
        return np.nan

    gravity = 9.81
    dz = upper_height_m - lower_height_m
    t_mean = 0.5 * (t_upper + t_lower)
    return float((gravity / t_mean) * (t_upper - t_lower) * dz / shear_squared)


def _zarr_attrs_match(path: Path, attrs: dict[str, str]) -> bool:
    try:
        ds = xr.open_zarr(path, consolidated=False)
        try:
            return all(str(ds.attrs.get(key)) == value for key, value in attrs.items())
        finally:
            ds.close()
    except Exception:
        return False


def _write_zarr(ds: xr.Dataset, output_path: Path) -> None:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(tmp_path, mode="w", zarr_format=2)
    if output_path.exists():
        shutil.rmtree(output_path)
    tmp_path.rename(output_path)
