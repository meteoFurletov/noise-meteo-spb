"""ERA5 v2 fetch and cache assembly via the CDS API.

Purpose
    Fetch the expanded ERA5 field set via the CDS API into a local Zarr cache
    for the Saint Petersburg propagation-climatology pipeline.

Why CDS rather than ARCO-Zarr
    The v2 cache needs instantaneous surface fluxes and sdfor, which are better
    aligned with ECMWF's canonical Monin-Obukhov and terrain-diagnostic recipes
    through the CDS endpoint than through the older ARCO-Zarr v1 path.

Idempotency contract
    Raw NetCDF files are the expensive boundary. Missing or empty files trigger
    CDS requests; existing non-empty files are reused without contacting CDS.

Output contract
    data/interim/era5_spb_v2.zarr with dimensions time, latitude, longitude,
    and level for pressure-level variables. Variable names are compact ERA5
    short names, with static geopotential renamed to z_surface so it cannot
    collide with pressure-level z.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable
from pathlib import Path
import json
import logging
import re
import shutil
import signal
import tempfile
import threading
import time
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

LOGGER = logging.getLogger(__name__)

DATASET_KINDS = ("single_levels", "pressure_levels", "static")
DEFAULT_CONFIG_PATH = Path("configs/spb_default.yaml")

SINGLE_LEVEL_OUTPUT_NAMES = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "surface_pressure": "sp",
    "total_cloud_cover": "tcc",
    "boundary_layer_height": "blh",
    "total_precipitation": "tp",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "skin_temperature": "skt",
    "instantaneous_surface_sensible_heat_flux": "ishf",
    "instantaneous_moisture_flux": "ie",
    "instantaneous_eastward_turbulent_surface_stress": "iews",
    "instantaneous_northward_turbulent_surface_stress": "inss",
    "friction_velocity": "zust",
    "surface_solar_radiation_downwards": "ssrd",
    "total_sky_direct_solar_radiation_at_surface": "fdir",
    "minimum_2m_temperature_since_previous_post_processing": "mn2t",
    "maximum_2m_temperature_since_previous_post_processing": "mx2t",
}

PRESSURE_LEVEL_OUTPUT_NAMES = {
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "geopotential": "z",
}

STATIC_OUTPUT_NAMES = {
    "geopotential": "z_surface",
    "standard_deviation_of_filtered_subgrid_orography": "sdfor",
}

BACKFILL_STAGES = ("pressure", "missing_single_levels", "all")

CDS_SHORT_TO_LONG = {
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "2t": "2m_temperature",
    "2d": "2m_dewpoint_temperature",
    "sp": "surface_pressure",
    "tcc": "total_cloud_cover",
    "blh": "boundary_layer_height",
    "tp": "total_precipitation",
    "100u": "100m_u_component_of_wind",
    "100v": "100m_v_component_of_wind",
    "skt": "skin_temperature",
    "ishf": "instantaneous_surface_sensible_heat_flux",
    "ie": "instantaneous_moisture_flux",
    "iews": "instantaneous_eastward_turbulent_surface_stress",
    "inss": "instantaneous_northward_turbulent_surface_stress",
    "zust": "friction_velocity",
    "ssrd": "surface_solar_radiation_downwards",
    "fdir": "total_sky_direct_solar_radiation_at_surface",
    "mn2t": "minimum_2m_temperature_since_previous_post_processing",
    "mx2t": "maximum_2m_temperature_since_previous_post_processing",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
    "z": "geopotential",
    "sdfor": "standard_deviation_of_filtered_subgrid_orography",
}

VARIABLE_METADATA = {
    "u10": {"units": "m s**-1", "long_name": "10 m U wind component"},
    "v10": {"units": "m s**-1", "long_name": "10 m V wind component"},
    "t2m": {"units": "K", "long_name": "2 m temperature"},
    "d2m": {"units": "K", "long_name": "2 m dewpoint temperature"},
    "sp": {"units": "Pa", "long_name": "Surface pressure"},
    "tcc": {"units": "1", "long_name": "Total cloud cover"},
    "blh": {"units": "m", "long_name": "Boundary layer height"},
    "tp": {"units": "m", "long_name": "Total precipitation"},
    "u100": {"units": "m s**-1", "long_name": "100 m U wind component"},
    "v100": {"units": "m s**-1", "long_name": "100 m V wind component"},
    "skt": {"units": "K", "long_name": "Skin temperature"},
    "ishf": {"units": "W m**-2", "long_name": "Instantaneous surface sensible heat flux"},
    "ie": {"units": "kg m**-2 s**-1", "long_name": "Instantaneous moisture flux"},
    "iews": {"units": "N m**-2", "long_name": "Instantaneous eastward turbulent stress"},
    "inss": {"units": "N m**-2", "long_name": "Instantaneous northward turbulent stress"},
    "zust": {"units": "m s**-1", "long_name": "Friction velocity"},
    "ssrd": {"units": "J m**-2", "long_name": "Surface solar radiation downwards"},
    "fdir": {"units": "J m**-2", "long_name": "Total sky direct solar radiation at surface"},
    "mn2t": {"units": "K", "long_name": "Minimum 2 m temperature since previous post-processing"},
    "mx2t": {"units": "K", "long_name": "Maximum 2 m temperature since previous post-processing"},
    "u": {"units": "m s**-1", "long_name": "U wind component"},
    "v": {"units": "m s**-1", "long_name": "V wind component"},
    "t": {"units": "K", "long_name": "Temperature"},
    "z": {"units": "m**2 s**-2", "long_name": "Geopotential"},
    "z_surface": {"units": "m**2 s**-2", "long_name": "Surface geopotential"},
    "sdfor": {"units": "m", "long_name": "Standard deviation of filtered subgrid orography"},
}


class QueueWaitTimeout(TimeoutError):
    """Raised when a CDS request exceeds the configured queue wait budget."""


def _load_v2_config(config_path: Path | str) -> dict[str, Any]:
    """Load and validate the era5_v2 config block."""
    path = Path(config_path)
    with path.open() as f:
        loaded = yaml.safe_load(f)

    if not isinstance(loaded, dict) or "era5_v2" not in loaded:
        raise KeyError(f"{path} does not contain a top-level 'era5_v2' block.")

    cfg = loaded["era5_v2"]
    required_paths = [
        ("time", "start"),
        ("time", "end"),
        ("area", "bbox"),
        ("datasets", "single_levels", "dataset_name"),
        ("datasets", "single_levels", "variables"),
        ("datasets", "pressure_levels", "dataset_name"),
        ("datasets", "pressure_levels", "pressure_levels"),
        ("datasets", "pressure_levels", "variables"),
        ("datasets", "static", "dataset_name"),
        ("datasets", "static", "reference_datetime"),
        ("datasets", "static", "variables"),
        ("request", "batch_strategy"),
        ("request", "format"),
        ("request", "polite_sleep_seconds"),
        ("request", "max_queue_wait_hours"),
        ("cache", "raw_dir"),
        ("cache", "interim_zarr"),
    ]
    missing = [path_parts for path_parts in required_paths if _nested_get(cfg, path_parts) is None]
    if missing:
        formatted = ", ".join(".".join(parts) for parts in missing)
        raise KeyError(f"Missing required era5_v2 config key(s): {formatted}")

    if cfg["request"]["batch_strategy"] != "per_year":
        raise ValueError("era5_v2.request.batch_strategy must be 'per_year'.")

    bbox = cfg["area"]["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("era5_v2.area.bbox must be [North, West, South, East].")

    for dataset_kind in DATASET_KINDS:
        variables = cfg["datasets"][dataset_kind]["variables"]
        if not variables:
            raise ValueError(f"era5_v2.datasets.{dataset_kind}.variables must not be empty.")

    cfg["_config_path"] = str(path)
    return cfg


def _nested_get(mapping: dict[str, Any], path_parts: Iterable[str]) -> Any:
    current: Any = mapping
    for part in path_parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _request_filename(dataset_kind: str, year: int | str | None, raw_dir: Path | str) -> Path:
    """Return the raw NetCDF cache path for one CDS request."""
    _validate_dataset_kind(dataset_kind)
    raw_path = Path(raw_dir)
    if dataset_kind == "static":
        return raw_path / "era5_spb_v2_static.nc"
    if year is None:
        raise ValueError("year is required for non-static ERA5 v2 request filenames.")
    return raw_path / f"era5_spb_v2_{dataset_kind}_{int(year)}.nc"


def _build_request(dataset_kind: str, year: int | str | None, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the CDS request dictionary for one dataset kind and year."""
    _validate_dataset_kind(dataset_kind)
    dataset_cfg = cfg["datasets"][dataset_kind]

    if dataset_kind == "static":
        reference = pd.Timestamp(dataset_cfg["reference_datetime"])
        years = [f"{reference.year:04d}"]
        months = [f"{reference.month:02d}"]
        days = [f"{reference.day:02d}"]
        hours = [f"{reference.hour:02d}:00"]
    else:
        if year is None:
            raise ValueError("year is required for non-static ERA5 v2 requests.")
        request_year = int(year)
        years = [f"{request_year:04d}"]
        months = [f"{month:02d}" for month in range(1, 13)]
        days = [f"{day:02d}" for day in range(1, 32)]
        hours = [f"{hour:02d}:00" for hour in range(24)]

    request: dict[str, Any] = {
        "product_type": "reanalysis",
        "variable": list(dataset_cfg["variables"]),
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "area": list(cfg["area"]["bbox"]),
        "data_format": cfg["request"].get("format", "netcdf"),
        "download_format": "unarchived",
    }
    if dataset_kind == "pressure_levels":
        request["pressure_level"] = list(dataset_cfg["pressure_levels"])
    return request


def fetch_one(
    client: Any,
    dataset_kind: str,
    year: int | str | None,
    cfg: dict[str, Any],
) -> Path | None:
    """Fetch one CDS request, reusing an existing non-empty raw file."""
    _validate_dataset_kind(dataset_kind)
    raw_dir = Path(cfg["cache"]["raw_dir"])
    target = _request_filename(dataset_kind, year, raw_dir)
    if target.exists() and target.stat().st_size > 0:
        LOGGER.info("ERA5 v2 cached: %s", target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(f"{target.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    dataset_name = cfg["datasets"][dataset_kind]["dataset_name"]
    request = _build_request(dataset_kind, year, cfg)
    label = "static" if dataset_kind == "static" else str(year)
    LOGGER.info("ERA5 v2 submitting %s %s to %s", dataset_kind, label, dataset_name)
    started = time.monotonic()

    try:
        _retrieve_with_timeout(
            client,
            dataset_name,
            request,
            tmp_path,
            max_wait_hours=float(cfg["request"]["max_queue_wait_hours"]),
        )
        tmp_path.rename(target)
    except QueueWaitTimeout:
        if tmp_path.exists():
            tmp_path.unlink()
        LOGGER.warning(
            "ERA5 v2 skipped %s %s after exceeding max_queue_wait_hours=%s",
            dataset_kind,
            label,
            cfg["request"]["max_queue_wait_hours"],
        )
        return None
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        if _is_cost_limit_exception(exc) and _can_split_by_month(request):
            LOGGER.warning(
                "ERA5 v2 %s %s exceeded CDS cost limits; falling back to packed "
                "monthly requests and merging into %s",
                dataset_kind,
                label,
                target,
            )
            return _fetch_one_split_by_months_packed(
                client,
                dataset_name,
                request,
                target,
                cfg,
                dataset_kind,
                label,
            )
        if _is_cost_limit_exception(exc) and len(request["variable"]) > 1:
            LOGGER.warning(
                "ERA5 v2 %s %s exceeded CDS cost limits; falling back to variable-group "
                "requests and merging into %s",
                dataset_kind,
                label,
                target,
            )
            return _fetch_one_split_by_variables(
                client,
                dataset_name,
                request,
                target,
                cfg,
                dataset_kind,
                label,
            )
        request_id = _extract_request_id(exc)
        if request_id:
            LOGGER.error("ERA5 v2 CDS request failed; request_id=%s", request_id)
        if _is_fatal_cds_exception(exc):
            raise
        LOGGER.warning("ERA5 v2 non-fatal CDS failure for %s %s: %s", dataset_kind, label, exc)
        return None

    elapsed = time.monotonic() - started
    LOGGER.info(
        "ERA5 v2 completed %s %s in %.1f min -> %s",
        dataset_kind,
        label,
        elapsed / 60.0,
        target,
    )
    return target


def fetch_all(cfg: dict[str, Any], years: Iterable[int] | None = None) -> dict[str, list[Path]]:
    """Fetch every configured ERA5 v2 NetCDF batch and return paths by kind."""
    import cdsapi

    selected_years = list(years) if years is not None else _years_from_cfg(cfg)
    if not selected_years:
        raise ValueError("No ERA5 v2 years selected for fetch.")

    downloaded: dict[str, list[Path]] = {kind: [] for kind in DATASET_KINDS}
    client = cdsapi.Client()
    sleep_seconds = float(cfg["request"].get("polite_sleep_seconds", 0.0))

    static_cached = _request_filename("static", None, cfg["cache"]["raw_dir"]).exists()
    static_path = fetch_one(client, "static", selected_years[0], cfg)
    if static_path is not None:
        downloaded["static"].append(static_path)
        if not static_cached and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    for year in selected_years:
        for dataset_kind in ("single_levels", "pressure_levels"):
            target = _request_filename(dataset_kind, year, cfg["cache"]["raw_dir"])
            was_cached = target.exists() and target.stat().st_size > 0
            path = fetch_one(client, dataset_kind, year, cfg)
            if path is None:
                continue
            downloaded[dataset_kind].append(path)
            if not was_cached and sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return downloaded


def fetch_pressure_then_missing_single_levels(
    cfg: dict[str, Any],
    years: Iterable[int] | None = None,
    stage: str = "all",
) -> dict[str, list[Path]]:
    """Backfill the slow CDS fields not covered by the fast time-series product.

    Pressure-level batches are fetched first. The single-level backfill only
    requests variables absent from ``single_levels_timeseries`` and writes them
    to ``era5_spb_v2_single_levels_missing_<year>.nc``.
    """
    import cdsapi

    if stage not in BACKFILL_STAGES:
        raise ValueError(f"stage must be one of {BACKFILL_STAGES}; got {stage!r}.")

    selected_years = list(years) if years is not None else _years_from_cfg(cfg)
    if not selected_years:
        raise ValueError("No ERA5 v2 years selected for backfill.")

    client = cdsapi.Client()
    downloaded: dict[str, list[Path]] = {
        "pressure_levels": [],
        "missing_single_levels": [],
    }

    if stage in ("pressure", "all"):
        for year in selected_years:
            path = fetch_pressure_level_year_split(client, year, cfg)
            if path is not None:
                downloaded["pressure_levels"].append(path)

    if stage in ("missing_single_levels", "all"):
        missing_variables = _missing_single_level_variables(cfg)
        if not missing_variables:
            LOGGER.info("ERA5 v2 no missing single-level variables to backfill.")
        for year in selected_years:
            full_single = _request_filename("single_levels", year, cfg["cache"]["raw_dir"])
            if _file_has_variables(full_single, "single_levels", missing_variables):
                LOGGER.info(
                    "ERA5 v2 missing single-level variables already covered by %s",
                    full_single,
                )
                downloaded["missing_single_levels"].append(full_single)
                continue
            path = fetch_single_level_missing_year(client, year, cfg, missing_variables)
            if path is not None:
                downloaded["missing_single_levels"].append(path)

    return downloaded


def fetch_pressure_level_year_split(client: Any, year: int, cfg: dict[str, Any]) -> Path | None:
    """Fetch one pressure-level year as packed monthly requests.

    A full year is too large for the CDS pressure-level endpoint in this domain,
    but one month with all four variables and three pressure levels is accepted
    and keeps the queued-request count much lower than variable-by-month shards.
    """
    target = _request_filename("pressure_levels", year, cfg["cache"]["raw_dir"])
    if target.exists() and target.stat().st_size > 0:
        LOGGER.info("ERA5 v2 cached: %s", target)
        return target

    request = _build_request("pressure_levels", year, cfg)
    dataset_name = cfg["datasets"]["pressure_levels"]["dataset_name"]
    return _fetch_one_split_by_months_packed(
        client,
        dataset_name,
        request,
        target,
        cfg,
        "pressure_levels",
        str(year),
    )


def fetch_single_level_missing_year(
    client: Any,
    year: int,
    cfg: dict[str, Any],
    variables: list[str] | None = None,
) -> Path | None:
    """Fetch one year of only the single-level variables missing from time series."""
    selected_variables = list(variables) if variables is not None else _missing_single_level_variables(cfg)
    if not selected_variables:
        return None

    raw_dir = Path(cfg["cache"]["raw_dir"])
    target = raw_dir / f"era5_spb_v2_single_levels_missing_{int(year)}.nc"
    dataset_name = cfg["datasets"]["single_levels"]["dataset_name"]
    request = _build_request("single_levels", year, cfg)
    request["variable"] = selected_variables
    return _fetch_one_split_by_quarters_packed(
        client,
        dataset_name,
        request,
        target,
        cfg,
        "single_levels",
        f"{year}-missing",
    )


def describe_backfill_dry_run(
    cfg: dict[str, Any],
    years: Iterable[int] | None = None,
    stage: str = "all",
) -> list[str]:
    """Return human-readable summaries for pressure/missing-single backfill."""
    if stage not in BACKFILL_STAGES:
        raise ValueError(f"stage must be one of {BACKFILL_STAGES}; got {stage!r}.")

    selected_years = list(years) if years is not None else _years_from_cfg(cfg)
    lines: list[str] = []
    if stage in ("pressure", "all"):
        for year in selected_years:
            request = _build_request("pressure_levels", year, cfg)
            raw_path = _request_filename("pressure_levels", year, cfg["cache"]["raw_dir"])
            lines.append(
                f"pressure_levels packed-monthly {year}: "
                f"12 requests x {len(request['variable'])} variables x "
                f"{len(request['pressure_level'])} levels -> {raw_path}"
            )
    if stage in ("missing_single_levels", "all"):
        missing_variables = _missing_single_level_variables(cfg)
        for year in selected_years:
            request = _build_request("single_levels", year, cfg)
            request["variable"] = missing_variables
            raw_path = Path(cfg["cache"]["raw_dir"]) / f"era5_spb_v2_single_levels_missing_{year}.nc"
            lines.append(
                f"single_levels missing {year}: "
                f"4 packed quarterly requests x {len(missing_variables)} variables "
                f"-> {raw_path}"
            )
    return lines


def enqueue_backfill_requests(
    cfg: dict[str, Any],
    years: Iterable[int] | None = None,
    stage: str = "all",
    log_path: Path | str | None = None,
) -> dict[str, Any]:
    """Submit pending packed backfill requests to CDS without waiting/downloading."""
    import cdsapi

    if stage not in BACKFILL_STAGES:
        raise ValueError(f"stage must be one of {BACKFILL_STAGES}; got {stage!r}.")

    selected_years = list(years) if years is not None else _years_from_cfg(cfg)
    client = cdsapi.Client(wait_until_complete=False)
    output_log = Path(log_path) if log_path is not None else Path("logs/era5_v2_enqueue_requests.jsonl")
    output_log.parent.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped = 0
    failed = 0
    with output_log.open("a") as f:
        for item in _iter_backfill_enqueue_items(cfg, selected_years, stage):
            if item["target"].exists() and item["target"].stat().st_size > 0:
                skipped += 1
                continue
            try:
                remote = client.retrieve(item["dataset_name"], item["request"])
                request_id = getattr(remote, "request_id", None)
                record = {
                    "status": "submitted",
                    "request_id": request_id,
                    "dataset_kind": item["dataset_kind"],
                    "label": item["label"],
                    "target": str(item["target"]),
                    "dataset_name": item["dataset_name"],
                    "request": item["request"],
                }
                submitted += 1
                LOGGER.info(
                    "ERA5 v2 enqueued %s %s request_id=%s",
                    item["dataset_kind"],
                    item["label"],
                    request_id,
                )
            except Exception as exc:
                record = {
                    "status": "failed",
                    "dataset_kind": item["dataset_kind"],
                    "label": item["label"],
                    "target": str(item["target"]),
                    "dataset_name": item["dataset_name"],
                    "error": str(exc),
                    "request": item["request"],
                }
                failed += 1
                LOGGER.warning(
                    "ERA5 v2 failed to enqueue %s %s: %s",
                    item["dataset_kind"],
                    item["label"],
                    exc,
                )
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()

    return {
        "submitted": submitted,
        "skipped": skipped,
        "failed": failed,
        "log_path": output_log,
    }


def _iter_backfill_enqueue_items(
    cfg: dict[str, Any],
    selected_years: list[int],
    stage: str,
) -> Iterable[dict[str, Any]]:
    if stage in ("pressure", "all"):
        dataset_cfg = cfg["datasets"]["pressure_levels"]
        dataset_name = dataset_cfg["dataset_name"]
        for year in selected_years:
            target = _request_filename("pressure_levels", year, cfg["cache"]["raw_dir"])
            if target.exists() and target.stat().st_size > 0:
                yield {
                    "dataset_kind": "pressure_levels",
                    "label": str(year),
                    "dataset_name": dataset_name,
                    "request": _build_request("pressure_levels", year, cfg),
                    "target": target,
                }
                continue
            request = _build_request("pressure_levels", year, cfg)
            for month in request["month"]:
                month_request = _request_for_month(request, month)
                month_target = target.with_name(f"{target.stem}_month{month}.nc")
                yield {
                    "dataset_kind": "pressure_levels",
                    "label": f"{year}-{month}",
                    "dataset_name": dataset_name,
                    "request": month_request,
                    "target": month_target,
                }

    if stage in ("missing_single_levels", "all"):
        dataset_name = cfg["datasets"]["single_levels"]["dataset_name"]
        missing_variables = _missing_single_level_variables(cfg)
        for year in selected_years:
            full_single = _request_filename("single_levels", year, cfg["cache"]["raw_dir"])
            if _file_has_variables(full_single, "single_levels", missing_variables):
                yield {
                    "dataset_kind": "missing_single_levels",
                    "label": str(year),
                    "dataset_name": dataset_name,
                    "request": _build_request("single_levels", year, cfg),
                    "target": full_single,
                }
                continue
            target = Path(cfg["cache"]["raw_dir"]) / f"era5_spb_v2_single_levels_missing_{year}.nc"
            request = _build_request("single_levels", year, cfg)
            request["variable"] = missing_variables
            yield {
                "dataset_kind": "missing_single_levels",
                "label": str(year),
                "dataset_name": dataset_name,
                "request": request,
                "target": target,
            }


def _open_and_normalize(
    nc_path: Path | str,
    dataset_kind: str,
    cfg: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Open one NetCDF file and normalize variable and coordinate names."""
    _validate_dataset_kind(dataset_kind)
    path = Path(nc_path)
    ds = _open_netcdf_or_zip(path)
    ds = _normalize_coords(ds)
    ds = _collapse_expver(ds)
    ds = _rename_output_variables(ds, dataset_kind)

    if "time" in ds.coords:
        ds = _normalize_time(ds)
    if dataset_kind == "static" and "time" in ds.dims:
        if ds.sizes["time"] != 1:
            raise ValueError(f"Static ERA5 file has {ds.sizes['time']} time steps: {path}")
        ds = ds.isel(time=0, drop=True)
    if dataset_kind == "pressure_levels":
        if "level" not in ds.coords:
            raise KeyError(f"Pressure-level ERA5 file has no 'level' coordinate: {path}")
        ds = ds.assign_coords(level=[int(level) for level in ds["level"].values])

    if cfg is not None:
        _verify_area_matches(ds, cfg, path)
    return ds


def assemble_zarr(downloaded_paths: dict[str, list[Path]], cfg: dict[str, Any]) -> Path:
    """Assemble raw ERA5 v2 NetCDF batches into the configured Zarr cache."""
    missing_kinds = [kind for kind in DATASET_KINDS if not downloaded_paths.get(kind)]
    if missing_kinds:
        raise ValueError(f"Cannot assemble ERA5 v2 zarr; missing raw files for {missing_kinds}.")

    single = _concat_time_kind(downloaded_paths["single_levels"], "single_levels", cfg)
    pressure = _concat_time_kind(downloaded_paths["pressure_levels"], "pressure_levels", cfg)
    static = _open_and_normalize(downloaded_paths["static"][0], "static", cfg)

    start, end = _time_bounds(cfg)
    single = single.sel(time=slice(start, end))
    pressure = pressure.sel(time=slice(start, end))

    ds = xr.merge([single, pressure, static], compat="override", combine_attrs="drop_conflicts")
    ds = ds.sortby("time")
    ds = ds.assign_coords(local_time=("time", _local_time_values(ds["time"])))
    ds = _apply_metadata(ds, cfg)

    output_path = Path(cfg["cache"]["interim_zarr"])
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    ds.to_zarr(tmp_path, mode="w", encoding=_zarr_encoding(ds), zarr_format=2)
    if output_path.exists():
        shutil.rmtree(output_path)
    tmp_path.rename(output_path)
    LOGGER.info("ERA5 v2 wrote %s in %.1f min", output_path, (time.monotonic() - started) / 60.0)
    return output_path


def assemble_zarr_from_raw_cache(cfg: dict[str, Any]) -> Path:
    """Assemble the completed hybrid raw cache into the configured Zarr cache."""
    raw_dir = Path(cfg["cache"]["raw_dir"])
    years = _years_from_cfg(cfg)

    single_timeseries = _open_timeseries_grid(cfg)
    single_missing = _open_missing_single_levels(raw_dir, years, cfg)
    pressure_paths = [_request_filename("pressure_levels", year, raw_dir) for year in years]
    pressure = _concat_time_kind(pressure_paths, "pressure_levels", cfg)
    static = _open_and_normalize(_request_filename("static", None, raw_dir), "static", cfg)

    start, end = _time_bounds(cfg)
    single_timeseries = single_timeseries.sel(time=slice(start, end))
    single_missing = single_missing.sel(time=slice(start, end))
    pressure = pressure.sel(time=slice(start, end))

    ds = xr.merge(
        [single_timeseries, single_missing, pressure, static],
        compat="override",
        combine_attrs="drop_conflicts",
    )
    ds = ds.sortby("time")
    ds = ds.assign_coords(local_time=("time", _local_time_values(ds["time"])))
    ds = _apply_metadata(ds, cfg)

    output_path = Path(cfg["cache"]["interim_zarr"])
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    ds.to_zarr(tmp_path, mode="w", encoding=_zarr_encoding(ds), zarr_format=2)
    if output_path.exists():
        shutil.rmtree(output_path)
    tmp_path.rename(output_path)
    LOGGER.info("ERA5 v2 wrote %s in %.1f min", output_path, (time.monotonic() - started) / 60.0)
    return output_path


def _open_timeseries_grid(cfg: dict[str, Any]) -> xr.Dataset:
    """Open point time-series ZIPs and combine them into the ERA5 bbox grid."""
    raw_dir = Path(cfg["cache"]["raw_dir"]).with_name("era5_v2_timeseries")
    points = _era5_grid_points_from_bbox(cfg["area"]["bbox"])
    expected_vars = {
        SINGLE_LEVEL_OUTPUT_NAMES[name]
        for name in cfg["datasets"].get("single_levels_timeseries", {}).get("variables", [])
    }
    if not expected_vars:
        raise KeyError("era5_v2.datasets.single_levels_timeseries.variables is required.")

    rows: list[xr.Dataset] = []
    for lat in sorted({point[0] for point in points}, reverse=True):
        row_cells: list[xr.Dataset] = []
        for lon in sorted({point[1] for point in points}):
            path = _timeseries_point_filename(raw_dir, lat, lon)
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing ERA5 v2 time-series point file: {path}")
            ds = _open_timeseries_point(path)
            missing = sorted(expected_vars.difference(ds.data_vars))
            if missing:
                raise ValueError(f"{path} is missing time-series variable(s): {missing}")
            ds = ds[list(sorted(expected_vars))]
            ds = ds.drop_vars(["latitude", "longitude"], errors="ignore")
            ds = ds.expand_dims(latitude=[float(lat)], longitude=[float(lon)])
            row_cells.append(ds)
        rows.append(
            xr.concat(
                row_cells,
                dim="longitude",
                data_vars="all",
                coords="minimal",
                compat="override",
                combine_attrs="drop_conflicts",
            )
        )

    combined = xr.concat(
        rows,
        dim="latitude",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )
    latitudes = np.array(sorted({point[0] for point in points}, reverse=True), dtype=float)
    longitudes = np.array(sorted({point[1] for point in points}), dtype=float)
    combined = combined.assign_coords(latitude=latitudes, longitude=longitudes)
    _verify_area_matches(combined, cfg, raw_dir)
    return _normalize_time(combined)


def _open_timeseries_point(path: Path) -> xr.Dataset:
    """Open one CDS time-series ZIP, loading it before the temp file disappears."""
    if not zipfile.is_zipfile(path):
        ds = xr.open_dataset(path, engine="netcdf4").load()
    else:
        with tempfile.TemporaryDirectory(prefix="era5_v2_timeseries_") as tmp_dir:
            with zipfile.ZipFile(path) as archive:
                members = sorted(member for member in archive.namelist() if member.endswith(".nc"))
                if len(members) != 1:
                    raise ValueError(f"{path} should contain exactly one NetCDF, found {members}")
                archive.extract(members[0], tmp_dir)
                ds = xr.open_dataset(Path(tmp_dir) / members[0], engine="netcdf4").load()

    ds = _normalize_coords(ds)
    ds = _collapse_expver(ds)
    ds = _rename_output_variables(ds, "single_levels")
    if "time" not in ds.coords:
        raise KeyError(f"ERA5 v2 time-series file has no time coordinate: {path}")
    return _normalize_time(ds)


def _open_missing_single_levels(
    raw_dir: Path,
    years: list[int],
    cfg: dict[str, Any],
) -> xr.Dataset:
    """Open variables absent from the time-series product for every year."""
    missing_variables = _missing_single_level_variables(cfg)
    expected = [SINGLE_LEVEL_OUTPUT_NAMES[name] for name in missing_variables]
    datasets: list[xr.Dataset] = []

    for year in years:
        full_single = _request_filename("single_levels", year, raw_dir)
        missing_single = raw_dir / f"era5_spb_v2_single_levels_missing_{year}.nc"
        path = full_single if _file_has_variables(full_single, "single_levels", missing_variables) else missing_single
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing ERA5 v2 single-level backfill file for {year}: {path}")
        ds = _open_and_normalize(path, "single_levels", cfg)
        absent = sorted(set(expected).difference(ds.data_vars))
        if absent:
            raise ValueError(f"{path} is missing single-level backfill variable(s): {absent}")
        datasets.append(ds[expected])

    combined = xr.concat(
        datasets,
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )
    combined = combined.sortby("time")
    index = pd.DatetimeIndex(pd.to_datetime(combined["time"].values))
    if index.has_duplicates:
        raise ValueError("ERA5 v2 missing single-level batches contain duplicate timestamps.")
    return combined


def verify_cache(zarr_path: Path | str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the ERA5 v2 Zarr cache and run sanity checks."""
    if cfg is None:
        cfg = _load_v2_config(DEFAULT_CONFIG_PATH)

    ds = xr.open_zarr(zarr_path, consolidated=False)
    results: dict[str, Any] = {}

    expected_vars = _expected_output_variables(cfg)
    missing_vars = sorted(expected_vars.difference(ds.data_vars))
    if missing_vars:
        raise ValueError(f"ERA5 v2 cache is missing required variables: {missing_vars}")
    results["variables_present"] = sorted(expected_vars)

    actual_time = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    start, end = _time_bounds(cfg)
    expected_time = pd.date_range(start, end, freq="h")
    results["time_first"] = str(actual_time[0]) if len(actual_time) else None
    results["time_last"] = str(actual_time[-1]) if len(actual_time) else None
    results["time_covers_config"] = bool(
        len(actual_time) > 0 and actual_time[0] <= start and actual_time[-1] >= end
    )
    results["time_has_duplicates"] = bool(actual_time.has_duplicates)
    deltas = pd.Series(actual_time).diff().dropna()
    results["time_hourly_cadence"] = bool((deltas == pd.Timedelta(hours=1)).all())
    results["time_has_gaps"] = bool(
        len(actual_time) != len(expected_time) or not np.array_equal(actual_time, expected_time)
    )

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    north, west, south, east = [float(value) for value in cfg["area"]["bbox"]]
    results["latitude_range"] = [float(np.nanmin(lat)), float(np.nanmax(lat))]
    results["longitude_range"] = [float(np.nanmin(lon)), float(np.nanmax(lon))]
    results["spatial_within_bbox"] = bool(
        np.nanmin(lat) >= south - 1e-6
        and np.nanmax(lat) <= north + 1e-6
        and np.nanmin(lon) >= west - 1e-6
        and np.nanmax(lon) <= east + 1e-6
    )
    results["spatial_shape"] = [int(ds.sizes["latitude"]), int(ds.sizes["longitude"])]
    results["spatial_expected_5x7"] = results["spatial_shape"] == [5, 7]

    if "level" not in ds.coords:
        raise ValueError("ERA5 v2 pressure-level cache is missing the 'level' coordinate.")
    level_values = [int(level) for level in ds["level"].values]
    results["level_values"] = level_values
    if level_values != [1000, 975, 950]:
        raise ValueError(f"ERA5 v2 level coordinate is {level_values}, expected [1000, 975, 950].")

    for static_name in ("z_surface", "sdfor"):
        dims = tuple(ds[static_name].dims)
        results[f"{static_name}_dims"] = dims
        if dims != ("latitude", "longitude"):
            raise ValueError(f"{static_name} must be 2D over latitude/longitude, got {dims}.")

    results["ranges"] = {
        "t2m_220_320K": _within_range(ds["t2m"], 220.0, 320.0),
        "sp_95000_105000Pa": _within_range(ds["sp"], 9.5e4, 1.05e5),
        "ishf_-1000_1000Wm-2": _within_range(ds["ishf"], -1000.0, 1000.0),
        "iews_-5_5Nm-2": _within_range(ds["iews"], -5.0, 5.0),
        "inss_-5_5Nm-2": _within_range(ds["inss"], -5.0, 5.0),
        "ssrd_nonnegative": _min_value(ds["ssrd"]) >= 0.0,
        "tp_nonnegative": _min_value(ds["tp"]) >= 0.0,
    }
    jja_ssrd = ds["ssrd"].where(ds["time"].dt.month.isin([6, 7, 8]), drop=True)
    results["ranges"]["ssrd_jja_max_positive"] = _max_value(jja_ssrd) > 0.0

    z1000 = ds["z"].sel(level=1000)
    subterranean = z1000 < ds["z_surface"]
    results["z1000_below_z_surface_fraction"] = float(subterranean.mean().compute())

    sdfor_bad = ds["sdfor"] >= 50.0
    results["sdfor_lt_50m_everywhere"] = not bool(sdfor_bad.any().compute())
    results["sdfor_offending_cells"] = _offending_sdfor_cells(ds, sdfor_bad)

    return results


def describe_dry_run(cfg: dict[str, Any], years: Iterable[int] | None = None) -> list[str]:
    """Return human-readable CDS request summaries without contacting CDS."""
    selected_years = list(years) if years is not None else _years_from_cfg(cfg)
    lines: list[str] = []
    static_request = _build_request("static", selected_years[0], cfg)
    lines.append(_request_summary("static", selected_years[0], cfg, static_request))
    for year in selected_years:
        for dataset_kind in ("single_levels", "pressure_levels"):
            request = _build_request(dataset_kind, year, cfg)
            lines.append(_request_summary(dataset_kind, year, cfg, request))
    return lines


def fetch_single_levels_timeseries_all(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fetch the fast CDS point time-series product for covered single-level variables.

    The time-series product is point-based, so this submits one request for each
    native ERA5 grid cell in the configured bbox. Existing non-empty files are
    reused.
    """
    dataset_cfg = cfg["datasets"].get("single_levels_timeseries")
    if not dataset_cfg:
        raise KeyError("era5_v2.datasets.single_levels_timeseries is not configured.")

    variables = list(dataset_cfg["variables"])
    points = _era5_grid_points_from_bbox(cfg["area"]["bbox"])
    raw_dir = Path(cfg["cache"]["raw_dir"]).with_name("era5_v2_timeseries")
    raw_dir.mkdir(parents=True, exist_ok=True)
    requested_workers = int(cfg["request"].get("timeseries_max_concurrent_requests", 5))
    max_workers = min(max(1, requested_workers), len(points))

    LOGGER.info(
        "ERA5 v2 time-series fetch: %s variables x %s grid points with %s workers",
        len(variables),
        len(points),
        max_workers,
    )

    results: list[Path] = []
    cached = 0
    started = time.monotonic()
    if max_workers == 1:
        for point in points:
            path, was_cached = _fetch_timeseries_point_with_new_client(
                dataset_cfg["dataset_name"],
                variables,
                point,
                cfg,
                raw_dir,
            )
            cached += int(was_cached)
            results.append(path)
            LOGGER.info(
                "ERA5 v2 time-series ready lat=%.2f lon=%.2f -> %s",
                point[0],
                point[1],
                path,
            )
        elapsed = time.monotonic() - started
        return {
            "raw_dir": raw_dir,
            "paths": sorted(results),
            "n_points": len(points),
            "n_variables": len(variables),
            "n_cached": cached,
            "elapsed_seconds": elapsed,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_timeseries_point_with_new_client,
                dataset_cfg["dataset_name"],
                variables,
                point,
                cfg,
                raw_dir,
            ): point
            for point in points
        }
        for future in as_completed(futures):
            point = futures[future]
            path, was_cached = future.result()
            cached += int(was_cached)
            results.append(path)
            LOGGER.info(
                "ERA5 v2 time-series ready lat=%.2f lon=%.2f -> %s",
                point[0],
                point[1],
                path,
            )

    elapsed = time.monotonic() - started
    return {
        "raw_dir": raw_dir,
        "paths": sorted(results),
        "n_points": len(points),
        "n_variables": len(variables),
        "n_cached": cached,
        "elapsed_seconds": elapsed,
    }


def describe_timeseries_dry_run(cfg: dict[str, Any]) -> list[str]:
    dataset_cfg = cfg["datasets"].get("single_levels_timeseries", {})
    variables = list(dataset_cfg.get("variables", []))
    points = _era5_grid_points_from_bbox(cfg["area"]["bbox"])
    raw_dir = Path(cfg["cache"]["raw_dir"]).with_name("era5_v2_timeseries")
    start, end = _time_bounds(cfg)
    return [
        (
            f"{dataset_cfg.get('dataset_name', 'missing-dataset')}: "
            f"{len(variables)} variables, {len(points)} grid points, "
            f"date={start:%Y-%m-%d}/{end:%Y-%m-%d}, raw_dir={raw_dir}"
        ),
        "variables: " + ", ".join(variables),
        "first_points: "
        + ", ".join(f"lat={lat:.2f},lon={lon:.2f}" for lat, lon in points[:5]),
    ]


def _fetch_timeseries_point_with_new_client(
    dataset_name: str,
    variables: list[str],
    point: tuple[float, float],
    cfg: dict[str, Any],
    raw_dir: Path,
) -> tuple[Path, bool]:
    import cdsapi

    return _fetch_timeseries_point(
        cdsapi.Client(),
        dataset_name,
        variables,
        point,
        cfg,
        raw_dir,
    )


def _fetch_timeseries_point(
    client: Any,
    dataset_name: str,
    variables: list[str],
    point: tuple[float, float],
    cfg: dict[str, Any],
    raw_dir: Path,
) -> tuple[Path, bool]:
    lat, lon = point
    target = _timeseries_point_filename(raw_dir, lat, lon)
    if target.exists() and target.stat().st_size > 0:
        LOGGER.info("ERA5 v2 time-series cached: %s", target)
        return target, True

    tmp_path = target.with_suffix(f"{target.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    start, end = _time_bounds(cfg)
    request = {
        "variable": variables,
        "location": {"longitude": float(lon), "latitude": float(lat)},
        "date": [f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"],
        "data_format": cfg["request"].get("format", "netcdf"),
    }
    LOGGER.info(
        "ERA5 v2 submitting time-series point lat=%.2f lon=%.2f (%s variables)",
        lat,
        lon,
        len(variables),
    )
    max_attempts = int(cfg["request"].get("timeseries_max_attempts", 1))
    retry_sleep = float(cfg["request"].get("timeseries_retry_sleep_seconds", 300))
    for attempt in range(1, max_attempts + 1):
        try:
            client.retrieve(dataset_name, request, str(tmp_path))
            tmp_path.rename(target)
            break
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < max_attempts and _is_retryable_timeseries_exception(exc):
                LOGGER.warning(
                    "ERA5 v2 time-series point lat=%.2f lon=%.2f hit CDS queue limit "
                    "on attempt %s/%s; sleeping %.0f seconds",
                    lat,
                    lon,
                    attempt,
                    max_attempts,
                    retry_sleep,
                )
                time.sleep(retry_sleep)
                continue
            raise
    return target, False


def _is_retryable_timeseries_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    retryable_fragments = (
        "too many requests",
        "temporarily limited",
        "number queued requests",
        "status has been updated to rejected",
    )
    return any(fragment in message for fragment in retryable_fragments)


def _missing_single_level_variables(cfg: dict[str, Any]) -> list[str]:
    time_series_variables = set(
        cfg["datasets"].get("single_levels_timeseries", {}).get("variables", [])
    )
    return [
        variable
        for variable in cfg["datasets"]["single_levels"]["variables"]
        if variable not in time_series_variables
    ]


def _file_has_variables(path: Path, dataset_kind: str, variables: list[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        ds = _open_and_normalize(path, dataset_kind)
        if dataset_kind == "single_levels":
            name_map = SINGLE_LEVEL_OUTPUT_NAMES
        elif dataset_kind == "pressure_levels":
            name_map = PRESSURE_LEVEL_OUTPUT_NAMES
        elif dataset_kind == "static":
            name_map = STATIC_OUTPUT_NAMES
        else:
            _validate_dataset_kind(dataset_kind)
            raise AssertionError("unreachable dataset kind")
        expected_names = {name_map[variable] for variable in variables}
        return expected_names.issubset(ds.data_vars)
    except Exception as exc:
        LOGGER.warning("ERA5 v2 could not inspect existing %s: %s", path, exc)
        return False
    finally:
        if "ds" in locals():
            ds.close()


def _fetch_custom_request(
    client: Any,
    dataset_name: str,
    request: dict[str, Any],
    target: Path,
    cfg: dict[str, Any],
    dataset_kind: str,
    label: str,
) -> Path | None:
    if target.exists() and target.stat().st_size > 0:
        LOGGER.info("ERA5 v2 cached: %s", target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(f"{target.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    LOGGER.info("ERA5 v2 submitting %s %s to %s", dataset_kind, label, dataset_name)
    started = time.monotonic()
    try:
        _retrieve_with_timeout(
            client,
            dataset_name,
            request,
            tmp_path,
            max_wait_hours=float(cfg["request"]["max_queue_wait_hours"]),
        )
        tmp_path.rename(target)
    except QueueWaitTimeout:
        if tmp_path.exists():
            tmp_path.unlink()
        LOGGER.warning(
            "ERA5 v2 skipped %s %s after exceeding max_queue_wait_hours=%s",
            dataset_kind,
            label,
            cfg["request"]["max_queue_wait_hours"],
        )
        return None
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        if _is_cost_limit_exception(exc) and len(request["variable"]) > 1:
            LOGGER.warning(
                "ERA5 v2 %s %s exceeded CDS cost limits; falling back to variable-group "
                "requests and merging into %s",
                dataset_kind,
                label,
                target,
            )
            return _fetch_one_split_by_variables(
                client,
                dataset_name,
                request,
                target,
                cfg,
                dataset_kind,
                label,
            )
        request_id = _extract_request_id(exc)
        if request_id:
            LOGGER.error("ERA5 v2 CDS request failed; request_id=%s", request_id)
        if _is_fatal_cds_exception(exc):
            raise
        LOGGER.warning("ERA5 v2 non-fatal CDS failure for %s %s: %s", dataset_kind, label, exc)
        return None

    elapsed = time.monotonic() - started
    LOGGER.info(
        "ERA5 v2 completed %s %s in %.1f min -> %s",
        dataset_kind,
        label,
        elapsed / 60.0,
        target,
    )
    return target


def _timeseries_point_filename(raw_dir: Path, lat: float, lon: float) -> Path:
    lat_label = f"{lat:+07.2f}".replace("+", "n").replace("-", "s").replace(".", "p")
    lon_label = f"{lon:+07.2f}".replace("+", "e").replace("-", "w").replace(".", "p")
    return raw_dir / f"era5_spb_v2_single_levels_timeseries_{lat_label}_{lon_label}.zip"


def _era5_grid_points_from_bbox(bbox: list[float]) -> list[tuple[float, float]]:
    north, west, south, east = [float(value) for value in bbox]
    latitudes = np.arange(north, south - 0.001, -0.25)
    longitudes = np.arange(west, east + 0.001, 0.25)
    return [(float(lat), float(lon)) for lat in latitudes for lon in longitudes]


def _validate_dataset_kind(dataset_kind: str) -> None:
    if dataset_kind not in DATASET_KINDS:
        raise ValueError(f"dataset_kind must be one of {DATASET_KINDS}; got {dataset_kind!r}.")


def _years_from_cfg(cfg: dict[str, Any]) -> list[int]:
    start, end = _time_bounds(cfg)
    return list(range(start.year, end.year + 1))


def _time_bounds(cfg: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(cfg["time"]["start"])
    end = pd.Timestamp(cfg["time"]["end"])
    if end.hour == 0 and end.minute == 0 and end.second == 0 and "T" not in str(cfg["time"]["end"]):
        end = end + pd.Timedelta(hours=23)
    return start, end


def _retrieve_with_timeout(
    client: Any,
    dataset_name: str,
    request: dict[str, Any],
    tmp_path: Path,
    max_wait_hours: float,
) -> None:
    timeout_seconds = max_wait_hours * 3600.0
    if timeout_seconds <= 0 or threading.current_thread() is not threading.main_thread():
        client.retrieve(dataset_name, request, str(tmp_path))
        return

    def _timeout_handler(signum: int, frame: Any) -> None:
        raise QueueWaitTimeout(
            f"CDS request exceeded max_queue_wait_hours={max_wait_hours:g}"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        client.retrieve(dataset_name, request, str(tmp_path))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _extract_request_id(exc: Exception) -> str | None:
    match = re.search(r"request[_ ]id[':= ]+([A-Za-z0-9-]+)", str(exc), flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _is_fatal_cds_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    transient_markers = (
        "timeout",
        "temporarily",
        "temporary",
        "connection",
        "gateway",
        "too many requests",
        "rate limit",
        "service unavailable",
        "ssl",
        "eof",
        "job not found",
        "job dismissed",
    )
    if any(marker in message for marker in transient_markers):
        return False
    return True


def _is_cost_limit_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cost limits exceeded" in message or "request is too large" in message


def _fetch_one_split_by_variables(
    client: Any,
    dataset_name: str,
    request: dict[str, Any],
    target: Path,
    cfg: dict[str, Any],
    dataset_kind: str,
    label: str,
) -> Path | None:
    variables = list(request["variable"])
    max_group_size = int(cfg["request"].get("max_variables_per_request", 6))
    groups = [
        (
            group_index,
            variables[start : start + max_group_size],
            target.with_name(f"{target.stem}_part{group_index:02d}.nc"),
        )
        for group_index, start in enumerate(range(0, len(variables), max_group_size), start=1)
    ]
    max_workers = min(
        int(cfg["request"].get("max_concurrent_requests", 1)),
        len(groups),
    )
    if max_workers > 1:
        LOGGER.info(
            "ERA5 v2 submitting up to %s concurrent %s variable-part request(s) for %s",
            max_workers,
            dataset_kind,
            label,
        )
        part_paths = _fetch_variable_groups_concurrently(
            groups,
            dataset_name,
            request,
            cfg,
            dataset_kind,
            label,
            max_workers,
        )
    else:
        part_paths = []
        for group_index, group, part_path in groups:
            _fetch_variable_group_part(
                client,
                dataset_name,
                request,
                group,
                part_path,
                cfg,
                dataset_kind,
                label,
                group_index,
            )
            part_paths.append(part_path)

    _merge_netcdf_parts(part_paths, target)
    LOGGER.info("ERA5 v2 merged %s variable parts into %s", len(part_paths), target)
    return target


def _fetch_one_split_by_months_packed(
    client: Any,
    dataset_name: str,
    request: dict[str, Any],
    target: Path,
    cfg: dict[str, Any],
    dataset_kind: str,
    label: str,
) -> Path | None:
    month_paths: list[Path] = []
    for month in request["month"]:
        month_request = _request_for_month(request, month)
        month_label = f"{label}-{month}"
        month_path = target.with_name(f"{target.stem}_month{month}.nc")
        legacy_month_path = _merge_legacy_variable_month_parts_if_present(
            target,
            month,
            len(month_request["variable"]),
        )
        if legacy_month_path is not None:
            month_paths.append(legacy_month_path)
            continue
        try:
            _fetch_variable_group(
                client,
                dataset_name,
                month_request,
                list(month_request["variable"]),
                month_path,
                cfg,
                dataset_kind,
                month_label,
            )
        except Exception as exc:
            if _is_cost_limit_exception(exc) and len(month_request["variable"]) > 1:
                LOGGER.warning(
                    "ERA5 v2 %s %s packed month is still too large; falling back "
                    "to variable groups for that month",
                    dataset_kind,
                    month_label,
                )
                split_path = _fetch_month_split_by_variables(
                    client,
                    dataset_name,
                    month_request,
                    month_path,
                    cfg,
                    dataset_kind,
                    month_label,
                )
                if split_path is None:
                    return None
            elif _is_fatal_cds_exception(exc):
                raise
            else:
                LOGGER.warning(
                    "ERA5 v2 non-fatal CDS failure for %s %s: %s",
                    dataset_kind,
                    month_label,
                    exc,
                )
                return None
        month_paths.append(month_path)

    _concat_netcdf_time_parts(month_paths, target)
    LOGGER.info("ERA5 v2 concatenated %s packed monthly parts into %s", len(month_paths), target)
    return target


def _fetch_one_split_by_quarters_packed(
    client: Any,
    dataset_name: str,
    request: dict[str, Any],
    target: Path,
    cfg: dict[str, Any],
    dataset_kind: str,
    label: str,
) -> Path | None:
    if target.exists() and target.stat().st_size > 0:
        LOGGER.info("ERA5 v2 cached: %s", target)
        return target

    quarter_paths: list[Path] = []
    quarters = (("q1", ["01", "02", "03"]), ("q2", ["04", "05", "06"]),
                ("q3", ["07", "08", "09"]), ("q4", ["10", "11", "12"]))
    for quarter_label, months in quarters:
        quarter_request = _request_for_months(request, months)
        part_label = f"{label}-{quarter_label}"
        quarter_path = target.with_name(f"{target.stem}_{quarter_label}.nc")
        _fetch_variable_group(
            client,
            dataset_name,
            quarter_request,
            list(quarter_request["variable"]),
            quarter_path,
            cfg,
            dataset_kind,
            part_label,
        )
        quarter_paths.append(quarter_path)

    _concat_netcdf_time_parts(quarter_paths, target)
    LOGGER.info("ERA5 v2 concatenated %s packed quarterly parts into %s", len(quarter_paths), target)
    return target


def _merge_legacy_variable_month_parts_if_present(
    target: Path,
    month: str,
    variable_count: int,
) -> Path | None:
    month_path = target.with_name(f"{target.stem}_month{month}.nc")
    if month_path.exists() and month_path.stat().st_size > 0:
        LOGGER.info("ERA5 v2 cached packed month: %s", month_path)
        return month_path

    legacy_paths = [
        target.with_name(f"{target.stem}_part{index:02d}_month{month}.nc")
        for index in range(1, variable_count + 1)
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in legacy_paths):
        return None

    LOGGER.info(
        "ERA5 v2 merging %s legacy variable-month parts into packed month %s",
        len(legacy_paths),
        month_path,
    )
    _merge_netcdf_parts(legacy_paths, month_path)
    return month_path


def _fetch_variable_groups_concurrently(
    groups: list[tuple[int, list[str], Path]],
    dataset_name: str,
    request: dict[str, Any],
    cfg: dict[str, Any],
    dataset_kind: str,
    label: str,
    max_workers: int,
) -> list[Path]:
    part_paths_by_index: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_variable_group_part_with_new_client,
                dataset_name,
                request,
                group,
                part_path,
                cfg,
                dataset_kind,
                label,
                group_index,
            ): (group_index, group, part_path)
            for group_index, group, part_path in groups
        }
        for future in as_completed(futures):
            group_index, group, part_path = futures[future]
            try:
                future.result()
            except Exception as exc:
                if _is_cost_limit_exception(exc) and len(group) > 1:
                    raise ValueError(
                        "Concurrent fallback received a grouped request that still "
                        "exceeded CDS cost limits. Reduce max_variables_per_request."
                    ) from exc
                if _is_fatal_cds_exception(exc):
                    raise
                LOGGER.warning(
                    "ERA5 v2 non-fatal CDS failure for %s %s part %s: %s",
                    dataset_kind,
                    label,
                    group_index,
                    exc,
                )
                raise
            part_paths_by_index[group_index] = part_path
    return [part_paths_by_index[index] for index, _, _ in groups]


def _fetch_variable_group_part_with_new_client(
    dataset_name: str,
    request: dict[str, Any],
    group: list[str],
    part_path: Path,
    cfg: dict[str, Any],
    dataset_kind: str,
    label: str,
    group_index: int,
) -> None:
    import cdsapi

    _fetch_variable_group_part(
        cdsapi.Client(),
        dataset_name,
        request,
        group,
        part_path,
        cfg,
        dataset_kind,
        label,
        group_index,
    )


def _fetch_variable_group_part(
    client: Any,
    dataset_name: str,
    request: dict[str, Any],
    group: list[str],
    part_path: Path,
    cfg: dict[str, Any],
    dataset_kind: str,
    label: str,
    group_index: int,
) -> None:
    try:
        _fetch_variable_group(
            client,
            dataset_name,
            request,
            group,
            part_path,
            cfg,
            dataset_kind,
            label,
        )
    except Exception as exc:
        if _is_cost_limit_exception(exc) and len(group) > 1:
            LOGGER.warning(
                "ERA5 v2 %s %s part %s is still too large; splitting into "
                "single-variable requests",
                dataset_kind,
                label,
                group_index,
            )
            single_paths = []
            for offset, variable in enumerate(group, start=1):
                single_path = part_path.with_name(
                    f"{part_path.stem}_{offset:02d}.nc"
                )
                _fetch_variable_group(
                    client,
                    dataset_name,
                    request,
                    [variable],
                    single_path,
                    cfg,
                    dataset_kind,
                    label,
                )
                single_paths.append(single_path)
            _merge_netcdf_parts(single_paths, part_path)
            return
        if _is_cost_limit_exception(exc) and _can_split_by_month(request):
            LOGGER.warning(
                "ERA5 v2 %s %s part %s is still too large for a full year; "
                "splitting this variable by month",
                dataset_kind,
                label,
                group_index,
            )
            _fetch_variable_group_by_months(
                client,
                dataset_name,
                request,
                group,
                part_path,
                cfg,
                dataset_kind,
                label,
                group_index,
            )
            return
        raise


def _fetch_variable_group_by_months(
    client: Any,
    dataset_name: str,
    request: dict[str, Any],
    group: list[str],
    part_path: Path,
    cfg: dict[str, Any],
    dataset_kind: str,
    label: str,
    group_index: int,
) -> None:
    month_paths: list[Path] = []
    for month in request["month"]:
        month_request = _request_for_month(request, month)
        month_path = part_path.with_name(f"{part_path.stem}_month{month}.nc")
        _fetch_variable_group(
            client,
            dataset_name,
            month_request,
            group,
            month_path,
            cfg,
            dataset_kind,
            f"{label}-{month}",
        )
        month_paths.append(month_path)
    _concat_netcdf_time_parts(month_paths, part_path)
    LOGGER.info(
        "ERA5 v2 concatenated %s monthly chunks into %s part %s",
        len(month_paths),
        dataset_kind,
        group_index,
    )


def _fetch_one_split_by_months(
    client: Any,
    dataset_name: str,
    request: dict[str, Any],
    target: Path,
    cfg: dict[str, Any],
    dataset_kind: str,
    label: str,
) -> Path | None:
    month_paths: list[Path] = []
    for month in request["month"]:
        month_request = _request_for_month(request, month)
        month_label = f"{label}-{month}"
        month_path = target.with_name(f"{target.stem}_month{month}.nc")
        try:
            split_path = _fetch_month_split_by_variables(
                client,
                dataset_name,
                month_request,
                month_path,
                cfg,
                dataset_kind,
                month_label,
            )
        except Exception as exc:
            if _is_fatal_cds_exception(exc):
                raise
            LOGGER.warning(
                "ERA5 v2 non-fatal CDS failure for %s %s: %s",
                dataset_kind,
                month_label,
                exc,
            )
            return None
        if split_path is None:
            return None
        month_paths.append(month_path)

    _concat_netcdf_time_parts(month_paths, target)
    LOGGER.info("ERA5 v2 concatenated %s monthly parts into %s", len(month_paths), target)
    return target


def _fetch_month_split_by_variables(
    client: Any,
    dataset_name: str,
    month_request: dict[str, Any],
    month_path: Path,
    cfg: dict[str, Any],
    dataset_kind: str,
    month_label: str,
) -> Path | None:
    variables = list(month_request["variable"])
    part_paths: list[Path] = []
    max_group_size = int(cfg["request"].get("max_variables_per_request", 6))
    for start in range(0, len(variables), max_group_size):
        group = variables[start : start + max_group_size]
        group_index = start // max_group_size + 1
        part_path = month_path.with_name(f"{month_path.stem}_part{group_index:02d}.nc")
        try:
            _fetch_variable_group(
                client,
                dataset_name,
                month_request,
                group,
                part_path,
                cfg,
                dataset_kind,
                month_label,
            )
        except Exception as exc:
            if _is_cost_limit_exception(exc) and len(group) > 1:
                single_paths = []
                for offset, variable in enumerate(group, start=1):
                    single_path = month_path.with_name(
                        f"{month_path.stem}_part{group_index:02d}_{offset:02d}.nc"
                    )
                    _fetch_variable_group(
                        client,
                        dataset_name,
                        month_request,
                        [variable],
                        single_path,
                        cfg,
                        dataset_kind,
                        month_label,
                    )
                    single_paths.append(single_path)
                _merge_netcdf_parts(single_paths, part_path)
            elif _is_fatal_cds_exception(exc):
                raise
            else:
                LOGGER.warning(
                    "ERA5 v2 non-fatal CDS failure for %s %s part %s: %s",
                    dataset_kind,
                    month_label,
                    group_index,
                    exc,
                )
                return None
        part_paths.append(part_path)
    _merge_netcdf_parts(part_paths, month_path)
    return month_path


def _fetch_variable_group(
    client: Any,
    dataset_name: str,
    base_request: dict[str, Any],
    variables: list[str],
    part_path: Path,
    cfg: dict[str, Any],
    dataset_kind: str,
    label: str,
) -> Path:
    if part_path.exists() and part_path.stat().st_size > 0:
        LOGGER.info("ERA5 v2 cached variable part: %s", part_path)
        return part_path

    request = dict(base_request)
    request["variable"] = variables
    tmp_path = part_path.with_suffix(f"{part_path.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    LOGGER.info(
        "ERA5 v2 submitting %s %s part with %s variable(s): %s",
        dataset_kind,
        label,
        len(variables),
        ", ".join(variables),
    )
    max_attempts = int(cfg["request"].get("cds_retry_attempts", 6))
    retry_sleep = float(cfg["request"].get("cds_retry_sleep_seconds", 180))
    for attempt in range(1, max_attempts + 1):
        try:
            _retrieve_with_timeout(
                client,
                dataset_name,
                request,
                tmp_path,
                max_wait_hours=float(cfg["request"]["max_queue_wait_hours"]),
            )
            tmp_path.rename(part_path)
            break
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < max_attempts and not _is_fatal_cds_exception(exc):
                LOGGER.warning(
                    "ERA5 v2 retryable CDS failure for %s %s part "
                    "(attempt %s/%s); sleeping %.0f seconds: %s",
                    dataset_kind,
                    label,
                    attempt,
                    max_attempts,
                    retry_sleep,
                    exc,
                )
                time.sleep(retry_sleep)
                continue
            raise
    sleep_seconds = float(cfg["request"].get("polite_sleep_seconds", 0.0))
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return part_path


def _can_split_by_month(request: dict[str, Any]) -> bool:
    return len(request.get("year", [])) == 1 and len(request.get("month", [])) > 1


def _request_for_month(request: dict[str, Any], month: str) -> dict[str, Any]:
    monthly = dict(request)
    year = int(monthly["year"][0])
    month_number = int(month)
    month_start = pd.Timestamp(year=year, month=month_number, day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    monthly["month"] = [month]
    monthly["day"] = [f"{day:02d}" for day in range(1, month_end.day + 1)]
    return monthly


def _request_for_months(request: dict[str, Any], months: list[str]) -> dict[str, Any]:
    grouped = dict(request)
    grouped["month"] = list(months)
    grouped["day"] = [f"{day:02d}" for day in range(1, 32)]
    return grouped


def _merge_netcdf_parts(part_paths: list[Path], target: Path) -> None:
    datasets = [_open_netcdf_or_zip(path) for path in part_paths]
    try:
        merged = xr.merge(datasets, compat="override", combine_attrs="drop_conflicts")
        loaded = merged.load()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(f"{target.suffix}.merge_tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        loaded.to_netcdf(tmp_path)
        tmp_path.rename(target)
    finally:
        for ds in datasets:
            ds.close()


def _concat_netcdf_time_parts(part_paths: list[Path], target: Path) -> None:
    datasets = [_open_netcdf_or_zip(path) for path in part_paths]
    try:
        time_coord = _time_coord_name(datasets[0])
        combined = xr.concat(
            datasets,
            dim=time_coord,
            data_vars="all",
            coords="minimal",
            compat="override",
            combine_attrs="drop_conflicts",
        ).sortby(time_coord)
        loaded = combined.load()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(f"{target.suffix}.concat_tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        loaded.to_netcdf(tmp_path)
        tmp_path.rename(target)
    finally:
        for ds in datasets:
            ds.close()


def _time_coord_name(ds: xr.Dataset) -> str:
    for name in ("time", "valid_time"):
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError("NetCDF part has no time or valid_time coordinate.")


def _open_netcdf_or_zip(path: Path) -> xr.Dataset:
    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path, engine="netcdf4")

    extract_dir = path.with_suffix(f"{path.suffix}.unzipped")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extract_dir)

    datasets = [xr.open_dataset(member, engine="netcdf4") for member in sorted(extract_dir.glob("*.nc"))]
    return xr.merge(datasets, compat="override", combine_attrs="drop_conflicts")


def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    for candidate in ("valid_time", "forecast_reference_time"):
        if candidate in ds.coords and "time" not in ds.coords:
            rename_map[candidate] = "time"
    for candidate in ("pressure_level", "isobaricInhPa"):
        if candidate in ds.coords and "level" not in ds.coords:
            rename_map[candidate] = "level"
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def _collapse_expver(ds: xr.Dataset) -> xr.Dataset:
    if "expver" not in ds.dims:
        return ds
    collapsed = ds.max("expver", skipna=True)
    return collapsed.drop_vars("expver", errors="ignore")


def _rename_output_variables(ds: xr.Dataset, dataset_kind: str) -> xr.Dataset:
    if dataset_kind == "single_levels":
        long_to_output = SINGLE_LEVEL_OUTPUT_NAMES
    elif dataset_kind == "pressure_levels":
        long_to_output = PRESSURE_LEVEL_OUTPUT_NAMES
    else:
        long_to_output = STATIC_OUTPUT_NAMES

    rename_map: dict[str, str] = {}
    for name in ds.data_vars:
        long_name = CDS_SHORT_TO_LONG.get(name, name)
        output_name = long_to_output.get(long_name)
        if output_name is not None and output_name != name:
            rename_map[name] = output_name
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def _normalize_time(ds: xr.Dataset) -> xr.Dataset:
    index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    if index.has_duplicates:
        duplicates = index[index.duplicated()].unique()
        raise ValueError(f"ERA5 NetCDF has duplicate timestamps: {list(duplicates[:5])}")
    ds = ds.assign_coords(time=index.to_numpy(dtype="datetime64[ns]"))
    return ds.sortby("time")


def _verify_area_matches(ds: xr.Dataset, cfg: dict[str, Any], path: Path) -> None:
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise KeyError(f"ERA5 file has no latitude/longitude coordinates: {path}")
    north, west, south, east = [float(value) for value in cfg["area"]["bbox"]]
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    tolerance = 1e-6
    if (
        abs(float(np.nanmax(lat)) - north) > tolerance
        or abs(float(np.nanmin(lat)) - south) > tolerance
        or abs(float(np.nanmin(lon)) - west) > tolerance
        or abs(float(np.nanmax(lon)) - east) > tolerance
    ):
        raise ValueError(
            f"ERA5 file area does not match bbox {cfg['area']['bbox']}: "
            f"lat=[{np.nanmin(lat)}, {np.nanmax(lat)}], "
            f"lon=[{np.nanmin(lon)}, {np.nanmax(lon)}], path={path}"
        )


def _concat_time_kind(paths: list[Path], dataset_kind: str, cfg: dict[str, Any]) -> xr.Dataset:
    datasets = [_open_and_normalize(path, dataset_kind, cfg) for path in sorted(paths)]
    combined = xr.concat(
        datasets,
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )
    combined = combined.sortby("time")
    index = pd.DatetimeIndex(pd.to_datetime(combined["time"].values))
    if index.has_duplicates:
        raise ValueError(f"ERA5 v2 {dataset_kind} batches contain duplicate timestamps.")
    return combined


def _local_time_values(time_coord: xr.DataArray) -> np.ndarray:
    utc = pd.DatetimeIndex(pd.to_datetime(time_coord.values))
    return (utc + pd.Timedelta(hours=3)).to_numpy(dtype="datetime64[ns]")


def _apply_metadata(ds: xr.Dataset, cfg: dict[str, Any]) -> xr.Dataset:
    for name in ds.data_vars:
        metadata = VARIABLE_METADATA.get(name, {})
        for key, value in metadata.items():
            ds[name].attrs.setdefault(key, value)
        cds_name = _output_to_cds_name(name)
        if cds_name is not None:
            ds[name].attrs.setdefault("era5_cds_variable", cds_name)

    spec_path = Path("docs/era5_data_specification.md")
    ds.attrs.update(
        {
            "title": "ERA5 v2 cache for Saint Petersburg propagation climatology",
            "source": "Copernicus CDS API",
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "configuration": yaml.safe_dump(_public_cfg(cfg), sort_keys=False),
            "spec_document": str(spec_path),
            "spec_document_present_at_creation": str(spec_path.exists()),
            "Conventions": "CF-1.10",
        }
    )
    return ds


def _output_to_cds_name(output_name: str) -> str | None:
    reverse = {
        **{value: key for key, value in SINGLE_LEVEL_OUTPUT_NAMES.items()},
        **{value: key for key, value in PRESSURE_LEVEL_OUTPUT_NAMES.items()},
        **{value: key for key, value in STATIC_OUTPUT_NAMES.items()},
    }
    return reverse.get(output_name)


def _public_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cfg.items() if not key.startswith("_")}


def _zarr_encoding(ds: xr.Dataset) -> dict[str, dict[str, tuple[int, ...]]]:
    return {name: {"chunks": _chunks_for_variable(ds, name)} for name in ds.data_vars}


def _chunks_for_variable(ds: xr.Dataset, name: str) -> tuple[int, ...]:
    chunks: list[int] = []
    for dim in ds[name].dims:
        if dim == "time":
            chunks.append(min(31 * 24, ds.sizes[dim]))
        elif dim == "level":
            chunks.append(1)
        else:
            chunks.append(ds.sizes[dim])
    return tuple(chunks)


def _expected_output_variables(cfg: dict[str, Any]) -> set[str]:
    expected = {
        SINGLE_LEVEL_OUTPUT_NAMES[name]
        for name in cfg["datasets"]["single_levels"]["variables"]
    }
    expected.update(
        PRESSURE_LEVEL_OUTPUT_NAMES[name]
        for name in cfg["datasets"]["pressure_levels"]["variables"]
    )
    expected.update(
        STATIC_OUTPUT_NAMES[name]
        for name in cfg["datasets"]["static"]["variables"]
    )
    return expected


def _within_range(da: xr.DataArray, lower: float, upper: float) -> bool:
    return bool(_min_value(da) >= lower and _max_value(da) <= upper)


def _min_value(da: xr.DataArray) -> float:
    return float(da.min(skipna=True).compute())


def _max_value(da: xr.DataArray) -> float:
    return float(da.max(skipna=True).compute())


def _offending_sdfor_cells(ds: xr.Dataset, mask: xr.DataArray) -> list[dict[str, float]]:
    if not bool(mask.any().compute()):
        return []
    offenders = mask.where(mask, drop=True)
    cells: list[dict[str, float]] = []
    for lat in offenders["latitude"].values:
        for lon in offenders["longitude"].values:
            value = ds["sdfor"].sel(latitude=lat, longitude=lon).item()
            if value >= 50.0:
                cells.append(
                    {
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "sdfor_m": float(value),
                    }
                )
    return cells


def _request_summary(
    dataset_kind: str,
    year: int,
    cfg: dict[str, Any],
    request: dict[str, Any],
) -> str:
    dataset_name = cfg["datasets"][dataset_kind]["dataset_name"]
    raw_path = _request_filename(dataset_kind, year, cfg["cache"]["raw_dir"])
    variables = request["variable"]
    period = (
        f"{request['year'][0]}-{request['month'][0]}-{request['day'][0]} "
        f"{request['time'][0]}"
    )
    if dataset_kind != "static":
        period = f"{year} hourly"
    return (
        f"{dataset_kind}: {dataset_name}, {period}, "
        f"{len(variables)} variables -> {raw_path}"
    )
