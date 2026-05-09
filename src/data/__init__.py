"""
Step 1: ERA5 fetch and dataset assembly.

Pulls ten years of hourly ERA5 reanalysis data for the Saint Petersburg
domain into a local Zarr cache. The cache is the input that all downstream
steps read from.

This is the only step that touches external data sources. Once it has run,
the rest of the pipeline runs offline.

────────────────────────────────────────────────────────────────────────────
Output contract — data/interim/era5_spb.zarr
────────────────────────────────────────────────────────────────────────────

Dimensions:
    time           ~96,000 hourly timesteps (2014-01-01 00:00 to 2024-12-31 23:00 UTC)
    latitude       ~5 cells (0.25° spacing across 59.5°–60.5°N)
    longitude      ~7 cells (0.25° spacing across 29.5°–31.0°E)
    pressure_level 1 (1000 hPa, used for upper wind in Richardson)

Coordinates:
    time           datetime64[ns], UTC
    latitude       float, decimal degrees N
    longitude      float, decimal degrees E
    pressure_level int, hPa
    local_time     datetime64[ns], computed from time + UTC+3, used for
                   day/evening/night classification downstream

Variables (single-level, all with dims (time, latitude, longitude)):
    u10            10m u-component of wind, m/s
    v10            10m v-component of wind, m/s
    t2m            2m temperature, K
    d2m            2m dewpoint temperature, K
    sp             surface pressure, Pa
    tcc            total cloud cover, fraction 0–1
    blh            boundary layer height, m
    tp             total precipitation in past hour, m

Variables (pressure-level, with dims (time, pressure_level, latitude, longitude)):
    u              u-component of wind, m/s
    v              v-component of wind, m/s
    t              temperature, K

────────────────────────────────────────────────────────────────────────────
Methodological choices
────────────────────────────────────────────────────────────────────────────

ARCO-ERA5 (Google Cloud, Zarr) is the preferred source — it allows lazy
spatial slicing without downloading global files. The CDS API is the
fallback for environments where Google Cloud is unreachable.

The 1000 hPa pressure level is used as the "~110 m AGL" reference for
Richardson's upper height. SPb is essentially at sea level so 1000 hPa
≈ 110 m geopotential height. This is one of the methodological choices
flagged for sensitivity analysis in the paper.

Downloads are cached. Reruns of this step are no-ops if the Zarr cache
exists and covers the requested time/space window; force a refresh by
deleting data/interim/era5_spb.zarr.

────────────────────────────────────────────────────────────────────────────
Implementation hints (for Claude Code)
────────────────────────────────────────────────────────────────────────────

1. Open the ARCO-ERA5 Zarr store (gs://gcp-public-data-arco-era5/...).
2. Slice spatially to the configured bounding box.
3. Slice temporally to the configured time window.
4. Select only the configured variables.
5. Compute and attach `local_time` coordinate.
6. Write to data/interim/era5_spb.zarr with sensible chunking
   (chunk by month in time, by 1 in pressure_level, full domain in space).

The whole step should be ~50 lines of xarray code; complexity is low.
"""

from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import zipfile

import pandas as pd
import xarray as xr

ARCO_ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_CACHE_PATH = Path("data/interim/era5_spb.zarr")

SINGLE_LEVEL_RENAMES = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "surface_pressure": "sp",
    "total_cloud_cover": "tcc",
    "boundary_layer_height": "blh",
    "total_precipitation": "tp",
}

PRESSURE_LEVEL_RENAMES = {
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
}

CDS_SHORT_NAMES = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "sp": "surface_pressure",
    "tcc": "total_cloud_cover",
    "blh": "boundary_layer_height",
    "tp": "total_precipitation",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
}


def fetch_era5(config: dict) -> xr.Dataset:
    """Fetch ERA5 data per config and return the assembled dataset.

    Implementation goes here. The function should:
    - read config["era5"], config["domain"], config["time"]
    - open ARCO-ERA5 (or fallback to CDS) per config["era5"]["source"]
    - slice spatially and temporally
    - add local_time coordinate
    - write to data/interim/era5_spb.zarr
    - return the opened dataset
    """
    output_path = Path(config.get("era5", {}).get("cache_path", DEFAULT_CACHE_PATH))
    if _cache_matches(output_path, config):
        return open_cached(output_path)

    era5_config = config["era5"]
    source = era5_config.get("source", "arco")
    if source == "arco":
        raw = _open_arco(era5_config)
    elif source == "cds":
        raw = _open_cds(config)
    else:
        raise ValueError(f"Unsupported ERA5 source: {source!r}")

    ds = _assemble_contract_dataset(raw, config)
    _validate_contract(ds, config)

    _write_zarr_cache(ds, output_path, config)
    return open_cached(output_path)


def open_cached(path: Path = DEFAULT_CACHE_PATH) -> xr.Dataset:
    """Open the cached Zarr dataset for downstream steps."""
    return xr.open_zarr(path, consolidated=False)


def _write_zarr_cache(ds: xr.Dataset, output_path: Path, config: dict[str, Any]) -> None:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(tmp_path, mode="w", encoding=_zarr_encoding(ds), zarr_format=2)

    written = open_cached(tmp_path)
    try:
        _validate_contract(written, config)
    finally:
        written.close()

    if output_path.exists():
        shutil.rmtree(output_path)
    tmp_path.rename(output_path)


def _open_arco(era5_config: dict[str, Any]) -> xr.Dataset:
    raw = xr.open_zarr(
        era5_config.get("arco_store", ARCO_ERA5_ZARR),
        chunks=None,
        storage_options={"token": "anon"},
    )
    if "valid_time_start" in raw.attrs and "valid_time_stop" in raw.attrs:
        raw = raw.sel(time=slice(raw.attrs["valid_time_start"], raw.attrs["valid_time_stop"]))
    return raw


def _open_cds(config: dict[str, Any]) -> xr.Dataset:
    raw_dir = Path(config.get("era5", {}).get("cds_raw_dir", "data/raw"))
    max_workers = int(config.get("era5", {}).get("cds_max_workers", 4))
    raw_dir.mkdir(parents=True, exist_ok=True)

    single_paths = _download_cds_batches(
        collection="reanalysis-era5-single-levels",
        config=config,
        variables=config["era5"].get("single_levels", []),
        raw_dir=raw_dir,
        prefix="era5_spb_single_levels",
        max_workers=max_workers,
    )
    pressure_paths = _download_cds_batches(
        collection="reanalysis-era5-pressure-levels",
        config=config,
        variables=config["era5"].get("pressure_level_variables", []),
        raw_dir=raw_dir,
        prefix="era5_spb_pressure_levels",
        pressure_levels=config["era5"].get("pressure_levels", []),
        max_workers=max_workers,
    )

    return xr.merge(
        [
            _combine_cds_batches(single_paths),
            _combine_cds_batches(pressure_paths),
        ],
        compat="override",
    )


def _download_cds_batches(
    collection: str,
    config: dict[str, Any],
    variables: list[str],
    raw_dir: Path,
    prefix: str,
    pressure_levels: list[int] | None = None,
    max_workers: int = 4,
) -> list[Path]:
    batches = _cds_requests(config, variables, pressure_levels)
    paths = [raw_dir / f"{prefix}_{label}.nc" for label, _ in batches]
    pending = [
        (path, request)
        for path, (_, request) in zip(paths, batches, strict=True)
        if not path.exists()
    ]
    if not pending:
        return paths

    workers = max(1, min(max_workers, len(pending)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_retrieve_cds_batch, collection, request, path)
            for path, request in pending
        ]
        for future in as_completed(futures):
            future.result()
    return paths


def _retrieve_cds_batch(collection: str, request: dict[str, Any], path: Path) -> None:
    import cdsapi

    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    cdsapi.Client().retrieve(collection, request, str(tmp_path))
    tmp_path.rename(path)


def _combine_cds_batches(paths: list[Path]) -> xr.Dataset:
    datasets = [_open_cds_dataset(path) for path in paths]
    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal", compat="override")


def _open_cds_dataset(path: Path) -> xr.Dataset:
    if not zipfile.is_zipfile(path):
        return _normalize_input_dataset(xr.open_dataset(path, engine="netcdf4"))

    extract_dir = path.with_suffix(f"{path.suffix}.unzipped")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extract_dir)

    datasets = [
        _normalize_input_dataset(xr.open_dataset(member, engine="netcdf4"))
        for member in sorted(extract_dir.glob("*.nc"))
    ]
    return xr.merge(datasets, compat="override")


def _cds_requests(
    config: dict[str, Any],
    variables: list[str],
    pressure_levels: list[int] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    start = pd.Timestamp(config["time"]["start"])
    end = pd.Timestamp(config["time"]["end"])
    batches = []
    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    for month in months:
        batch_start = month.start_time
        batch_end = batch_start + pd.offsets.MonthEnd(0)
        period_start = max(start, batch_start)
        period_end = min(end, batch_end.replace(hour=23))
        if period_start > period_end:
            continue
        label = f"{period_start:%Y%m}"
        request = _cds_request(config, variables, pressure_levels, period_start, period_end)
        batches.append((label, request))
    if not batches:
        label = f"{start:%Y%m}"
        batches.append((label, _cds_request(config, variables, pressure_levels, start, end)))
    return batches


def _cds_request(
    config: dict[str, Any],
    variables: list[str],
    pressure_levels: list[int] | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
    hours = sorted({f"{ts.hour:02d}:00" for ts in pd.date_range(start, end, freq="h")})
    request: dict[str, Any] = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": sorted({str(date.year) for date in dates}),
        "month": sorted({f"{date.month:02d}" for date in dates}),
        "day": sorted({f"{date.day:02d}" for date in dates}),
        "time": hours,
        "area": [
            config["domain"]["lat_max"],
            config["domain"]["lon_min"],
            config["domain"]["lat_min"],
            config["domain"]["lon_max"],
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if pressure_levels is not None:
        request["pressure_level"] = [str(level) for level in pressure_levels]
    return request


def _assemble_contract_dataset(raw: xr.Dataset, config: dict[str, Any]) -> xr.Dataset:
    """Select ERA5 fields and normalize them to the Step 1 output contract."""
    raw = _normalize_input_dataset(raw)
    era5_config = config["era5"]
    domain = config["domain"]
    time = config["time"]

    requested_single = era5_config.get("single_levels", [])
    requested_pressure = era5_config.get("pressure_level_variables", [])
    missing = [
        name
        for name in [*requested_single, *requested_pressure]
        if name not in raw.data_vars
    ]
    if missing:
        available = ", ".join(sorted(raw.data_vars))
        raise KeyError(
            "ERA5 store is missing requested variable(s): "
            f"{', '.join(missing)}. Available variables: {available}"
        )

    ds = raw[[*requested_single, *requested_pressure]]
    ds = _select_domain_and_time(ds, domain, time)

    level_coord = _level_coord_name(ds)
    if level_coord is None:
        raise KeyError("ERA5 pressure-level coordinate 'level' or 'pressure_level' was not found.")
    ds = ds.sel({level_coord: era5_config.get("pressure_levels", [])})
    if level_coord != "pressure_level":
        ds = ds.rename({level_coord: "pressure_level"})

    rename_map = {
        **{name: SINGLE_LEVEL_RENAMES[name] for name in requested_single},
        **{name: PRESSURE_LEVEL_RENAMES[name] for name in requested_pressure},
    }
    ds = ds.rename(rename_map)
    ds = ds.assign_coords(local_time=("time", _local_time_values(ds["time"], time)))

    return ds[
        [
            "u10",
            "v10",
            "t2m",
            "d2m",
            "sp",
            "tcc",
            "blh",
            "tp",
            "u",
            "v",
            "t",
        ]
    ]


def _normalize_input_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename_map["valid_time"] = "time"
    for short_name, contract_name in CDS_SHORT_NAMES.items():
        if short_name in ds.data_vars and contract_name not in ds.data_vars:
            rename_map[short_name] = contract_name
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def _select_domain_and_time(
    ds: xr.Dataset,
    domain: dict[str, Any],
    time: dict[str, Any],
) -> xr.Dataset:
    return ds.sel(
        time=slice(time["start"], time["end"]),
        latitude=_coord_slice(ds["latitude"], domain["lat_min"], domain["lat_max"]),
        longitude=_coord_slice(ds["longitude"], domain["lon_min"], domain["lon_max"]),
    )


def _coord_slice(coord: xr.DataArray, lower: float, upper: float) -> slice:
    first = float(coord.values[0])
    last = float(coord.values[-1])
    if first <= last:
        return slice(lower, upper)
    return slice(upper, lower)


def _level_coord_name(ds: xr.Dataset) -> str | None:
    for name in ("level", "pressure_level"):
        if name in ds.coords or name in ds.dims:
            return name
    return None


def _local_time_values(time_coord: xr.DataArray, time_config: dict[str, Any]) -> Any:
    offset_hours = time_config.get("utc_offset_hours", 3)
    utc = pd.DatetimeIndex(time_coord.values)
    return (utc + pd.Timedelta(hours=offset_hours)).to_numpy()


def _dim_chunk(ds: xr.Dataset, dim: str) -> int:
    if dim == "time":
        return min(31 * 24, ds.sizes[dim])
    if dim == "pressure_level":
        return 1
    return ds.sizes[dim]


def _zarr_encoding(ds: xr.Dataset) -> dict[str, dict[str, tuple[int, ...]]]:
    return {name: {"chunks": _chunks_for_variable(ds, name)} for name in ds.data_vars}


def _chunks_for_variable(ds: xr.Dataset, name: str) -> tuple[int, ...]:
    chunks: dict[str, int] = {}
    for dim in ds[name].dims:
        chunks[dim] = _dim_chunk(ds, dim)
    return tuple(chunks.values())


def _cache_matches(path: Path, config: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    if _metadata_only_zarr(path):
        return False

    try:
        ds = open_cached(path)
        _validate_contract(ds, config)
    except Exception:
        return False
    finally:
        if "ds" in locals():
            ds.close()
    return True


def _metadata_only_zarr(path: Path) -> bool:
    if not path.is_dir():
        return False
    children = {child.name for child in path.iterdir()}
    return children.issubset({"zarr.json", ".zgroup", ".zattrs", ".zmetadata"})


def _validate_contract(ds: xr.Dataset, config: dict[str, Any]) -> None:
    required_dims = {"time", "latitude", "longitude", "pressure_level"}
    required_coords = {*required_dims, "local_time"}
    required_vars = {"u10", "v10", "t2m", "d2m", "sp", "tcc", "blh", "tp", "u", "v", "t"}

    missing_dims = required_dims.difference(ds.dims)
    missing_coords = required_coords.difference(ds.coords)
    missing_vars = required_vars.difference(ds.data_vars)
    if missing_dims or missing_coords or missing_vars:
        raise ValueError(
            "Cached ERA5 dataset does not match Step 1 contract: "
            f"missing dims={sorted(missing_dims)}, "
            f"coords={sorted(missing_coords)}, vars={sorted(missing_vars)}"
        )
    if ds.sizes["time"] == 0:
        raise ValueError("Cached ERA5 dataset has no time steps.")

    domain = config["domain"]
    time = config["time"]
    if pd.Timestamp(ds["time"].values[0]) > pd.Timestamp(time["start"]):
        raise ValueError("Cached ERA5 dataset starts after requested start time.")
    if pd.Timestamp(ds["time"].values[-1]) < pd.Timestamp(time["end"]):
        raise ValueError("Cached ERA5 dataset ends before requested end time.")

    lat_min = float(ds["latitude"].min())
    lat_max = float(ds["latitude"].max())
    lon_min = float(ds["longitude"].min())
    lon_max = float(ds["longitude"].max())
    if lat_min > domain["lat_min"] or lat_max < domain["lat_max"]:
        raise ValueError("Cached ERA5 dataset does not cover requested latitude range.")
    if lon_min > domain["lon_min"] or lon_max < domain["lon_max"]:
        raise ValueError("Cached ERA5 dataset does not cover requested longitude range.")
