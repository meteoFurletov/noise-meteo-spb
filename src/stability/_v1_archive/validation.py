"""Validation utilities joining Voeikovo soundings to ERA5 Richardson values."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

DEFAULT_PAIRS_PATH = Path("data/interim/sounding_era5_pairs.zarr")
SEASON_MONTHS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def find_nearest_era5_cell(era5_ds: xr.Dataset, lat: float, lon: float) -> dict[str, float]:
    """Return the nearest ERA5 latitude/longitude coordinate to ``lat, lon``."""
    selected = era5_ds.sel(latitude=lat, longitude=lon, method="nearest")
    return {
        "latitude": float(selected["latitude"].item()),
        "longitude": float(selected["longitude"].item()),
    }


def match_soundings_to_era5(
    soundings_ds: xr.Dataset,
    era5_ds: xr.Dataset,
    stability_ds: xr.Dataset,
    station_lat: float,
    station_lon: float,
) -> xr.Dataset:
    """Pair sounding-derived Ri with ERA5 Ri at the nearest grid cell and hour."""
    if "ri_sounding" not in soundings_ds:
        raise ValueError("soundings_ds must contain ri_sounding; run compute_sounding_richardson first")
    if "ri_value" not in stability_ds:
        raise ValueError("stability_ds must contain ri_value from classify_stability")

    cell = find_nearest_era5_cell(era5_ds, station_lat, station_lon)
    sounding_ri = soundings_ds["ri_sounding"]
    if "valid" in soundings_ds:
        sounding_ri = sounding_ri.where(soundings_ds["valid"])

    era5_ri = (
        stability_ds["ri_value"]
        .sel(latitude=cell["latitude"], longitude=cell["longitude"], method="nearest")
        .sel(time=soundings_ds["time"], method="nearest")
    )
    era5_ri = era5_ri.drop_vars(
        [name for name in ("latitude", "longitude") if name in era5_ri.coords],
        errors="ignore",
    )

    paired = xr.Dataset(
        {
            "ri_sounding": sounding_ri.astype("float32"),
            "ri_era5": era5_ri.astype("float32"),
        },
        coords={"time": soundings_ds["time"]},
        attrs={
            "station_latitude": float(station_lat),
            "station_longitude": float(station_lon),
            "era5_latitude": cell["latitude"],
            "era5_longitude": cell["longitude"],
            "time_basis": "UTC; Wyoming archive and ERA5 are both UTC; matched by nearest ERA5 hour",
        },
    )
    paired["paired_valid"] = np.isfinite(paired["ri_sounding"]) & np.isfinite(paired["ri_era5"])
    _write_zarr(paired, DEFAULT_PAIRS_PATH)
    return xr.open_zarr(DEFAULT_PAIRS_PATH, consolidated=False)


def agreement_statistics(paired_ds: xr.Dataset, stable_threshold: float = 0.1) -> dict[str, Any]:
    """Compute raw-Ri and binary stable/non-stable agreement statistics."""
    overall = _statistics_for_arrays(
        paired_ds["ri_sounding"].values,
        paired_ds["ri_era5"].values,
        stable_threshold,
    )
    seasonal = {}
    months = pd.DatetimeIndex(paired_ds["time"].values).month
    for season, season_months in SEASON_MONTHS.items():
        mask = np.isin(months, season_months)
        seasonal[season] = _statistics_for_arrays(
            paired_ds["ri_sounding"].values[mask],
            paired_ds["ri_era5"].values[mask],
            stable_threshold,
        )

    return {
        **overall,
        "stable_threshold": float(stable_threshold),
        "seasonal": seasonal,
    }


def _statistics_for_arrays(
    sounding: np.ndarray,
    era5: np.ndarray,
    stable_threshold: float,
) -> dict[str, Any]:
    sounding = np.asarray(sounding, dtype=float)
    era5 = np.asarray(era5, dtype=float)
    valid = np.isfinite(sounding) & np.isfinite(era5)
    sounding = sounding[valid]
    era5 = era5[valid]
    n = int(len(sounding))

    if n >= 2:
        pearson_r = float(np.corrcoef(sounding, era5)[0, 1])
        spearman_r = float(stats.spearmanr(sounding, era5, nan_policy="omit").statistic)
    else:
        pearson_r = np.nan
        spearman_r = np.nan

    if n:
        diff = era5 - sounding
        rmse = float(np.sqrt(np.mean(diff**2)))
        bias = float(np.mean(diff))
    else:
        rmse = np.nan
        bias = np.nan

    sounding_stable = sounding > stable_threshold
    era5_stable = era5 > stable_threshold
    tn = int((~sounding_stable & ~era5_stable).sum())
    fp = int((~sounding_stable & era5_stable).sum())
    fn = int((sounding_stable & ~era5_stable).sum())
    tp = int((sounding_stable & era5_stable).sum())

    hit_denominator = tp + fn
    false_alarm_denominator = fp + tn
    return {
        "n": n,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "rmse": rmse,
        "bias": bias,
        "confusion_matrix": np.asarray([[tn, fp], [fn, tp]], dtype=int),
        "confusion_labels": ("non_stable", "stable"),
        "hit_rate": tp / hit_denominator if hit_denominator else np.nan,
        "false_alarm_rate": fp / false_alarm_denominator if false_alarm_denominator else np.nan,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def _write_zarr(ds: xr.Dataset, output_path: Path) -> None:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(tmp_path, mode="w", zarr_format=2)

    if output_path.exists():
        shutil.rmtree(output_path)
    tmp_path.rename(output_path)
