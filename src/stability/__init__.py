"""
Step 2: Atmospheric stability classification.

Computes two independent estimates of atmospheric stability for every hour
and grid cell:

    (a) Bulk Richardson number from gradients of T and U between 2 m and
        ~110 m (1000 hPa pressure level), discretized into 7 classes A–G
        following Stull (1988).

    (b) Pasquill-Gifford stability class from external forcings (10 m wind
        speed, total cloud cover, solar zenith angle).

Both are retained. Their agreement statistics are a paper result — they
characterize how much classification ambiguity exists in SPb conditions
and which kinds of atmospheric states are most uncertain.

────────────────────────────────────────────────────────────────────────────
Output contract — data/interim/stability.zarr
────────────────────────────────────────────────────────────────────────────

Same dimensions and coords as input ERA5 dataset (without pressure_level).
Adds variables:

    stability_ri        int8, range 0..6 mapping classes A..G respectively.
                        -1 indicates missing / could not compute.
    stability_pasquill  int8, same mapping.
    ri_value            float32, the bulk Richardson number itself,
                        retained for diagnostics and sensitivity studies.
    solar_zenith        float32, degrees, retained for traceability.

────────────────────────────────────────────────────────────────────────────
Methodological choices (see article_draft.md Section 3.1, Decision 8)
────────────────────────────────────────────────────────────────────────────

Bulk Richardson is primary. It uses actual gradients and handles coastal
SPb conditions correctly. The implementation uses:

    Ri_b = (g / T̄) · (T_upper - T_lower) · Δz / (U_upper - U_lower)²

where:
    T̄ is the mean temperature between the two levels
    g = 9.81 m/s²
    Δz is computed from the geopotential height of the upper pressure level
    minus 2 m (the height of t2m / d2m)

Pasquill is the cross-check. It is well-known to over-estimate stability
in maritime conditions and may disagree systematically with Richardson on
hours when wind blows from the Gulf of Finland. That disagreement is
itself a result for the paper.

Class boundaries for Richardson follow Stull (1988) Table 5.1 and are
defined in configs/spb_default.yaml under stability.richardson.classes.

────────────────────────────────────────────────────────────────────────────
Implementation hints (for Claude Code)
────────────────────────────────────────────────────────────────────────────

For Richardson:
- ERA5 1000 hPa geopotential height needs to be computed or assumed
  (~111 m above MSL at sea-level surface pressure, approximate).
- Use np.searchsorted or xr.apply_ufunc to map ri_value to class index.

For Pasquill:
- Solar zenith angle: use metpy.calc.solar_zenith_angle, or compute
  directly from latitude, day-of-year, and hour.
- Standard P-G table from Stull (1988) Table 9.4 — implement as a
  lookup function from (wind_speed_class, insolation_class).
- Insolation level is determined by solar elevation and cloud cover.

Diagnostic figure (docs/figures/stability_agreement.pdf):
- 7×7 confusion matrix, normalized by row, with class labels A–G.
- Optional panel: agreement by season showing whether disagreement is
  seasonally biased (it likely is — winter inversions in particular).
"""

from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_OUTPUT_PATH = Path("data/interim/stability.zarr")
CLASS_LABELS = tuple("ABCDEFG")
CLASS_TO_INT = {name: i for i, name in enumerate(CLASS_LABELS)}


def classify_stability(ds: xr.Dataset, config: dict) -> xr.Dataset:
    """Compute Richardson and Pasquill stability classes for the dataset.

    Returns a new dataset with stability_ri, stability_pasquill, ri_value,
    and solar_zenith variables added. The result is written to
    ``data/interim/stability.zarr`` unless ``stability.output_path`` is set in
    the config.
    """
    ri = richardson_bulk(ds, config)
    stability_ri = _classify_richardson_values(ri, config)
    solar_zenith = solar_zenith_angle(ds)
    stability_pasquill = _pasquill_class_from_solar_zenith(ds, config, solar_zenith)

    out = ds.drop_vars(
        [name for name in ("u", "v", "t") if name in ds],
        errors="ignore",
    )
    if "pressure_level" in out.coords:
        out = out.drop_vars("pressure_level")

    out = out.assign(
        stability_ri=stability_ri,
        stability_pasquill=stability_pasquill,
        ri_value=ri.astype("float32"),
        solar_zenith=solar_zenith.astype("float32"),
    )
    out["stability_ri"].attrs.update(
        long_name="bulk Richardson stability class",
        class_mapping="0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G, -1=missing",
    )
    out["stability_pasquill"].attrs.update(
        long_name="Pasquill-Gifford stability class",
        class_mapping="0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G, -1=missing",
    )

    output_path = Path(config.get("stability", {}).get("output_path", DEFAULT_OUTPUT_PATH))
    _write_zarr(out, output_path)
    return xr.open_zarr(output_path, consolidated=False)


def richardson_bulk(ds: xr.Dataset, config: dict) -> xr.DataArray:
    """Compute bulk Richardson number per hour and grid cell.

    The upper ERA5 1000 hPa pressure-level fields are treated as an
    approximately 110 m AGL reference level for Saint Petersburg. This is an
    approximation: SPb is close to sea level, so 1000 hPa is near the desired
    100 m layer in typical pressure conditions, but it is not a true model-level
    height.
    """
    richardson_config = config["stability"]["richardson"]
    pressure_level = richardson_config["upper_pressure_level_hpa"]
    gravity = float(richardson_config["gravity_ms2"])
    upper_height = float(richardson_config["upper_height_agl_m"])
    lower_height = float(richardson_config["lower_temperature_height_agl_m"])
    dz = upper_height - lower_height

    t_upper = _pressure_level(ds["t"], pressure_level)
    u_upper = _pressure_level(ds["u"], pressure_level)
    v_upper = _pressure_level(ds["v"], pressure_level)

    t_lower = ds["t2m"]
    t_mean = 0.5 * (t_upper + t_lower)
    shear_squared = (u_upper - ds["u10"]) ** 2 + (v_upper - ds["v10"]) ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        ri = (gravity / t_mean) * (t_upper - t_lower) * dz / shear_squared

    ri = ri.where(np.isfinite(ri))
    ri.name = "ri_value"
    ri.attrs.update(
        long_name="bulk Richardson number",
        formula="(g / T_mean) * (T_1000hPa - T_2m) * dz / ((du)^2 + (dv)^2)",
        upper_pressure_level_hpa=pressure_level,
        gravity_ms2=gravity,
        upper_height_agl_m=upper_height,
        lower_temperature_height_agl_m=lower_height,
        units="1",
    )
    return ri


def pasquill_class(ds: xr.Dataset, config: dict) -> xr.DataArray:
    """Compute Pasquill-Gifford stability class per hour and grid cell."""
    return _pasquill_class_from_solar_zenith(ds, config, solar_zenith_angle(ds))


def agreement_diagnostics(ds: xr.Dataset) -> dict:
    """Compute and return agreement statistics between Richardson and Pasquill.

    Used by the diagnostic figure and reported as a result in the paper.
    """
    ri = ds["stability_ri"].values
    pasquill = ds["stability_pasquill"].values
    valid = (ri >= 0) & (ri <= 6) & (pasquill >= 0) & (pasquill <= 6)

    confusion = np.zeros((7, 7), dtype=np.int64)
    np.add.at(confusion, (ri[valid].astype(int), pasquill[valid].astype(int)), 1)

    row_totals = confusion.sum(axis=1, keepdims=True)
    row_normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(confusion, dtype=float),
        where=row_totals > 0,
    )
    n_valid = int(valid.sum())
    exact_agreement_count = int(np.trace(confusion))
    exact_agreement_fraction = exact_agreement_count / n_valid if n_valid else np.nan

    ri_stable = ri[valid] >= CLASS_TO_INT["E"]
    pasquill_stable = pasquill[valid] >= CLASS_TO_INT["E"]
    agreement_count = int((ri_stable == pasquill_stable).sum())
    agreement_fraction = agreement_count / n_valid if n_valid else np.nan

    return {
        "labels": CLASS_LABELS,
        "confusion_matrix": confusion,
        "row_normalized": row_normalized,
        "agreement_count": agreement_count,
        "n_valid": n_valid,
        "agreement_fraction": agreement_fraction,
        "agreement_basis": "stable_classes_EFG_vs_nonstable_ABCD",
        "exact_class_agreement_count": exact_agreement_count,
        "exact_class_agreement_fraction": exact_agreement_fraction,
    }


def solar_zenith_angle(ds: xr.Dataset) -> xr.DataArray:
    """Solar zenith angle in degrees for each ERA5 time/latitude/longitude.

    Uses the NOAA fractional-year approximation with ERA5 UTC timestamps and
    the longitude correction for local solar time. Accuracy is well within the
    needs of the Pasquill insolation bins.
    """
    time_index = pd.DatetimeIndex(ds["time"].values)
    fractional_hour = (
        time_index.hour
        + time_index.minute / 60.0
        + time_index.second / 3600.0
        + time_index.microsecond / 3_600_000_000.0
    )
    gamma = 2.0 * np.pi / 365.0 * (time_index.dayofyear - 1 + (fractional_hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    utc_minutes = xr.DataArray(
        fractional_hour * 60.0,
        dims=("time",),
        coords={"time": ds["time"]},
    )
    eqtime_da = xr.DataArray(eqtime, dims=("time",), coords={"time": ds["time"]})
    decl_da = xr.DataArray(decl, dims=("time",), coords={"time": ds["time"]})
    lat_rad = np.deg2rad(ds["latitude"])
    true_solar_time = (utc_minutes + eqtime_da + 4.0 * ds["longitude"]) % 1440.0
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)

    cos_zenith = (
        np.sin(lat_rad) * np.sin(decl_da)
        + np.cos(lat_rad) * np.cos(decl_da) * np.cos(hour_angle)
    )
    zenith = np.rad2deg(np.arccos(cos_zenith.clip(min=-1.0, max=1.0)))
    zenith = zenith.transpose("time", "latitude", "longitude")
    zenith.name = "solar_zenith"
    zenith.attrs.update(long_name="solar zenith angle", units="degrees")
    return zenith


def plot_stability_agreement(ds: xr.Dataset, output_path: Path) -> None:
    """Write the row-normalized Richardson × Pasquill confusion matrix figure."""
    import matplotlib.pyplot as plt

    from src.viz import plotting_config

    diagnostics = agreement_diagnostics(ds)
    counts = diagnostics["confusion_matrix"]
    normalized = diagnostics["row_normalized"]
    labels = diagnostics["labels"]

    with plt.rc_context(plotting_config()):
        fig, ax = plt.subplots(figsize=(5.8, 4.9), constrained_layout=True)
        im = ax.imshow(normalized, cmap="viridis", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row-normalized fraction")

        ax.set_xticks(np.arange(7), labels=labels)
        ax.set_yticks(np.arange(7), labels=labels)
        ax.set_xlabel("Pasquill-Gifford class")
        ax.set_ylabel("Bulk Richardson class")
        ax.set_title(
            "Stability-class agreement\n"
            f"stable/non-stable: {diagnostics['agreement_fraction']:.0%}; "
            f"exact class: {diagnostics['exact_class_agreement_fraction']:.0%}"
        )

        for i in range(7):
            for j in range(7):
                value = normalized[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.0%}\n({counts[i, j]:,})",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)


def _pressure_level(da: xr.DataArray, pressure_level: float) -> xr.DataArray:
    selected = da.sel(pressure_level=pressure_level, method="nearest")
    return selected.drop_vars("pressure_level", errors="ignore")


def _classify_richardson_values(ri: xr.DataArray, config: dict) -> xr.DataArray:
    boundaries = np.asarray(
        [entry["ri_max"] for entry in config["stability"]["richardson"]["classes"]],
        dtype=float,
    )
    classes = xr.apply_ufunc(
        _search_class_boundaries,
        ri,
        input_core_dims=[[]],
        output_core_dims=[[]],
        kwargs={"boundaries": boundaries},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int8],
    )
    classes = classes.where(np.isfinite(ri), -1).astype("int8")
    classes.name = "stability_ri"
    return classes


def _search_class_boundaries(value: float, boundaries: np.ndarray) -> np.int8:
    if not np.isfinite(value):
        return np.int8(-1)
    return np.int8(np.clip(np.searchsorted(boundaries, value, side="left"), 0, 6))


def _pasquill_class_from_solar_zenith(
    ds: xr.Dataset, config: dict, solar_zenith: xr.DataArray
) -> xr.DataArray:
    pasquill_config = config["stability"]["pasquill"]
    wind_speed = np.hypot(ds["u10"], ds["v10"])
    cloud = ds["tcc"].clip(min=0.0, max=1.0)

    bins = _pasquill_lookup(
        wind_speed.values,
        cloud.values,
        solar_zenith.values,
        pasquill_config,
    )
    out = xr.DataArray(
        bins.astype("int8"),
        dims=wind_speed.dims,
        coords=wind_speed.coords,
        name="stability_pasquill",
    )
    out = out.where(np.isfinite(wind_speed) & np.isfinite(cloud) & np.isfinite(solar_zenith), -1)
    out.attrs.update(
        long_name="Pasquill-Gifford stability class",
        reference="Stull (1988), Table 9.4",
    )
    return out.astype("int8")


def _pasquill_lookup(
    wind_speed: np.ndarray,
    cloud: np.ndarray,
    solar_zenith: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    day_threshold = float(config["solar_zenith_day_threshold_deg"])
    strong_threshold = float(config["strong_insolation_zenith_max_deg"])
    moderate_threshold = float(config["moderate_insolation_zenith_max_deg"])
    cloud_downgrade = float(config["day_cloud_downgrade_fraction"])
    overcast = float(config["day_overcast_fraction"])
    night_cloudy = float(config["night_cloudy_fraction"])
    very_calm = float(config["very_calm_wind_ms"])

    wind_bin = np.digitize(wind_speed, config["wind_speed_bins_ms"], right=False)
    result = np.full(wind_speed.shape, CLASS_TO_INT["D"], dtype=np.int8)

    is_day = solar_zenith < day_threshold
    insolation = np.full(wind_speed.shape, 2, dtype=np.int8)
    insolation[solar_zenith <= strong_threshold] = 0
    moderate = (solar_zenith > strong_threshold) & (solar_zenith <= moderate_threshold)
    insolation[moderate] = 1
    insolation[cloud >= cloud_downgrade] = np.minimum(insolation[cloud >= cloud_downgrade] + 1, 2)

    # Rows: strong, moderate, slight insolation. Columns: wind-speed bins
    # <2, 2-3, 3-5, 5-6, >=6 m/s. Split classes in Stull's table are rounded
    # toward the more stable class for reproducible integer A-G output.
    day_table = np.asarray(
        [
            [CLASS_TO_INT["A"], CLASS_TO_INT["B"], CLASS_TO_INT["B"], CLASS_TO_INT["C"], CLASS_TO_INT["C"]],
            [CLASS_TO_INT["B"], CLASS_TO_INT["B"], CLASS_TO_INT["C"], CLASS_TO_INT["D"], CLASS_TO_INT["D"]],
            [CLASS_TO_INT["B"], CLASS_TO_INT["C"], CLASS_TO_INT["C"], CLASS_TO_INT["D"], CLASS_TO_INT["D"]],
        ],
        dtype=np.int8,
    )
    result[is_day] = day_table[insolation[is_day], wind_bin[is_day]]
    result[is_day & (cloud >= overcast)] = CLASS_TO_INT["D"]

    is_night = ~is_day
    cloudy_night = is_night & (cloud >= night_cloudy)
    clear_night = is_night & ~cloudy_night
    cloudy_table = np.asarray(
        [CLASS_TO_INT["E"], CLASS_TO_INT["E"], CLASS_TO_INT["D"], CLASS_TO_INT["D"], CLASS_TO_INT["D"]],
        dtype=np.int8,
    )
    clear_table = np.asarray(
        [CLASS_TO_INT["F"], CLASS_TO_INT["F"], CLASS_TO_INT["E"], CLASS_TO_INT["D"], CLASS_TO_INT["D"]],
        dtype=np.int8,
    )
    result[cloudy_night] = cloudy_table[wind_bin[cloudy_night]]
    result[clear_night] = clear_table[wind_bin[clear_night]]
    result[clear_night & (wind_speed < very_calm)] = CLASS_TO_INT["G"]

    invalid = ~np.isfinite(wind_speed) | ~np.isfinite(cloud) | ~np.isfinite(solar_zenith)
    result[invalid] = -1
    return result


def _write_zarr(ds: xr.Dataset, output_path: Path) -> None:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(tmp_path, mode="w", zarr_format=2)

    if output_path.exists():
        shutil.rmtree(output_path)
    tmp_path.rename(output_path)
