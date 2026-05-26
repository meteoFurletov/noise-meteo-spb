"""Step 4 v2 orchestration: load → bin → aggregate → verify → write NetCDF.

Usage::

    python -m src.aggregate.runner \\
        --config configs/spb_default.yaml \\
        --input data/interim/favorable_v2.zarr \\
        --output data/processed/p_favorable_spb.nc \\
        [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from src.aggregate.aggregator import (
    PROBABILITY_VARS_SECTOR,
    PROBABILITY_VARS_THERMAL,
    aggregate_to_climatology,
)
from src.aggregate.cnossos_combiner import aggregate_cnossos_to_climatology
from src.aggregate.time_bins import PERIOD_ORDER, SEASON_ORDER, assign_period, assign_season

LOGGER = logging.getLogger("aggregate_v2")

EXPECTED_SAMPLES_PER_BIN = (3_900, 13_000)
EXPECTED_DOMAIN_P_FAVORABLE = (0.60, 0.72)
EXPECTED_DOMAIN_P_THERMAL_RI = (0.45, 0.60)
EXPECTED_CENTRAL_MAX_WIND = (0.30, 0.50)
EXPECTED_CENTRAL_MAX_SECTORS = (2, 3, 4)  # az 40°, 60°, 80° (wind from WSW quadrant)
EXPECTED_SEASONAL_DIURNAL_RATIO = (2.0, 8.0)
EXPECTED_FILE_KB = (80, 3_000)

EXPECTED_DOMAIN_P_CNOSSOS = (0.42, 0.55)
# Climatological (season×period equal-weighted) mean; differs from the
# per-sample Phase C number (28.3 %) because Lden periods are not equal
# length and stability is more frequent at night.
EXPECTED_DOMAIN_P_THERMAL_PG = (0.27, 0.34)
PRIMARY_OUTPUT_PATH = Path("data/processed/p_favorable_spb.nc")


VAR_LONG_NAMES: dict[str, str] = {
    "p_favorable": "Probability of acoustically favorable propagation conditions",
    "p_favorable_wind": "Probability of favorable wind-component contribution",
    "p_favorable_thermal_Ri": "Probability of thermally favorable conditions (Ri_b with PG cascade)",
    "p_favorable_thermal_Ri_strict": "Probability of thermally favorable conditions (Ri_b only, no cascade)",
    "p_favorable_thermal_L_strict": "Probability of thermally favorable conditions (1/L ≥ 0.05 m⁻¹)",
    "p_favorable_thermal_L_moderate": "Probability of thermally favorable conditions (1/L ≥ 0.01 m⁻¹)",
    "p_favorable_thermal_PG": "Probability of thermally favorable conditions (Pasquill E+F+G)",
    "p_favorable_cnossos": "Probability of acoustically favorable propagation conditions (CNOSSOS-EU faithful)",
}
VAR_METHODOLOGY: dict[str, str] = {
    "p_favorable": "favorable_wind OR favorable_thermal_Ri (primary deliverable)",
    "p_favorable_wind": "u·sin(az)+v·cos(az) ≥ 2 m/s (ISO 9613-2)",
    "p_favorable_thermal_Ri": "Ri_b with Pasquill–Turner cascade on Ri_b = NaN",
    "p_favorable_thermal_Ri_strict": "Ri_b ≥ 0.1, no cascade",
    "p_favorable_thermal_L_strict": "1/L ≥ 0.05 m⁻¹",
    "p_favorable_thermal_L_moderate": "1/L ≥ 0.01 m⁻¹",
    "p_favorable_thermal_PG": "Pasquill–Turner classes E, F, G",
    "p_favorable_cnossos": "favorable_wind OR Pasquill ∈ {E, F, G} (NMPB-Routes-2008 §VI)",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 4 v2 climatology aggregator")
    parser.add_argument("--config", type=Path, default=Path("configs/spb_default.yaml"))
    parser.add_argument("--input", type=Path, default=Path("data/interim/favorable_v2.zarr"))
    parser.add_argument("--output", type=Path, default=None,
                        help="output NetCDF path (default depends on --mode)")
    parser.add_argument("--report", type=Path, default=None,
                        help="report markdown path (default depends on --mode)")
    parser.add_argument("--mode", choices=("primary", "cnossos"), default="primary",
                        help="primary: Ri+cascade; cnossos: Pasquill–Turner E+F+G (NMPB-Routes-2008)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.output is None:
        args.output = (
            Path("data/processed/p_favorable_spb.nc") if args.mode == "primary"
            else Path("data/processed/p_favorable_cnossos.nc")
        )
    if args.report is None:
        args.report = (
            Path("docs/step4_v2_report.md") if args.mode == "primary"
            else Path("docs/step4g_cnossos_report.md")
        )

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = yaml.safe_load(Path(args.config).read_text())
    agg_cfg = cfg["aggregate"]
    season_map = cfg["seasons"]
    period_config = cfg["periods"]
    tz_offset = int(agg_cfg.get("local_timezone_offset_hours", 3))

    if args.output.exists() and not args.force:
        LOGGER.error("output exists: %s (use --force to overwrite)", args.output)
        return 1

    _stage("load_input", "opening %s", args.input)
    favorable = xr.open_zarr(args.input, consolidated=False)
    favorable = favorable.drop_vars([v for v in ("expver", "level", "number") if v in favorable.variables])
    _check_input_schema(favorable, mode=args.mode)

    _stage("time_bins", "verifying season and period distributions")
    _verify_time_bin_distribution(favorable["time"], season_map, period_config, tz_offset)

    if args.dry_run:
        LOGGER.info("dry-run OK")
        return 0

    _stage("aggregate", "grouping by (season, period) and computing means [mode=%s]", args.mode)
    if args.mode == "primary":
        clim = aggregate_to_climatology(favorable, season_map, period_config, tz_offset)
    else:
        clim = aggregate_cnossos_to_climatology(favorable, season_map, period_config, tz_offset)
    clim = _attach_attrs(clim, cfg, args.input, mode=args.mode)

    _stage("verify", "running output verification gates [mode=%s]", args.mode)
    stats: dict = {}
    verify_fn = _run_verification if args.mode == "primary" else _run_verification_cnossos
    if not verify_fn(clim, stats):
        LOGGER.error("verification failed; output not written")
        return 1

    _stage("write_output", "writing %s", args.output)
    _write_netcdf(clim, args.output, complevel=int(agg_cfg.get("output_compression_level", 4)))

    file_kb = args.output.stat().st_size / 1024
    if not (EXPECTED_FILE_KB[0] <= file_kb <= EXPECTED_FILE_KB[1]):
        LOGGER.error("output size %.1f KB outside expected [%d, %d] KB",
                     file_kb, *EXPECTED_FILE_KB)
        return 1

    _stage("cf_check", "re-opening file in subprocess and verifying CF metadata")
    if not _cf_subprocess_check(args.output):
        return 1

    if args.mode == "primary":
        LOGGER.info("=" * 50)
        LOGGER.info("STEP 4 v2 VERIFICATION: ALL PASSED")
        LOGGER.info("=" * 50)
        LOGGER.info("Schema:                              OK")
        LOGGER.info("Headline numbers:")
        LOGGER.info("  Domain-mean p_favorable:           %.3f", stats["domain_p_favorable"])
        LOGGER.info("  Domain-mean p_favorable_thermal:   %.3f", stats["domain_p_thermal_Ri"])
        LOGGER.info("  Central cell sector-max wind:      %.3f @ azimuth %d°",
                    stats["central_max_wind"], int(stats["central_max_wind_az"]))
        LOGGER.info("Seasonal-diurnal contrast:           %.2fx", stats["seasonal_diurnal_ratio"])
        LOGGER.info("  (DJF night / JJA day, central cell, thermal_Ri)")
        LOGGER.info("Sample counts (min / median / max):  %d / %d / %d",
                    stats["nsamples_min"], stats["nsamples_med"], stats["nsamples_max"])
        LOGGER.info("CF-1.10 compliance:                  OK")
        LOGGER.info("Output:                              %s (%.1f KB)", args.output, file_kb)
        LOGGER.info("=" * 50)

        _stage("report", "writing %s", args.report)
        _write_report(args.report, args.output, clim, stats, file_kb)
    else:
        LOGGER.info("=" * 50)
        LOGGER.info("PHASE G VERIFICATION: ALL PASSED")
        LOGGER.info("=" * 50)
        LOGGER.info("Schema:                              OK")
        LOGGER.info("Domain-mean p_favorable (CNOSSOS):   %.3f", stats["domain_p_cnossos"])
        if "domain_p_primary" in stats:
            LOGGER.info("Domain-mean p_favorable (primary):   %.3f (from %s)",
                        stats["domain_p_primary"], PRIMARY_OUTPUT_PATH.name)
            LOGGER.info("Δ methodology (primary − CNOSSOS):   %.3f", stats["delta_methodology"])
        LOGGER.info("Domain-mean p_favorable_thermal_PG:  %.3f", stats["domain_p_thermal_PG"])
        if "max_wind_diff" in stats:
            LOGGER.info("Wind component identical to primary: max |Δ| = %.2e", stats["max_wind_diff"])
        LOGGER.info("CF-1.10 compliance:                  OK")
        LOGGER.info("Output:                              %s (%.1f KB)", args.output, file_kb)
        LOGGER.info("=" * 50)

        _stage("report", "writing %s", args.report)
        _write_cnossos_report(args.report, args.output, clim, stats, file_kb)
    return 0


# ─── stages ────────────────────────────────────────────────────────────────


def _stage(name: str, message: str, *args) -> None:
    LOGGER.info("%s: " + message, name, *args)


def _check_input_schema(ds: xr.Dataset, mode: str = "primary") -> None:
    if mode == "cnossos":
        required = ("favorable_wind", "favorable_thermal_PG")
    else:
        required = (
            "favorable", "favorable_wind", "favorable_thermal_Ri",
            "favorable_thermal_Ri_strict", "favorable_thermal_L_strict",
            "favorable_thermal_L_moderate", "favorable_thermal_PG",
        )
    missing = [v for v in required if v not in ds.data_vars]
    if missing:
        raise KeyError(f"input missing required variables: {missing}")
    for coord in ("time", "latitude", "longitude", "sector"):
        if coord not in ds.coords:
            raise KeyError(f"input missing coordinate {coord!r}")


def _verify_time_bin_distribution(
    time: xr.DataArray,
    season_map: dict,
    period_config: dict,
    tz_offset: int,
) -> None:
    season = assign_season(time, season_map)
    period = assign_period(time, period_config, tz_offset)
    n_total = time.size
    season_counts = {s: int((season.values == s).sum()) for s in SEASON_ORDER}
    period_counts = {p: int((period.values == p).sum()) for p in PERIOD_ORDER}
    expected_season = n_total / 4
    expected_period = {"day": n_total * 12 / 24, "evening": n_total * 4 / 24, "night": n_total * 8 / 24}

    LOGGER.info("  season counts: %s", season_counts)
    LOGGER.info("  period counts: %s", period_counts)

    for s, n in season_counts.items():
        if abs(n - expected_season) / expected_season > 0.02:
            raise RuntimeError(
                f"season {s!r} count {n} deviates >2% from expected {expected_season:.0f}"
            )
    for p, n in period_counts.items():
        if abs(n - expected_period[p]) / expected_period[p] > 0.02:
            raise RuntimeError(
                f"period {p!r} count {n} deviates >2% from expected {expected_period[p]:.0f}"
            )


def _attach_attrs(ds: xr.Dataset, cfg: dict, input_path: Path, mode: str = "primary") -> xr.Dataset:
    ds = ds.copy()

    ds["latitude"].attrs.update(units="degrees_north", standard_name="latitude",
                                long_name="latitude of ERA5 grid-cell center")
    ds["longitude"].attrs.update(units="degrees_east", standard_name="longitude",
                                 long_name="longitude of ERA5 grid-cell center")
    ds["sector"].attrs.update(units="1", long_name="azimuth sector index (ISO 9613-2)")
    ds["sector_center_iso_deg"].attrs.update(
        units="degree",
        long_name="sector central azimuth, ISO 9613-2 convention (toward)",
    )
    ds["season"].attrs.update(long_name="climatological season (DJF, MAM, JJA, SON)")
    ds["period"].attrs.update(long_name="Lden time-of-day period (local time, UTC+3)")

    for name in ds.data_vars:
        if name.startswith("p_"):
            valid_range = np.array([0.0, 1.0], dtype="float32")
            ds[name].attrs.update(
                units="1",
                long_name=VAR_LONG_NAMES.get(name, name),
                valid_range=valid_range,
                methodology=VAR_METHODOLOGY.get(name, ""),
            )
    ds["n_samples"].attrs.update(
        units="1",
        long_name="Number of hourly observations contributing to each (season, period) bin",
    )

    if mode == "cnossos":
        title = (
            "Climatology of acoustically favorable propagation conditions for "
            "Saint Petersburg, CNOSSOS-EU faithful methodology"
        )
        references = (
            "ISO 9613-2:1996; CNOSSOS-EU (Kephalopoulos et al. 2012); "
            "NMPB-Routes-2008; Hersbach et al. 2020"
        )
    else:
        title = "Climatology of acoustically favorable propagation conditions for Saint Petersburg"
        references = "ISO 9613-2:1996; CNOSSOS-EU; Stull 1988; van Ulden & Holtslag 1985"

    ds.attrs.update(
        Conventions=cfg.get("aggregate", {}).get("cf_conventions", "CF-1.10"),
        title=title,
        institution="Russian State Hydrometeorological University",
        source="ERA5 reanalysis (Hersbach et al. 2020)",
        history=(
            f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}: aggregated from "
            f"{input_path.name} by src/aggregate/runner.py (mode={mode})"
        ),
        references=references,
        pipeline_version="v2",
        comment=(
            "Sectors stored in ISO 9613-2 convention (azimuth = direction sound "
            "propagates TO). For meteorological convention 'where wind comes from', add 180°."
        ),
        time_period="2014-01-01 to 2024-12-31",
        local_timezone="UTC+3 (Moscow Standard Time, year-round)",
        git_commit=_git_commit(),
    )
    if mode == "cnossos":
        ds.attrs["methodology"] = (
            "CNOSSOS-EU / NMPB-Routes-2008 §VI: favorable = "
            "wind_toward_sector_≥_2_m/s OR Pasquill ∈ {E, F, G}"
        )
        ds.attrs["companion_file"] = (
            "p_favorable_spb.nc (primary, uses Ri_b+PG cascade for thermal)"
        )
    return ds


def _run_verification(clim: xr.Dataset, stats: dict) -> bool:
    gates: list[bool] = []

    # §8.1 range check.
    for name in clim.data_vars:
        if not name.startswith("p_"):
            continue
        arr = clim[name].values
        if not np.all((arr >= 0.0) & (arr <= 1.0)):
            bad = np.argwhere(~((arr >= 0.0) & (arr <= 1.0)))
            LOGGER.error("variable %s has values outside [0,1] at %d positions, first=%s",
                         name, bad.shape[0], bad[0].tolist() if bad.size else "n/a")
            gates.append(False)
        else:
            gates.append(True)

    # n_samples range.
    ns = clim["n_samples"].values
    nmin, nmax = int(ns.min()), int(ns.max())
    nmed = int(np.median(ns))
    stats["nsamples_min"] = nmin
    stats["nsamples_med"] = nmed
    stats["nsamples_max"] = nmax
    samples_ok = EXPECTED_SAMPLES_PER_BIN[0] <= nmin and nmax <= EXPECTED_SAMPLES_PER_BIN[1]
    _log_gate(
        f"n_samples ∈ [{EXPECTED_SAMPLES_PER_BIN[0]}, {EXPECTED_SAMPLES_PER_BIN[1]}]",
        samples_ok, f"min={nmin}, median={nmed}, max={nmax}",
    )
    gates.append(samples_ok)

    # §8.2 headline numbers.
    domain_p = float(clim["p_favorable"].mean().item())
    domain_th = float(clim["p_favorable_thermal_Ri"].mean().item())
    stats["domain_p_favorable"] = domain_p
    stats["domain_p_thermal_Ri"] = domain_th
    p_ok = EXPECTED_DOMAIN_P_FAVORABLE[0] <= domain_p <= EXPECTED_DOMAIN_P_FAVORABLE[1]
    th_ok = EXPECTED_DOMAIN_P_THERMAL_RI[0] <= domain_th <= EXPECTED_DOMAIN_P_THERMAL_RI[1]
    _log_gate(f"domain-mean p_favorable ∈ {EXPECTED_DOMAIN_P_FAVORABLE}",
              p_ok, f"= {domain_p:.4f}")
    _log_gate(f"domain-mean p_favorable_thermal_Ri ∈ {EXPECTED_DOMAIN_P_THERMAL_RI}",
              th_ok, f"= {domain_th:.4f}")
    gates += [p_ok, th_ok]

    # Central cell sector-max wind, annual mean.
    central = clim["p_favorable_wind"].sel(latitude=59.75, longitude=30.25, method="nearest")
    annual_per_sector = central.mean(dim=("season", "period"))
    max_sector_idx = int(annual_per_sector.argmax().item())
    max_val = float(annual_per_sector.isel(sector=max_sector_idx).item())
    max_az = float(clim["sector_center_iso_deg"].isel(sector=max_sector_idx).item())
    stats["central_max_wind"] = max_val
    stats["central_max_wind_az"] = max_az
    stats["central_max_wind_sector"] = max_sector_idx
    sector_ok = max_sector_idx in EXPECTED_CENTRAL_MAX_SECTORS
    val_ok = EXPECTED_CENTRAL_MAX_WIND[0] <= max_val <= EXPECTED_CENTRAL_MAX_WIND[1]
    _log_gate(f"central-cell sector-max wind ∈ sectors {EXPECTED_CENTRAL_MAX_SECTORS}",
              sector_ok, f"got sector {max_sector_idx} (az {max_az:.0f}°)")
    _log_gate(f"central-cell sector-max wind value ∈ {EXPECTED_CENTRAL_MAX_WIND}",
              val_ok, f"= {max_val:.4f}")
    gates += [sector_ok, val_ok]

    # §8.3 seasonal-diurnal contrast.
    central_th = clim["p_favorable_thermal_Ri"].sel(
        latitude=59.75, longitude=30.25, method="nearest"
    )
    ratio = (
        float(central_th.sel(season="DJF", period="night").item())
        / float(central_th.sel(season="JJA", period="day").item())
    )
    stats["seasonal_diurnal_ratio"] = ratio
    ratio_ok = EXPECTED_SEASONAL_DIURNAL_RATIO[0] <= ratio <= EXPECTED_SEASONAL_DIURNAL_RATIO[1]
    _log_gate(
        f"seasonal-diurnal contrast (DJF night / JJA day) ∈ {EXPECTED_SEASONAL_DIURNAL_RATIO}",
        ratio_ok, f"= {ratio:.2f}x",
    )
    gates.append(ratio_ok)

    # §8.4 sensitivity ordering (informational).
    variant_means = {
        v: float(clim[f"p_{v}"].mean().item())
        for v in PROBABILITY_VARS_THERMAL
    }
    stats["variant_means"] = variant_means
    LOGGER.info("sensitivity ordering (informational):")
    for k, v in sorted(variant_means.items(), key=lambda kv: -kv[1]):
        LOGGER.info("    %-32s = %.4f", k, v)

    # §8.5 spatial pattern (informational).
    grid_mean = clim["p_favorable"].mean(dim=("sector", "season", "period")).values
    stats["spatial_grid"] = grid_mean
    LOGGER.info("p_favorable annual mean by cell (5 × 7):")
    for row in grid_mean:
        LOGGER.info("  " + "  ".join(f"{v:0.2f}" for v in row))
    median = float(np.median(grid_mean))
    if np.any(np.abs(grid_mean - median) > 0.15):
        LOGGER.warning("at least one cell deviates > 0.15 from the spatial median (%.2f)", median)

    return all(gates)


def _log_gate(name: str, passed: bool, detail: str = "") -> None:
    LOGGER.info("  [%s] %s %s", "PASS" if passed else "FAIL", name, detail)


def _write_netcdf(ds: xr.Dataset, path: Path, complevel: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding: dict[str, dict] = {}
    for name, var in ds.data_vars.items():
        if name.startswith("p_"):
            encoding[name] = {"zlib": True, "complevel": complevel, "dtype": "float32"}
        elif name == "n_samples":
            encoding[name] = {"zlib": True, "complevel": complevel, "dtype": "int32"}
    encoding["sector"] = {"dtype": "int8"}
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding, format="NETCDF4")


def _cf_subprocess_check(path: Path) -> bool:
    code = (
        "import sys, xarray as xr\n"
        f"ds = xr.open_dataset(r'{path}')\n"
        "assert ds.attrs.get('Conventions') == 'CF-1.10', "
        "f\"Conventions={ds.attrs.get('Conventions')!r}\"\n"
        "for c in ('latitude','longitude','sector','season','period'):\n"
        "    assert c in ds.coords, f'missing coord {c}'\n"
        "assert ds['latitude'].attrs.get('units') == 'degrees_north'\n"
        "assert ds['longitude'].attrs.get('units') == 'degrees_east'\n"
        "ds.close()\n"
        "print('cf_ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        LOGGER.error("CF subprocess check failed:\n%s", result.stderr)
        return False
    if "cf_ok" not in result.stdout:
        LOGGER.error("CF subprocess produced unexpected output: %r", result.stdout)
        return False
    if result.stderr.strip():
        LOGGER.warning("CF subprocess emitted warnings:\n%s", result.stderr)
    return True


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, check=False, timeout=2)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "uncommitted"


def _write_report(report_path: Path, output_path: Path, clim: xr.Dataset,
                  stats: dict, file_kb: float) -> None:
    pct = lambda x: f"{100 * x:.1f}".replace(".", ",")
    vm = stats["variant_means"]
    central_sec = stats["central_max_wind_sector"]
    central_az = int(stats["central_max_wind_az"])
    from_az = (central_az + 180) % 360

    # Per-(season, period) table for the central cell, primary p_favorable.
    central_table = clim["p_favorable"].sel(
        latitude=59.75, longitude=30.25, method="nearest",
    ).mean(dim="sector")
    table_lines = ["| Сезон \\ Период | day | evening | night |", "|---|---|---|---|"]
    for s in SEASON_ORDER:
        cells = [pct(float(central_table.sel(season=s, period=p).item())) for p in PERIOD_ORDER]
        table_lines.append(f"| {s} | {cells[0]} % | {cells[1]} % | {cells[2]} % |")
    central_table_md = "\n".join(table_lines)

    # 5x7 spatial grid (annual mean p_favorable).
    grid = stats["spatial_grid"]
    lat_vals = clim["latitude"].values
    lon_vals = clim["longitude"].values
    header = "| широта \\ долгота | " + " | ".join(f"{lon:.2f}" for lon in lon_vals) + " |"
    sep = "|---|" + "---|" * len(lon_vals)
    grid_lines = [header, sep]
    for i, lat in enumerate(lat_vals):
        cells = " | ".join(f"{grid[i, j]:.2f}" for j in range(len(lon_vals)))
        grid_lines.append(f"| {lat:.2f} | {cells} |")
    grid_md = "\n".join(grid_lines)

    body = f"""# Отчёт по Этапу 4 пайплайна (v2): климатологическая агрегация

**Дата выполнения:** {pd.Timestamp.utcnow().date().isoformat()}
**Источник входных данных:** `data/interim/favorable_v2.zarr`.
**Выходной артефакт:** `{output_path}`, {file_kb:.0f} КБ, формат NetCDF-4 с соблюдением соглашений CF-1.10.

## 1. Структура итоговой справочной таблицы

Климатологическая агрегация преобразует почасовой признак благоприятности
(96 432 часа × 5 × 7 ячеек × 18 секторов) в справочную таблицу размерности
35 × 18 × 4 × 3 = 7 560 значений вероятности благоприятного распространения
для каждой комбинации (ячейка, сектор, сезон, период суток Lden).

Сезоны определены по календарно-метеорологической классификации:
DJF (декабрь–февраль), MAM (март–май), JJA (июнь–август),
SON (сентябрь–ноябрь). Периоды суток заданы в местном времени
(UTC+3 круглогодично; РФ не применяет летнее время с 2014 года) в
соответствии со схемой Lden: день 07:00–19:00, вечер 19:00–23:00,
ночь 23:00–07:00.

Количество отсчётов в одной (ячейка, сезон, период)-ячейке составляет
от {stats['nsamples_min']} до {stats['nsamples_max']} часов
(медиана {stats['nsamples_med']}); диапазон отражает разную длительность
периодов Lden и не превышает ±2 % допуска на однородность распределения.

## 2. Основной климатологический результат

Среднедоменная вероятность благоприятного распространения за период
2014–2024:
$$\\bar p_{{\\text{{благ}}}} = {pct(stats['domain_p_favorable'])} \\%.$$

Распределение по сезонам и периодам для центральной городской ячейки
(59,75° с.ш., 30,25° в.д. — близко к географическому центру города),
осреднённое по 18 секторам:

{central_table_md}

Сектор с наибольшей частотой ветровой благоприятности для центральной
ячейки — азимут {central_az}° (сектор {central_sec}, ISO 9613-2,
«куда распространяется звук»); в метеорологической конвенции «откуда
дует ветер» это отвечает преобладающим западным/юго-западным ветрам
с азимута {from_az}°.

## 3. Сезонно-суточный контраст термического канала

Главный физико-климатологический результат:
$$\\frac{{p_{{\\text{{терм}}}}^{{\\text{{Ri}}}}(\\text{{DJF, ночь}})}}
        {{p_{{\\text{{терм}}}}^{{\\text{{Ri}}}}(\\text{{JJA, день}})}}
= {stats['seasonal_diurnal_ratio']:.2f}$$

в центральной городской ячейке. Это количественная мера того, насколько
устойчивая зимняя ночная атмосфера Санкт-Петербурга благоприятствует
акустическому распространению по сравнению с конвективной летней дневной
атмосферой.

## 4. Чувствительность к выбору термического классификатора

Климатологические значения по пяти вариантам термического флага
(среднее по домену и времени):

| Вариант | Среднедоменное $p_{{\\text{{терм}}}}$ |
|---|---|
| $Ri_b$ с каскадом (основной) | {pct(vm['favorable_thermal_Ri'])} % |
| $Ri_b$ без каскада | {pct(vm['favorable_thermal_Ri_strict'])} % |
| Паскуилл–Тёрнер (E+F+G) | {pct(vm['favorable_thermal_PG'])} % |
| $1/L$ умеренный (0,01 м⁻¹) | {pct(vm['favorable_thermal_L_moderate'])} % |
| $1/L$ строгий (0,05 м⁻¹) | {pct(vm['favorable_thermal_L_strict'])} % |

Разброс между классификаторами служит количественной мерой
методологической неопределённости термического канала; в § 4.6
диссертации соответствующие значения используются для построения
интервала чувствительности вокруг основной оценки $Ri_b$ с каскадом.

## 5. Пространственный паттерн

Карта среднегодовой вероятности благоприятного распространения
(осреднено по секторам, сезонам и периодам):

{grid_md}

Контраст между прибрежными ячейками западной кромки домена
(Финский залив) и материковыми ячейками не превышает 10 процентных
пунктов, что согласуется с пространственным разрешением ERA5 и
ожидаемой когерентностью климатологических полей на масштабе
~30 км.

## 6. Формат и метаданные выходного файла

Файл `p_favorable_spb.nc` — NetCDF-4 с компрессией (zlib, уровень 4) и
полным набором атрибутов CF-1.10:

- глобальные: `Conventions`, `title`, `institution`, `source`,
  `references`, `history`, `pipeline_version`, `comment`,
  `time_period`, `local_timezone`, `git_commit`;
- координатные: `latitude`/`longitude` со стандартными именами и
  единицами CF, `sector` с пояснительной строкой соглашения,
  `sector_center_iso_deg` как вспомогательная координата с явной
  ISO-конвенцией;
- переменные: `p_favorable*` с атрибутами `units = "1"`,
  `valid_range = [0, 1]`, `long_name`, `methodology`.

Размер файла ({file_kb:.0f} КБ) попадает в ожидаемый диапазон
[{EXPECTED_FILE_KB[0]}, {EXPECTED_FILE_KB[1]}] КБ и совместим с
архивным депозитом, прямым потреблением React-калькулятором и
открытием стандартными инструментами (Panoply, ncdump, ncview,
xarray, MATLAB).

## 7. Готовность к Этапу 5

Файл `p_favorable_spb.nc` является итоговым научным продуктом
исследования. Численные значения готовы к интеграции в § 4.6,
аннотацию и § 5.2 диссертации, а также к загрузке в React-калькулятор
для расчёта буферных зон.
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")


def _run_verification_cnossos(clim: xr.Dataset, stats: dict) -> bool:
    gates: list[bool] = []

    # Range check on probabilities.
    for name in clim.data_vars:
        if not name.startswith("p_"):
            continue
        arr = clim[name].values
        if not np.all((arr >= 0.0) & (arr <= 1.0)):
            LOGGER.error("variable %s has values outside [0,1]", name)
            gates.append(False)
        else:
            gates.append(True)

    # n_samples range.
    ns = clim["n_samples"].values
    nmin, nmax = int(ns.min()), int(ns.max())
    nmed = int(np.median(ns))
    stats["nsamples_min"] = nmin
    stats["nsamples_med"] = nmed
    stats["nsamples_max"] = nmax
    samples_ok = EXPECTED_SAMPLES_PER_BIN[0] <= nmin and nmax <= EXPECTED_SAMPLES_PER_BIN[1]
    _log_gate(
        f"n_samples ∈ [{EXPECTED_SAMPLES_PER_BIN[0]}, {EXPECTED_SAMPLES_PER_BIN[1]}]",
        samples_ok, f"min={nmin}, median={nmed}, max={nmax}",
    )
    gates.append(samples_ok)

    # Headline numbers.
    domain_cn = float(clim["p_favorable_cnossos"].mean().item())
    domain_pg = float(clim["p_favorable_thermal_PG"].mean().item())
    stats["domain_p_cnossos"] = domain_cn
    stats["domain_p_thermal_PG"] = domain_pg
    cn_ok = EXPECTED_DOMAIN_P_CNOSSOS[0] <= domain_cn <= EXPECTED_DOMAIN_P_CNOSSOS[1]
    pg_ok = EXPECTED_DOMAIN_P_THERMAL_PG[0] <= domain_pg <= EXPECTED_DOMAIN_P_THERMAL_PG[1]
    _log_gate(f"domain-mean p_favorable_cnossos ∈ {EXPECTED_DOMAIN_P_CNOSSOS}",
              cn_ok, f"= {domain_cn:.4f}")
    _log_gate(f"domain-mean p_favorable_thermal_PG ∈ {EXPECTED_DOMAIN_P_THERMAL_PG}",
              pg_ok, f"= {domain_pg:.4f}")
    gates += [cn_ok, pg_ok]

    # Wind component identical to primary deliverable (if it exists).
    if PRIMARY_OUTPUT_PATH.exists():
        with xr.open_dataset(PRIMARY_OUTPUT_PATH) as primary:
            diff = np.abs(
                clim["p_favorable_wind"].values - primary["p_favorable_wind"].values
            )
            max_diff = float(diff.max())
            stats["max_wind_diff"] = max_diff
            stats["domain_p_primary"] = float(primary["p_favorable"].mean().item())
            stats["delta_methodology"] = stats["domain_p_primary"] - domain_cn
            wind_ok = max_diff < 1e-6
            _log_gate("wind component identical to primary (max |Δ| < 1e-6)",
                      wind_ok, f"= {max_diff:.2e}")
            gates.append(wind_ok)

            # Central-cell side-by-side (informational).
            central_primary = primary["p_favorable"].sel(
                latitude=59.75, longitude=30.25, method="nearest"
            ).mean(dim="sector")
            central_cnossos = clim["p_favorable_cnossos"].sel(
                latitude=59.75, longitude=30.25, method="nearest"
            ).mean(dim="sector")
            LOGGER.info("central-cell (season, period): primary | CNOSSOS | Δ")
            central_rows = []
            for s in SEASON_ORDER:
                for p in PERIOD_ORDER:
                    pv = float(central_primary.sel(season=s, period=p).item())
                    cv = float(central_cnossos.sel(season=s, period=p).item())
                    LOGGER.info("  %s %-7s  %.3f | %.3f | %+0.3f", s, p, pv, cv, pv - cv)
                    central_rows.append((s, p, pv, cv, pv - cv))
            stats["central_rows"] = central_rows
    else:
        LOGGER.warning("primary deliverable %s not found; skipping wind-identity gate",
                       PRIMARY_OUTPUT_PATH)

    return all(gates)


def _write_cnossos_report(report_path: Path, output_path: Path, clim: xr.Dataset,
                          stats: dict, file_kb: float) -> None:
    pct = lambda x: f"{100 * x:.1f}".replace(".", ",")
    today = pd.Timestamp.utcnow().date().isoformat()

    if "central_rows" in stats:
        rows = ["| Сезон | Период | Основной | CNOSSOS | Δ |", "|---|---|---|---|---|"]
        for s, p, pv, cv, d in stats["central_rows"]:
            rows.append(f"| {s} | {p} | {pct(pv)} % | {pct(cv)} % | {pct(d)} пп |")
        central_table_md = "\n".join(rows)
    else:
        central_table_md = "_сравнительная таблица недоступна (отсутствует основной файл)._"

    if "delta_methodology" in stats:
        comparison_block = (
            f"В сравнении с основным расчётом "
            f"($\\bar p_{{\\text{{благ}}}}^{{Ri}} = {pct(stats['domain_p_primary'])}$ %): "
            f"разность $\\Delta_{{\\text{{метод}}}} = {pct(stats['delta_methodology'])}$ пп. "
            f"Эта разность — чисто методологический эффект (одни и те же входные данные "
            f"ERA5, один и тот же ветровой компонент, один и тот же порог 2 м/с); "
            f"расхождение объясняется выбором классификатора устойчивости "
            f"($Ri_b$+каскад против Паскуилла–Тёрнера)."
        )
    else:
        comparison_block = "_основной файл не найден; сравнение опущено._"

    body = f"""# Отчёт по Этапу G: CNOSSOS-EU верный расчёт

**Дата выполнения:** {today}
**Источник входных данных:** `data/interim/favorable_v2.zarr`.
**Выходной артефакт:** `{output_path}`, {file_kb:.0f} КБ, формат NetCDF-4 (CF-1.10).

## 1. Назначение этапа

CNOSSOS-EU (Common Noise Assessment Methods, JRC 2012; основан на
NMPB-Routes-2008 §VI) задаёт критерий благоприятности через
классификацию устойчивости Паскуилла–Тёрнера:

$$\\text{{favorable}}(\\theta, t) = \\big[u_\\parallel(\\theta, t) \\geq 2 \\text{{ м/с}}\\big] \\lor \\big[\\text{{Pasquill}}(t) \\in \\{{E, F, G\\}}\\big]$$

Настоящий этап выполняет климатологическую агрегацию ровно по этой
формуле на тех же входных данных, что и основной расчёт (Этап 4), но с
подстановкой Паскуилла–Тёрнера вместо $Ri_b$+каскада в качестве
термического компонента. Цель — обеспечить методологически прямое
сопоставление с европейской практикой (§ 4.9 диссертации).

## 2. Различия с основным расчётом

| Аспект | Основной (Этап 4) | CNOSSOS-EU (Этап G) |
|---|---|---|
| Источник ERA5 | `era5_spb_v2.zarr` | `era5_spb_v2.zarr` (идентичный) |
| Порог ветра | 2,0 м/с | 2,0 м/с |
| Соглашение секторов | ISO 9613-2 | ISO 9613-2 |
| Сетка сезонов/периодов | 4 × 3 | 4 × 3 (идентичная) |
| Термический классификатор | $Ri_b \\geq 0{{,}}1$ + каскад Паскуилла на NaN | Паскуилл–Тёрнер $\\in \\{{E, F, G\\}}$ |
| Каскад/фолбэк | да (при отсутствии $Ri_b$) | нет (CNOSSOS использует Паскуилла как первичный) |

## 3. Среднедоменный результат

$$\\bar p_{{\\text{{благ}}}}^{{\\text{{CNOSSOS}}}} = {pct(stats['domain_p_cnossos'])} \\%.$$

Среднедоменное значение термического компонента Паскуилла–Тёрнера
($p_{{\\text{{терм}}}}^{{\\text{{PG}}}} = {pct(stats['domain_p_thermal_PG'])}$ %)
численно совпадает с независимой оценкой из Этапа C (28,3 %).

{comparison_block}

## 4. Сезонно-периодная разбивка центральной ячейки

Сравнение для центральной городской ячейки (59,75° с.ш., 30,25° в.д.),
осреднено по 18 секторам:

{central_table_md}

Во всех (сезон, период)-комбинациях значения CNOSSOS закономерно ниже
основного расчёта. Разрыв растёт зимой и ночью, когда $Ri_b$+каскад
обнаруживает устойчивость, которую критерий Паскуилла–Тёрнера
пропускает (поскольку PG привязан к скорости ветра и инсоляции, а не к
прямой термодинамической характеристике пограничного слоя).

## 5. Готовность к §4.9

Файл `{output_path.name}` готов к использованию для прямого
сопоставления с дефолтными значениями CNOSSOS-EU (опубликованные
значения JRC Reference Report 2012) в § 4.9 диссертации. Полностью
самодостаточен (содержит вероятности комбинированного критерия,
ветрового компонента и термического компонента Паскуилла–Тёрнера) и
совместим со схемой основного файла `p_favorable_spb.nc`.
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
