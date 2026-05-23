"""
Meteostat usability audit for surface-station validation of ERA5 wind fields.

Runs six steps:
  1. Station discovery within 50 km of Saint Petersburg
  2. Variable inventory for top candidates (test month June 2023)
  3. Temporal coverage 2014-2024 for top + backup station
  4. Sanity checks on meteorological values
  5. ERA5 pairing feasibility (if era5_spb.zarr exists)
  6. Generate Russian-language audit report docs/meteostat_audit.md
"""

import logging
import math
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _df_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table without tabulate."""
    if index:
        df = df.reset_index()
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, sep] + rows)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DOCS_AUDIT = ROOT / "docs" / "meteostat_audit"
COVERAGE_DIR = DOCS_AUDIT / "coverage"
SANITY_DIR = DOCS_AUDIT / "sanity"
ERA5_DIR = DOCS_AUDIT / "era5_vs_station"
INTERIM = ROOT / "data" / "interim"

for _d in [DOCS_AUDIT, COVERAGE_DIR, SANITY_DIR, ERA5_DIR, INTERIM]:
    _d.mkdir(parents=True, exist_ok=True)

# Saint Petersburg city centre
SPB_LAT, SPB_LON = 59.94, 30.31
# Pulkovo airport approximate location
PULKOVO_LAT, PULKOVO_LON = 59.80, 30.27

PERIOD_START = datetime(2014, 1, 1)
PERIOD_END = datetime(2024, 12, 31, 23, 59)
TEST_START = datetime(2023, 6, 1)
TEST_END = datetime(2023, 6, 30, 23, 59)

EXPECTED_HOURS = 11 * 365.25 * 24  # ~96 435


# ---------------------------------------------------------------------------
# Step 1: Station discovery
# ---------------------------------------------------------------------------

def step1_station_discovery() -> dict:
    """
    Find all meteostat stations within 50 km of Saint Petersburg.

    Verification: Pulkovo (ICAO ULLI / WMO 26063) must appear in the result.
    Returns dict with 'stations_df' and 'top_candidates' (list of station IDs).
    """
    log.info("Step 1: station discovery")
    from meteostat import stations as stations_db, Point

    spb_point = Point(SPB_LAT, SPB_LON)
    df = stations_db.nearby(spb_point, radius=50_000, limit=50)

    if df.empty:
        raise RuntimeError("No stations found within 50 km of Saint Petersburg")

    log.info("Found %d stations within 50 km", len(df))

    print("\n=== Stations within 50 km of Saint Petersburg ===")
    print(df.to_string())

    # Check for Pulkovo — look for ULLI in identifiers or name containing Pulkovo
    pulkovo_sid = None
    pulkovo_found = False
    for sid in df.index:
        name = str(df.loc[sid, "name"]).lower() if "name" in df.columns else ""
        if "pulkovo" in name or "ulli" in name:
            pulkovo_found = True
            pulkovo_sid = sid
            log.info("✓ Pulkovo found: station ID %s ('%s')", sid, df.loc[sid, "name"])
            break
    if not pulkovo_found:
        log.warning("Pulkovo not found by name; taking top station as primary")

    # Try to determine coverage from inventory for each station
    candidate_info = []
    for sid in df.index[:10]:
        try:
            inv = stations_db.inventory(sid)
            inv_df = inv.df if hasattr(inv, "df") else None
            if inv_df is not None and not inv_df.empty:
                param_level = inv_df.index.get_level_values("parameter").str.lower()
                hourly_rows = inv_df[param_level.isin(["wspd", "wdir", "temp"])]
                start_yr = hourly_rows["start"].min() if not hourly_rows.empty else None
                end_yr = hourly_rows["end"].max() if not hourly_rows.empty else None
            else:
                start_yr = None
                end_yr = None
        except Exception:
            start_yr = None
            end_yr = None

        candidate_info.append({
            "station_id": sid,
            "name": df.loc[sid, "name"] if "name" in df.columns else "",
            "distance_m": df.loc[sid, "distance"] if "distance" in df.columns else None,
            "inv_start": start_yr,
            "inv_end": end_yr,
        })

    cand_df = pd.DataFrame(candidate_info)
    log.info("Candidate overview:\n%s", cand_df.to_string(index=False))

    # Top candidates: prefer Pulkovo first, then closest
    top_ids = []
    if pulkovo_sid and pulkovo_sid in df.index:
        top_ids.append(pulkovo_sid)
    for sid in df.index:
        if sid not in top_ids:
            top_ids.append(sid)
        if len(top_ids) >= 5:
            break

    out_path = DOCS_AUDIT / "stations.csv"
    df.to_csv(out_path)
    log.info("Saved station list to %s", out_path)

    return {
        "stations_df": df,
        "top_candidates": top_ids,
        "pulkovo_found": pulkovo_found,
        "pulkovo_sid": pulkovo_sid,
    }


# ---------------------------------------------------------------------------
# Step 2: Variable inventory for test month
# ---------------------------------------------------------------------------

CRITICAL_VARS = ["wspd", "wdir", "temp"]
ALL_VARS = ["temp", "dwpt", "rhum", "wdir", "wspd", "wpgt", "pres"]


def step2_variable_inventory(top_candidates: list) -> dict:
    """
    Pull June 2023 for top candidates and report non-null fraction per variable.

    Verification: critical triple (wspd, wdir, temp) must all be present.
    Returns dict with 'availability_df'.
    """
    log.info("Step 2: variable inventory for test month June 2023")
    from meteostat import hourly

    records = []
    for sid in top_candidates[:5]:
        log.info("  Fetching test month for station %s", sid)
        try:
            ts = hourly(sid, TEST_START, TEST_END)
            data = ts.fetch()
        except Exception as exc:
            log.warning("  Station %s fetch failed: %s", sid, exc)
            continue

        if data is None or data.empty:
            log.warning("  Station %s: no data returned for test month", sid)
            continue

        # Flatten multi-index if present
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index(level=0, drop=True)

        log.info("  Station %s: %d rows, columns: %s", sid, len(data), list(data.columns))

        row = {"station_id": sid, "n_rows": len(data)}
        for var in ALL_VARS:
            if var in data.columns:
                row[f"{var}_pct"] = round(data[var].notna().mean() * 100, 1)
            else:
                row[f"{var}_pct"] = float("nan")
        records.append(row)

    if not records:
        raise RuntimeError("No data returned for any candidate station in test month")

    avail_df = pd.DataFrame(records).set_index("station_id")
    print("\n=== Variable availability (%, June 2023) ===")
    print(avail_df.to_string())

    out_path = DOCS_AUDIT / "variable_availability.csv"
    avail_df.to_csv(out_path)
    log.info("Saved variable availability to %s", out_path)

    return {"availability_df": avail_df}


# ---------------------------------------------------------------------------
# Step 3: Temporal coverage 2014-2024
# ---------------------------------------------------------------------------

def step3_temporal_coverage(top_candidates: list) -> dict:
    """
    Pull full 2014-2024 record for top and backup candidate.

    Verification thresholds:
      > 95%  → usable as-is
      80-95% → usable with caveats
      < 80%  → flag as concerning
    Returns dict with coverage stats for each station.
    """
    log.info("Step 3: temporal coverage 2014-2024")
    from meteostat import hourly, config as meteostat_config
    meteostat_config.block_large_requests = False

    results = {}
    for rank, sid in enumerate(top_candidates[:2]):
        label = "primary" if rank == 0 else "backup"
        log.info("  Fetching full period for station %s (%s)", sid, label)

        try:
            ts = hourly(sid, PERIOD_START, PERIOD_END)
            data = ts.fetch()
        except Exception as exc:
            log.warning("  Station %s: fetch failed: %s", sid, exc)
            results[sid] = {"label": label, "error": str(exc)}
            continue

        if data is None or data.empty:
            log.warning("  Station %s: no data returned", sid)
            results[sid] = {"label": label, "error": "empty"}
            continue

        # Flatten multi-index if present
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index(level=0, drop=True)

        # Cache to parquet
        parquet_path = INTERIM / f"meteostat_{sid}.parquet"
        data.to_parquet(parquet_path)
        log.info("  Cached %d rows to %s", len(data), parquet_path)

        n_rows = len(data)
        has_wspd = "wspd" in data.columns
        has_wdir = "wdir" in data.columns
        has_temp = "temp" in data.columns

        if has_wspd and has_wdir and has_temp:
            critical_mask = data["wspd"].notna() & data["wdir"].notna() & data["temp"].notna()
        elif has_wspd and has_temp:
            critical_mask = data["wspd"].notna() & data["temp"].notna()
            log.warning("  Station %s: wdir column missing", sid)
        else:
            critical_mask = pd.Series(False, index=data.index)
            log.warning("  Station %s: critical columns missing: %s", sid, list(data.columns))

        n_critical = critical_mask.sum()
        coverage_frac = n_critical / EXPECTED_HOURS

        if coverage_frac > 0.95:
            verdict = "usable"
        elif coverage_frac > 0.80:
            verdict = "usable_with_caveats"
        else:
            verdict = "concerning"

        log.info(
            "  Station %s: %d rows, critical-triple coverage %.1f%% → %s",
            sid, n_rows, coverage_frac * 100, verdict,
        )

        # Coverage by year
        data_with_critical = data.copy()
        data_with_critical["_critical"] = critical_mask
        yearly = data_with_critical.groupby(data_with_critical.index.year)["_critical"]
        yearly_frac = yearly.sum() / yearly.count()

        fig, ax = plt.subplots(figsize=(8, 4))
        yearly_frac.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.axhline(0.95, color="green", linestyle="--", linewidth=1, label="95%")
        ax.axhline(0.80, color="orange", linestyle="--", linewidth=1, label="80%")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Year")
        ax.set_ylabel("Critical-triple coverage fraction")
        ax.set_title(f"Station {sid} ({label}) — annual coverage")
        ax.legend()
        fig.tight_layout()
        fig.savefig(COVERAGE_DIR / f"{sid}_annual_coverage.png", dpi=150)
        plt.close(fig)

        # Coverage by month
        monthly = data_with_critical.groupby(data_with_critical.index.month)["_critical"]
        monthly_frac = monthly.sum() / monthly.count()

        fig, ax = plt.subplots(figsize=(8, 4))
        monthly_frac.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Month")
        ax.set_ylabel("Coverage fraction")
        ax.set_title(f"Station {sid} ({label}) — monthly coverage (climatological)")
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45)
        fig.tight_layout()
        fig.savefig(COVERAGE_DIR / f"{sid}_monthly_coverage.png", dpi=150)
        plt.close(fig)

        # Top-10 largest gaps
        is_gap = ~critical_mask
        gap_starts = is_gap & (~is_gap.shift(1, fill_value=False))
        gap_ends = is_gap & (~is_gap.shift(-1, fill_value=False))
        gs = data.index[gap_starts].tolist()
        ge = data.index[gap_ends].tolist()
        gaps = []
        for s, e in zip(gs, ge):
            dur = int((e - s).total_seconds() / 3600) + 1
            gaps.append({"start": s, "end": e, "duration_h": dur})
        if gaps:
            gaps_df = pd.DataFrame(gaps).sort_values("duration_h", ascending=False).head(10)
            print(f"\n=== Top-10 gaps for station {sid} ===")
            print(gaps_df.to_string(index=False))
            gaps_df.to_csv(COVERAGE_DIR / f"{sid}_top_gaps.csv", index=False)

        results[sid] = {
            "label": label,
            "n_rows": n_rows,
            "n_critical": int(n_critical),
            "coverage_frac": float(coverage_frac),
            "verdict": verdict,
            "data": data,
        }

    return results


# ---------------------------------------------------------------------------
# Step 4: Sanity checks
# ---------------------------------------------------------------------------

def step4_sanity_checks(coverage_results: dict) -> dict:
    """
    Check value distributions for the primary station over 2014-2024.

    Flags: median wind < 2 m/s or > 6 m/s suggests unit confusion.
    Flags: uniform wind rose suggests data corruption.
    Returns dict with any flag messages.
    """
    log.info("Step 4: sanity checks on values")

    primary_sid = None
    primary_data = None
    for sid, res in coverage_results.items():
        if res.get("label") == "primary" and "data" in res:
            primary_sid = sid
            primary_data = res["data"]
            break
    if primary_data is None:
        for sid, res in coverage_results.items():
            if "data" in res:
                primary_sid = sid
                primary_data = res["data"]
                break

    if primary_data is None:
        log.warning("No data available for sanity checks")
        return {"flags": ["no data"]}

    flags = []
    data = primary_data.copy()

    # Wind speed in m/s
    if "wspd" in data.columns:
        wspd_ms = data["wspd"].dropna() / 3.6
        median_ws = wspd_ms.median()
        p99_ws = wspd_ms.quantile(0.99)
        log.info("  Wind speed: median=%.2f m/s, p99=%.2f m/s", median_ws, p99_ws)
        if median_ws > 6.0:
            flags.append(f"WARN: wind speed median {median_ws:.1f} m/s > 6 m/s — possible unit issue")
        if median_ws < 2.0:
            flags.append(f"WARN: wind speed median {median_ws:.1f} m/s < 2 m/s — check units")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(wspd_ms, bins=50, color="steelblue", edgecolor="white", density=True)
        ax.axvline(median_ws, color="red", linestyle="--", label=f"median={median_ws:.1f} m/s")
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Density")
        ax.set_title(f"Station {primary_sid} — wind speed distribution 2014–2024")
        ax.legend()
        fig.tight_layout()
        fig.savefig(SANITY_DIR / f"{primary_sid}_wspd_hist.png", dpi=150)
        plt.close(fig)

    # Wind direction rose
    if "wdir" in data.columns:
        wdir = data["wdir"].dropna()
        sectors = np.arange(0, 360, 22.5)
        counts, _ = np.histogram(wdir, bins=np.append(sectors, 360))
        counts_norm = counts / counts.sum()
        uniformity = counts_norm.std()
        log.info("  Wind direction std of sector fractions: %.4f", uniformity)
        if uniformity < 0.01:
            flags.append("WARN: wind direction distribution nearly uniform — possible corruption")

        from matplotlib.projections.polar import PolarAxes
        from typing import cast
        fig = plt.figure(figsize=(6, 6))
        ax_polar = cast(PolarAxes, fig.add_subplot(111, projection="polar"))
        theta = np.deg2rad(sectors)
        width = np.deg2rad(22.5)
        ax_polar.bar(theta, counts_norm, width=width, align="center",
                     color="steelblue", edgecolor="white", alpha=0.8)
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)
        ax_polar.set_title(f"Station {primary_sid} — wind rose 2014–2024", pad=15)
        fig.tight_layout()
        fig.savefig(SANITY_DIR / f"{primary_sid}_wind_rose.png", dpi=150)
        plt.close(fig)

    # Temperature by season
    if "temp" in data.columns:
        temp = data["temp"].dropna()
        temp_min, temp_max = temp.min(), temp.max()
        log.info("  Temperature range: %.1f to %.1f °C", temp_min, temp_max)
        if temp_min < -40 or temp_max > 40:
            flags.append(f"WARN: temperature out of expected range [{temp_min:.0f}, {temp_max:.0f}] °C")

        season_map = {12: "DJF", 1: "DJF", 2: "DJF",
                      3: "MAM", 4: "MAM", 5: "MAM",
                      6: "JJA", 7: "JJA", 8: "JJA",
                      9: "SON", 10: "SON", 11: "SON"}
        data["_season"] = data.index.month.map(season_map)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, season in zip(axes.flat, ["DJF", "MAM", "JJA", "SON"]):
            vals = data.loc[data["_season"] == season, "temp"].dropna()
            ax.hist(vals, bins=40, color="steelblue", edgecolor="white", density=True)
            ax.set_title(season)
            ax.set_xlabel("T (°C)")
        fig.suptitle(f"Station {primary_sid} — temperature by season 2014–2024")
        fig.tight_layout()
        fig.savefig(SANITY_DIR / f"{primary_sid}_temp_seasons.png", dpi=150)
        plt.close(fig)

    # Sentinel values
    for col, sentinels in [("wspd", [-999, 9999, 0]), ("pres", [9999, 1013.25])]:
        if col in data.columns:
            for sv in sentinels:
                n_sv = (data[col] == sv).sum()
                if n_sv > 100:
                    flags.append(f"WARN: {col}=={sv} repeated {n_sv} times (sentinel?)")

    if flags:
        log.warning("Sanity flags:\n  " + "\n  ".join(flags))
    else:
        log.info("  No sanity flags raised")

    return {"primary_sid": primary_sid, "flags": flags}


# ---------------------------------------------------------------------------
# Step 5: ERA5 pairing
# ---------------------------------------------------------------------------

def step5_era5_pairing(coverage_results: dict) -> dict:
    """
    Compare meteostat Pulkovo data with nearest ERA5 grid cell.

    Returns dict with pairing statistics, or {'skipped': reason} if ERA5 unavailable.
    """
    log.info("Step 5: ERA5 pairing feasibility")

    era5_path = INTERIM / "era5_spb.zarr"
    if not era5_path.exists():
        log.info("  era5_spb.zarr not found — Step 5 skipped")
        return {"skipped": "era5_spb.zarr not present"}

    primary_sid = None
    station_data = None
    for sid, res in coverage_results.items():
        if res.get("label") == "primary" and "data" in res:
            primary_sid = sid
            station_data = res["data"].copy()
            break
    if station_data is None:
        return {"skipped": "no station data available"}

    log.info("  Loading ERA5 dataset")
    import xarray as xr
    ds = xr.open_zarr(era5_path)

    lat_idx = abs(ds.latitude - PULKOVO_LAT).argmin().item()
    lon_idx = abs(ds.longitude - PULKOVO_LON).argmin().item()
    cell_lat = float(ds.latitude[lat_idx])
    cell_lon = float(ds.longitude[lon_idx])
    log.info("  Using ERA5 cell: %.2f°N, %.2f°E", cell_lat, cell_lon)

    u10 = ds["u10"].isel(latitude=lat_idx, longitude=lon_idx)
    v10 = ds["v10"].isel(latitude=lat_idx, longitude=lon_idx)
    t2m = ds["t2m"].isel(latitude=lat_idx, longitude=lon_idx)

    era5_df = pd.DataFrame({
        "u10": u10.values,
        "v10": v10.values,
        "t2m": t2m.values - 273.15,
    }, index=pd.to_datetime(ds.time.values))

    era5_df["wspd_era5"] = np.sqrt(era5_df["u10"] ** 2 + era5_df["v10"] ** 2)
    era5_df["wdir_era5"] = (np.degrees(np.arctan2(-era5_df["u10"], -era5_df["v10"])) % 360)

    if "wspd" not in station_data.columns:
        return {"skipped": "station lacks wspd column"}

    station_data["wspd_ms"] = station_data["wspd"] / 3.6

    # Ensure tz-naive UTC index
    if isinstance(station_data.index, pd.DatetimeIndex) and station_data.index.tz is not None:
        station_data.index = station_data.index.tz_convert("UTC").tz_localize(None)
    if isinstance(era5_df.index, pd.DatetimeIndex) and era5_df.index.tz is not None:
        era5_df.index = era5_df.index.tz_localize(None)

    # Flatten multi-index if needed
    if isinstance(station_data.index, pd.MultiIndex):
        station_data = station_data.reset_index(level=0, drop=True)

    wdir_col = "wdir" if "wdir" in station_data.columns else None
    join_cols = ["wspd_ms"] + ([wdir_col] if wdir_col else [])
    paired = station_data[join_cols].join(
        era5_df[["wspd_era5", "wdir_era5"]], how="inner"
    ).dropna()

    n_pairs = len(paired)
    log.info("  Valid pairs: %d", n_pairs)

    if n_pairs < 100:
        return {"skipped": f"only {n_pairs} valid pairs — insufficient"}

    r_wspd = paired["wspd_ms"].corr(paired["wspd_era5"])
    bias_wspd = (paired["wspd_era5"] - paired["wspd_ms"]).mean()
    rmse_wspd = np.sqrt(((paired["wspd_era5"] - paired["wspd_ms"]) ** 2).mean())

    if wdir_col:
        delta_rad = np.deg2rad(paired["wdir_era5"] - paired[wdir_col])
        cos_agreement = float(np.cos(delta_rad).mean())
    else:
        cos_agreement = float("nan")

    log.info("  Wind speed: r=%.3f, bias=%.2f m/s, RMSE=%.2f m/s",
             r_wspd, bias_wspd, rmse_wspd)
    log.info("  Wind direction cos-agreement: %s",
             f"{cos_agreement:.3f}" if not math.isnan(cos_agreement) else "n/a")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    sample = paired.sample(min(5000, n_pairs), random_state=42)
    ax.scatter(sample["wspd_ms"], sample["wspd_era5"], s=4, alpha=0.3, color="steelblue")
    lim = max(paired["wspd_ms"].quantile(0.99), paired["wspd_era5"].quantile(0.99)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="1:1")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Station wind speed (m/s)")
    ax.set_ylabel("ERA5 wind speed (m/s)")
    ax.set_title(
        f"ERA5 vs. Pulkovo wind speed\n"
        f"r={r_wspd:.3f}, bias={bias_wspd:+.2f} m/s, RMSE={rmse_wspd:.2f} m/s"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(ERA5_DIR / f"{primary_sid}_wspd_scatter.png", dpi=150)
    plt.close(fig)

    if r_wspd > 0.8 and (math.isnan(cos_agreement) or cos_agreement > 0.7):
        era5_verdict = "good"
    elif r_wspd > 0.65 or (not math.isnan(cos_agreement) and cos_agreement > 0.55):
        era5_verdict = "moderate"
    else:
        era5_verdict = "poor"

    return {
        "primary_sid": primary_sid,
        "cell_lat": cell_lat,
        "cell_lon": cell_lon,
        "n_pairs": n_pairs,
        "r_wspd": r_wspd,
        "bias_wspd": bias_wspd,
        "rmse_wspd": rmse_wspd,
        "cos_agreement": cos_agreement,
        "era5_verdict": era5_verdict,
    }


# ---------------------------------------------------------------------------
# Step 6: Report
# ---------------------------------------------------------------------------

def step6_generate_report(s1, s2, s3, s4, s5, stations_df) -> None:
    """Write docs/meteostat_audit.md in Russian academic prose."""
    log.info("Step 6: generating audit report")

    primary_sid = None
    primary_coverage = None
    for sid, res in s3.items():
        if res.get("label") == "primary":
            primary_sid = sid
            primary_coverage = res.get("coverage_frac")
            break

    s3_verdict = s3.get(primary_sid, {}).get("verdict", "unknown") if primary_sid else "unknown"
    era5_ran = "skipped" not in s5
    era5_verdict = s5.get("era5_verdict", "n/a")

    if s3_verdict == "usable" and (not era5_ran or era5_verdict == "good"):
        overall_verdict = "positive"
        verdict_text = (
            "Meteostat пригоден для валидации ветрового компонента $p_{fav}$ в текущей форме. "
            "Рекомендуется перейти к разработке раздела §4.X статьи по наземной валидации."
        )
    elif s3_verdict in ("usable", "usable_with_caveats") and era5_verdict in ("good", "moderate", "n/a"):
        overall_verdict = "conditional"
        verdict_text = (
            "Meteostat пригоден с оговорками. "
            "Необходимо документировать пробелы в покрытии и проверить согласованность данных "
            "за годы с пониженным покрытием перед включением в валидационный анализ."
        )
    else:
        overall_verdict = "negative"
        verdict_text = (
            "Meteostat в данной форме не пригоден для прямой валидации ввиду недостаточного "
            "покрытия критической тройки переменных. Рекомендуется обратиться к альтернативным "
            "источникам: rp5.ru, NOAA ISD Lite (библиотека `noaa_isd_lite`), или архивы ВНИИГМИ-МЦД."
        )

    coverage_pct = f"{primary_coverage * 100:.1f}%" if primary_coverage is not None else "н/д"

    if stations_df is not None and not stations_df.empty:
        disp = stations_df.copy()
        keep = [c for c in ["name", "country", "latitude", "longitude",
                             "elevation", "distance"] if c in disp.columns]
        disp = disp[keep].head(10)
        if "distance" in disp.columns:
            disp["distance"] = disp["distance"].round(0).astype(int)
        station_table = _df_to_markdown(disp, index=True)
    else:
        station_table = "_Таблица недоступна_"

    avail_df = s2.get("availability_df")
    if avail_df is not None and not avail_df.empty:
        avail_table = _df_to_markdown(avail_df, index=True)
    else:
        avail_table = "_Таблица недоступна_"

    if era5_ran:
        ca = s5["cos_agreement"]
        ca_str = f"{ca:.3f}" if not math.isnan(ca) else "н/д"
        era5_text = f"""
Сопоставление выполнено для станции {s5['primary_sid']} с ближайшей ячейкой ERA5
({s5['cell_lat']:.2f}°N, {s5['cell_lon']:.2f}°E).

| Метрика | Значение |
|---|---|
| Число совпадающих пар | {s5['n_pairs']:,} |
| Корреляция Пирсона (скорость ветра) | {s5['r_wspd']:.3f} |
| Смещение ERA5 − станция (м/с) | {s5['bias_wspd']:+.2f} |
| RMSE скорости ветра (м/с) | {s5['rmse_wspd']:.2f} |
| Cos-согласованность направления | {ca_str} |

Общая оценка сопоставимости: **{era5_verdict}**.

Диаграмма рассеяния: `docs/meteostat_audit/era5_vs_station/{s5['primary_sid']}_wspd_scatter.png`.
"""
    else:
        era5_text = f"""
ERA5-данные (`data/interim/era5_spb.zarr`) не обнаружены или станционные данные недоступны.
Шаг 5 пропущен. Причина: {s5.get('skipped', 'неизвестна')}.

После получения ERA5 выполнить:

```python
# в src/exploration/meteostat_audit.py → step5_era5_pairing()
```
"""

    sanity_flags = s4.get("flags", [])
    flags_text = "\n".join(f"- {f}" for f in sanity_flags) if sanity_flags else "Флаги качества не выявлены."

    report = f"""# Аудит Meteostat для валидации ветрового компонента $p_{{fav}}$

_Дата аудита: {datetime.now().strftime('%Y-%m-%d')}_

---

## 1. Цель аудита

Настоящий аудит отвечает на вопрос: **пригодна ли библиотека meteostat
(агрегатор данных NOAA ISD / DWD и других архивов) для независимой валидации
ветрового компонента $p_{{wind}}$ методологии расчёта $p_{{fav}}$
по ГОСТ 31295.2-2005?**

Основной кандидат для сравнения — аэрологическая станция Пулково (ICAO ULLI).
Период интереса: 2014–2024 гг. Критическая тройка переменных: скорость ветра
(`wspd`), направление ветра (`wdir`), температура воздуха (`temp`).

---

## 2. Доступные станции в регионе СПб

Поиск в радиусе 50 км от центра Санкт-Петербурга (59.94°N, 30.31°E).

{station_table}

Наличие Пулково в базе: **{"да" if s1.get("pulkovo_found") else "нет — проверить радиус поиска"}**.

---

## 3. Доступность переменных (июнь 2023)

{avail_table}

Критическая тройка (`wspd`, `wdir`, `temp`) — необходимое условие для расчёта $p_{{wind}}$.

---

## 4. Временное покрытие 2014–2024

Основная станция: **{primary_sid or "н/д"}**.
Доля часов с заполненной критической тройкой: **{coverage_pct}**
от ожидаемых ≈ {int(EXPECTED_HOURS):,} ч.

Оценка по пороговым критериям:
- > 95 % → пригодна без оговорок
- 80–95 % → пригодна с оговорками
- < 80 % → вызывает серьёзные сомнения

Вердикт по покрытию: **{s3_verdict}**.

Графики по годам и месяцам: `docs/meteostat_audit/coverage/`.

---

## 5. Качество данных

{flags_text}

Гистограмма скорости ветра, роза ветров и распределения температуры по сезонам:
`docs/meteostat_audit/sanity/`.

---

## 6. Сопоставимость с ERA5

{era5_text}

---

## 7. Заключение и рекомендация

**{verdict_text}**

_Общая оценка:_ **{overall_verdict}**.
"""

    out_path = ROOT / "docs" / "meteostat_audit.md"
    out_path.write_text(report, encoding="utf-8")
    log.info("Report written to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import time
    t0 = time.time()

    s1 = step1_station_discovery()
    s2 = step2_variable_inventory(s1["top_candidates"])
    s3 = step3_temporal_coverage(s1["top_candidates"])
    s4 = step4_sanity_checks(s3)
    s5 = step5_era5_pairing(s3)
    step6_generate_report(s1, s2, s3, s4, s5, s1["stations_df"])

    elapsed = time.time() - t0
    log.info("Audit complete in %.0f s", elapsed)

    primary_sid = None
    primary_cov = None
    for sid, res in s3.items():
        if res.get("label") == "primary":
            primary_sid = sid
            primary_cov = res.get("coverage_frac")
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print(f"  Primary station: {primary_sid}")
    if primary_cov is not None:
        print(f"  Critical-triple coverage: {primary_cov * 100:.1f}%")
    if "skipped" not in s5:
        print(f"  ERA5 r(wspd): {s5.get('r_wspd', 'n/a'):.3f}")
        ca = s5.get("cos_agreement", float("nan"))
        print(f"  ERA5 dir cos-agreement: {ca:.3f}" if not math.isnan(ca) else "  ERA5 dir cos-agreement: n/a")
        print(f"  ERA5 verdict: {s5.get('era5_verdict')}")
    else:
        print(f"  ERA5 step: skipped ({s5['skipped']})")
    print(f"  Elapsed: {elapsed:.0f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
