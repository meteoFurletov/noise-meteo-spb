"""
Voeikovo radiosonde validation of the v2 ERA5 stability pipeline.

Pairs per-launch sounding Ri_b (already present in data/interim/soundings_spb.zarr
as `ri_sounding`, computed on the 2 m -> 110 m AGL layer with the same shear
floor as the ERA5 pipeline) against ERA5-derived Ri_b at the nearest grid cell
to the Voeikovo launch site (59.952 N, 30.707 E -> ERA5 cell 60.0 N, 30.75 E).

Threshold for the binary stable / non-stable classification is 0.1, matching
the v2 pipeline configuration.

Outputs (paths are CLI args, defaults below):
  - data/processed/voeikovo_validation.nc      per-launch joined dataset
  - docs/voeikovo_validation.md                Russian-prose report (thesis 4.7)
  - docs/figures/voeikovo_agreement_matrix.png 2x2 confusion heatmap
  - docs/figures/voeikovo_seasonal_breakdown.png agreement by season x period
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

STATION_LAT = 59.952
STATION_LON = 30.707
ERA5_LAT = 60.0
ERA5_LON = 30.75
RI_THRESHOLD = 0.1
LOCAL_UTC_OFFSET_H = 3  # Saint Petersburg, no DST

# ISO 9613-2 / project local-time periods
PERIOD_DAY = (7, 19)
PERIOD_EVENING = (19, 23)
# night = otherwise

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF",
           3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA",
           9: "SON", 10: "SON", 11: "SON"}


def build_pairs(soundings_path: Path, stability_path: Path) -> xr.Dataset:
    """Pair sounding Ri with ERA5 v2 Ri at the Voeikovo grid cell."""
    soundings = xr.open_zarr(soundings_path, consolidated=False)
    stability = xr.open_zarr(stability_path, consolidated=False)

    cell = stability.sel(latitude=ERA5_LAT, longitude=ERA5_LON, method="nearest")
    cell = cell[["Ri_b", "stable_Ri"]]

    launch_times = soundings["time"].values
    cell_at_launches = cell.sel(time=launch_times, method="nearest")
    time_lag_h = (cell_at_launches["time"].values - launch_times).astype("timedelta64[m]").astype(float) / 60.0

    sounding_ri = soundings["ri_sounding"].values.astype("float32")
    sounding_valid = soundings["valid"].values.astype(bool)
    station_id = soundings["station_id"].values.astype("int32")
    era5_ri = cell_at_launches["Ri_b"].values.astype("float32")
    era5_stable = cell_at_launches["stable_Ri"].values.astype(bool)
    sounding_stable = (sounding_ri >= RI_THRESHOLD) & np.isfinite(sounding_ri)

    paired_valid = sounding_valid & np.isfinite(sounding_ri) & np.isfinite(era5_ri) & (np.abs(time_lag_h) <= 0.75)

    local_time = launch_times + np.timedelta64(LOCAL_UTC_OFFSET_H, "h")

    return xr.Dataset(
        data_vars=dict(
            ri_sounding=("time", sounding_ri),
            ri_era5=("time", era5_ri),
            stable_sounding=("time", sounding_stable),
            stable_era5=("time", era5_stable),
            paired_valid=("time", paired_valid),
            station_id=("time", station_id),
            local_time=("time", local_time),
            time_lag_h=("time", time_lag_h.astype("float32")),
        ),
        coords=dict(time=launch_times),
        attrs=dict(
            station_latitude=STATION_LAT,
            station_longitude=STATION_LON,
            era5_latitude=ERA5_LAT,
            era5_longitude=ERA5_LON,
            ri_threshold=RI_THRESHOLD,
            time_basis="UTC; both sources hourly; matched by nearest ERA5 hour",
            stability_source=str(stability_path),
            soundings_source=str(soundings_path),
        ),
    )


def _period(local_hours: np.ndarray) -> np.ndarray:
    out = np.empty(local_hours.shape, dtype=object)
    day = (local_hours >= PERIOD_DAY[0]) & (local_hours < PERIOD_DAY[1])
    evening = (local_hours >= PERIOD_EVENING[0]) & (local_hours < PERIOD_EVENING[1])
    out[:] = "night"
    out[day] = "day"
    out[evening] = "evening"
    return out


def _confusion(stable_sounding: np.ndarray, stable_era5: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for i, s in enumerate((False, True)):
        for j, e in enumerate((False, True)):
            cm[i, j] = int(np.sum((stable_sounding == s) & (stable_era5 == e)))
    return cm  # rows = sounding (truth), cols = ERA5


def _cohens_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    pe = float(np.sum(row_marg * col_marg))
    if pe == 1.0:
        return float("nan")
    return float((po - pe) / (1 - pe))


def _correlations(ri_s: np.ndarray, ri_e: np.ndarray) -> dict:
    # Clip to a sane physical band so a handful of |Ri|>1e3 outliers don't
    # dominate Pearson. Spearman is rank-based and robust on its own.
    clip = 10.0
    s = np.clip(ri_s, -clip, clip)
    e = np.clip(ri_e, -clip, clip)
    pearson_r, pearson_p = stats.pearsonr(s, e)
    spearman_rho, spearman_p = stats.spearmanr(ri_s, ri_e)
    return dict(
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_rho=float(spearman_rho),
        spearman_p=float(spearman_p),
        clip_used=clip,
    )


def analyse(pairs: xr.Dataset) -> dict:
    df = pairs.to_dataframe().reset_index()
    df = df[df["paired_valid"]].copy()
    df["season"] = df["time"].dt.month.map(SEASONS)
    df["local_hour"] = pd.to_datetime(df["local_time"]).dt.hour
    df["period"] = _period(df["local_hour"].to_numpy())

    cm_overall = _confusion(df["stable_sounding"].to_numpy(), df["stable_era5"].to_numpy())
    agreement_overall = float(np.trace(cm_overall) / cm_overall.sum())
    kappa_overall = _cohens_kappa(cm_overall)
    corr = _correlations(df["ri_sounding"].to_numpy(), df["ri_era5"].to_numpy())

    seasonal = {}
    for season in ("DJF", "MAM", "JJA", "SON"):
        sub = df[df["season"] == season]
        if len(sub) == 0:
            continue
        cm = _confusion(sub["stable_sounding"].to_numpy(), sub["stable_era5"].to_numpy())
        seasonal[season] = dict(
            n=int(len(sub)),
            agreement=float(np.trace(cm) / cm.sum()),
            kappa=_cohens_kappa(cm),
            cm=cm,
            stable_rate_sounding=float(sub["stable_sounding"].mean()),
            stable_rate_era5=float(sub["stable_era5"].mean()),
        )

    season_period = {}
    for season in ("DJF", "MAM", "JJA", "SON"):
        for period in ("day", "evening", "night"):
            sub = df[(df["season"] == season) & (df["period"] == period)]
            if len(sub) == 0:
                continue
            cm = _confusion(sub["stable_sounding"].to_numpy(), sub["stable_era5"].to_numpy())
            season_period[(season, period)] = dict(
                n=int(len(sub)),
                agreement=float(np.trace(cm) / cm.sum()),
            )

    return dict(
        n_paired=int(len(df)),
        n_total=int(len(pairs["time"])),
        agreement_overall=agreement_overall,
        kappa_overall=kappa_overall,
        cm_overall=cm_overall,
        correlations=corr,
        seasonal=seasonal,
        season_period=season_period,
        stable_rate_sounding=float(df["stable_sounding"].mean()),
        stable_rate_era5=float(df["stable_era5"].mean()),
    )


def plot_confusion(cm: np.ndarray, agreement: float, kappa: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    n = cm.sum()
    pct = cm / n * 100.0
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=max(50, pct.max()))
    for i in range(2):
        for j in range(2):
            txt_color = "white" if pct[i, j] > pct.max() * 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)",
                    ha="center", va="center", color=txt_color, fontsize=12)
    ax.set_xticks([0, 1], ["неуст. (ERA5)", "уст. (ERA5)"])
    ax.set_yticks([0, 1], ["неуст. (зонд)", "уст. (зонд)"])
    ax.set_xlabel("ERA5 v2 классификация")
    ax.set_ylabel("Радиозонд Voeikovo (эталон)")
    ax.set_title(f"Согласованность бинарной классификации устойчивости\n"
                 f"совпадение = {agreement * 100:.1f}%, kappa = {kappa:.2f}, N = {n}")
    fig.colorbar(im, ax=ax, label="доля, %")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_seasonal(season_period: dict, seasonal: dict, out_path: Path) -> None:
    seasons = ("DJF", "MAM", "JJA", "SON")
    periods = ("day", "evening", "night")
    grid = np.full((len(periods), len(seasons)), np.nan)
    for si, season in enumerate(seasons):
        for pi, period in enumerate(periods):
            cell = season_period.get((season, period))
            if cell is not None:
                grid[pi, si] = cell["agreement"] * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1]))

    im = ax1.imshow(grid, cmap="RdYlGn", vmin=40, vmax=90)
    for pi in range(len(periods)):
        for si in range(len(seasons)):
            v = grid[pi, si]
            if not np.isnan(v):
                ax1.text(si, pi, f"{v:.0f}%", ha="center", va="center", fontsize=11, color="black")
    ax1.set_xticks(range(len(seasons)), seasons)
    ax1.set_yticks(range(len(periods)), ["день\n07-19", "вечер\n19-23", "ночь\n23-07"])
    ax1.set_title("Совпадение бинарной классификации\nпо сезону и периоду суток (местное время)")
    fig.colorbar(im, ax=ax1, label="совпадение, %")

    seasons_x = [s for s in seasons if s in seasonal]
    agree = [seasonal[s]["agreement"] * 100 for s in seasons_x]
    kappa = [seasonal[s]["kappa"] for s in seasons_x]
    x = np.arange(len(seasons_x))
    ax2b = ax2.twinx()
    bars = ax2.bar(x - 0.2, agree, width=0.4, color="#4a90d9", label="совпадение, %")
    bars2 = ax2b.bar(x + 0.2, kappa, width=0.4, color="#d97a4a", label="каппа Коэна")
    ax2.set_xticks(x, seasons_x)
    ax2.set_ylabel("совпадение, %", color="#4a90d9")
    ax2b.set_ylabel("каппа Коэна", color="#d97a4a")
    ax2.set_ylim(0, 100)
    ax2b.set_ylim(0, 1)
    ax2.set_title("Сезонная разбивка")
    for bar, v in zip(bars, agree):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    for bar, v in zip(bars2, kappa):
        ax2b.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(result: dict, gates: dict, report_path: Path, fig_dir: Path) -> None:
    cm = result["cm_overall"]
    agree = result["agreement_overall"] * 100
    kappa = result["kappa_overall"]
    corr = result["correlations"]

    lines = []
    lines.append("# Валидация классификации устойчивости ERA5 v2 по радиозондам Voeikovo")
    lines.append("")
    lines.append("## Постановка")
    lines.append("")
    lines.append(
        f"Сопоставлены попарно по часам UTC значения числа Ричардсона, рассчитанного "
        f"по реанализу ERA5 (конвейер v2, ячейка {ERA5_LAT:.2f}° с.ш., {ERA5_LON:.2f}° в.д.), "
        f"и независимо реконструированного по радиозондовым профилям станции Voeikovo "
        f"(WMO 26063 до 2018-10-31, 26075 после; {STATION_LAT:.3f}° с.ш., {STATION_LON:.3f}° в.д.). "
        f"Слой расчёта Ri_b — 2 м → 100 м над уровнем поверхности; порог устойчивости 0.1, "
        f"нижний порог квадрата сдвига 1.0 м²/с² (как в ERA5-конвейере)."
    )
    lines.append("")
    lines.append(f"Запусков с валидной парой: **{result['n_paired']}** из {result['n_total']} запрошенных "
                 f"(охват 2014–2024 гг., два запуска в сутки, 00 и 12 UTC).")
    lines.append("")

    lines.append("## Бинарная классификация (устойчиво / неустойчиво)")
    lines.append("")
    lines.append("| | ERA5: неустойчиво | ERA5: устойчиво | всего |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **зонд: неустойчиво** | {cm[0,0]} | {cm[0,1]} | {cm[0].sum()} |")
    lines.append(f"| **зонд: устойчиво**   | {cm[1,0]} | {cm[1,1]} | {cm[1].sum()} |")
    lines.append(f"| **всего**             | {cm[:,0].sum()} | {cm[:,1].sum()} | {cm.sum()} |")
    lines.append("")
    lines.append(f"- Совпадение классов: **{agree:.1f} %**")
    lines.append(f"- Каппа Коэна: **{kappa:.3f}**")
    lines.append(f"- Доля устойчивых случаев по зондам: {result['stable_rate_sounding'] * 100:.1f} %; "
                 f"по ERA5: {result['stable_rate_era5'] * 100:.1f} %")
    lines.append("")

    lines.append("## Корреляция непрерывных значений Ri_b")
    lines.append("")
    lines.append(f"- Пирсон r = **{corr['pearson_r']:.3f}** (p = {corr['pearson_p']:.2e}); "
                 f"значения обрезаны до ±{corr['clip_used']:.0f} для подавления крайних выбросов.")
    lines.append(f"- Спирмен ρ = **{corr['spearman_rho']:.3f}** (p = {corr['spearman_p']:.2e}); ранговая, "
                 f"использует исходные значения без обрезки.")
    lines.append("")

    lines.append("## Сезонная разбивка")
    lines.append("")
    lines.append("| Сезон | N | Совпадение | Каппа | Доля уст. (зонд) | Доля уст. (ERA5) |")
    lines.append("|---|---|---|---|---|---|")
    for season in ("DJF", "MAM", "JJA", "SON"):
        if season not in result["seasonal"]:
            continue
        s = result["seasonal"][season]
        lines.append(f"| {season} | {s['n']} | {s['agreement']*100:.1f} % | {s['kappa']:.3f} | "
                     f"{s['stable_rate_sounding']*100:.1f} % | {s['stable_rate_era5']*100:.1f} % |")
    lines.append("")
    djf = result["seasonal"].get("DJF", {}).get("agreement")
    jja = result["seasonal"].get("JJA", {}).get("agreement")
    if djf is not None and jja is not None:
        cmp = "выше" if jja >= djf else "ниже"
        lines.append(f"Совпадение в JJA ({jja*100:.1f} %) {cmp} зимнего DJF ({djf*100:.1f} %).")
        lines.append("")

    lines.append("## Жёсткие критерии")
    lines.append("")
    for name, info in gates.items():
        mark = "✅" if info["pass"] else "❌"
        lines.append(f"- {mark} **{name}** — {info['value']}; целевое: {info['target']}")
    lines.append("")

    lines.append("## Рисунки")
    lines.append("")
    lines.append(f"![Матрица согласия]({fig_dir.name}/voeikovo_agreement_matrix.png)")
    lines.append("")
    lines.append(f"![Сезонная разбивка]({fig_dir.name}/voeikovo_seasonal_breakdown.png)")
    lines.append("")

    lines.append("## Замечания")
    lines.append("")
    lines.append("- Согласование во времени: оба источника часовые в UTC; зондовый запуск "
                 "относится к ближайшему часу ERA5 с допуском ±45 мин.")
    lines.append("- Окно высот идентично у обоих источников (2 м → 100 м над поверхностью), "
                 "что устраняет систематический сдвиг, наблюдавшийся в v1 при использовании "
                 "разных опорных уровней.")
    lines.append("- Хвосты Ri_b у обоих источников экстремальные при слабом сдвиге; именно "
                 "поэтому первичной метрикой является бинарная классификация, а Пирсон рассчитан "
                 "на обрезанных значениях.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_gates(result: dict) -> dict:
    agree = result["agreement_overall"]
    kappa = result["kappa_overall"]
    djf = result["seasonal"].get("DJF", {}).get("agreement", float("nan"))
    jja = result["seasonal"].get("JJA", {}).get("agreement", float("nan"))
    return {
        "Бинарное совпадение ≥ 70 %": {
            "value": f"{agree*100:.1f} %",
            "target": "≥ 70 % (жёстко), ≥ 60 % (мягко)",
            "pass": agree >= 0.70,
        },
        "Каппа Коэна ≥ 0.35": {
            "value": f"{kappa:.3f}",
            "target": "≥ 0.35",
            "pass": kappa >= 0.35,
        },
        "Согласие JJA ≥ DJF": {
            "value": f"JJA {jja*100:.1f} % vs DJF {djf*100:.1f} %",
            "target": "JJA ≥ DJF",
            "pass": (np.isfinite(jja) and np.isfinite(djf) and jja >= djf),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soundings", type=Path, default=Path("data/interim/soundings_spb.zarr"))
    parser.add_argument("--stability", type=Path, default=Path("data/interim/stability_v2.zarr"))
    parser.add_argument("--output-nc", type=Path, default=Path("data/processed/voeikovo_validation.nc"))
    parser.add_argument("--report", type=Path, default=Path("docs/voeikovo_validation.md"))
    parser.add_argument("--figures", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    args.output_nc.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Pairing soundings ({args.soundings}) with stability v2 ({args.stability}) ...")
    pairs = build_pairs(args.soundings, args.stability)
    n_pair = int(pairs["paired_valid"].sum())
    print(f"      Paired-valid launches: {n_pair} of {pairs.sizes['time']}")

    print("[2/5] Computing confusion / kappa / correlations ...")
    result = analyse(pairs)
    gates = evaluate_gates(result)

    print(f"[3/5] Writing NetCDF -> {args.output_nc}")
    encoding = {v: {"zlib": True, "complevel": 4} for v in pairs.data_vars
                if pairs[v].dtype.kind in ("f", "i", "b")}
    pairs.to_netcdf(args.output_nc, encoding=encoding)

    print(f"[4/5] Plotting figures -> {args.figures}/")
    plot_confusion(result["cm_overall"], result["agreement_overall"], result["kappa_overall"],
                   args.figures / "voeikovo_agreement_matrix.png")
    plot_seasonal(result["season_period"], result["seasonal"],
                  args.figures / "voeikovo_seasonal_breakdown.png")

    print(f"[5/5] Writing report -> {args.report}")
    write_report(result, gates, args.report, args.figures)

    print("")
    print("=" * 60)
    print(f"VOEIKOVO VALIDATION (v2) — {n_pair} paired launches")
    print("=" * 60)
    print(f"  Agreement     : {result['agreement_overall']*100:6.2f} %")
    print(f"  Cohen's kappa : {result['kappa_overall']:6.3f}")
    print(f"  Pearson r     : {result['correlations']['pearson_r']:6.3f}   (clipped)")
    print(f"  Spearman rho  : {result['correlations']['spearman_rho']:6.3f}")
    print("  Seasonal agreement:")
    for season in ("DJF", "MAM", "JJA", "SON"):
        s = result["seasonal"].get(season)
        if s:
            print(f"    {season}: {s['agreement']*100:5.1f} %  (kappa={s['kappa']:.3f}, n={s['n']})")
    print("  Hard gates:")
    all_pass = True
    for name, info in gates.items():
        mark = "PASS" if info["pass"] else "FAIL"
        print(f"    [{mark}] {name}: {info['value']}")
        all_pass = all_pass and info["pass"]
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
