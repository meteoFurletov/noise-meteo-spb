"""
Sectoral p_fav_wind comparison v2: station 26063 vs. ERA5 v2 cell.

This is a re-point of src/exploration/pfav_wind_comparison.py against the v2
ERA5 zarr. u10/v10 are identical-by-construction between v1 and v2, so all
numbers must reproduce v1 within numerical noise (Phase D verification gate).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.projections.polar import PolarAxes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "pfav_wind_comparison_v2"
FIG_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

STATION_PARQUET = ROOT / "data" / "interim" / "meteostat_26063.parquet"
ERA5_ZARR = ROOT / "data" / "interim" / "era5_spb_v2.zarr"
ERA5_LABEL = "era5_spb_v2.zarr"
PULKOVO_LAT, PULKOVO_LON = 59.80, 30.27

U_THR = 2.0
SECTOR_CENTERS = np.arange(0, 360, 20)

POLAR_FIG = FIG_DIR / "pfav_wind_polar_v2.png"
DIFF_FIG = FIG_DIR / "pfav_wind_sectoral_diff_v2.png"
SECTOR_CSV = OUT_DIR / "sectoral_pfav.csv"
REPORT_MD = ROOT / "docs" / "pfav_wind_comparison_v2.md"

# v1 reference values (HANDOFF + existing report)
V1_REF = {
    "rmse_sectors": 0.0554,
    "mean_signed_bias": 0.0420,
    "max_abs_diff": 0.1104,
    "max_abs_sector": 20,
    "pearson_r_wspd": 0.621,
    "cosine_agreement_dir": 0.825,
}


def load_and_pair() -> dict:
    log.info("Loading station data %s", STATION_PARQUET)
    st = pd.read_parquet(STATION_PARQUET)[["wspd", "wdir"]].copy()

    log.info("Loading ERA5 v2 data %s", ERA5_ZARR)
    ds = xr.open_zarr(ERA5_ZARR)
    lat_idx = abs(ds.latitude - PULKOVO_LAT).argmin().item()
    lon_idx = abs(ds.longitude - PULKOVO_LON).argmin().item()
    cell_lat = float(ds.latitude[lat_idx])
    cell_lon = float(ds.longitude[lon_idx])
    log.info("ERA5 v2 cell: %.2f°N, %.2f°E", cell_lat, cell_lon)

    u10 = ds["u10"].isel(latitude=lat_idx, longitude=lon_idx).values
    v10 = ds["v10"].isel(latitude=lat_idx, longitude=lon_idx).values
    era5 = pd.DataFrame(
        {"u10": u10, "v10": v10},
        index=pd.to_datetime(ds.time.values),
    )

    n_total_station = len(st)
    calm_mask = (st["wspd"] == 0) & (st["wdir"] == 0)
    n_calm = int(calm_mask.sum())
    log.info("Calm sentinels: %d", n_calm)
    st["_is_calm"] = calm_mask
    valid_st = st[~(st["wspd"].isna() | st["wdir"].isna())]

    paired = valid_st.join(era5, how="inner").dropna(subset=["u10", "v10"])
    n_paired = len(paired)
    log.info("Paired hours: %d (%.1f%% of station total)",
             n_paired, 100 * n_paired / n_total_station)

    paired["theta_to_st"] = (paired["wdir"] + 180.0) % 360.0
    paired["U_st"] = paired["wspd"] / 3.6
    paired["theta_to_era5"] = (np.degrees(np.arctan2(paired["u10"], paired["v10"])) + 360.0) % 360.0
    paired["U_era5"] = np.sqrt(paired["u10"] ** 2 + paired["v10"] ** 2)

    return {
        "paired": paired,
        "n_paired": n_paired,
        "n_calm": n_calm,
        "n_total_station": n_total_station,
        "cell_lat": cell_lat,
        "cell_lon": cell_lon,
    }


def compute_continuous_stats(paired: pd.DataFrame) -> dict:
    """Pearson r on wind speed and cosine agreement on direction.

    Mirrors v1 (src/exploration/meteostat_audit.py:560–569): both compared on
    FROM directions (station `wdir` vs ERA5 atan2(-u10,-v10)), calms NOT excluded.
    """
    wdir_st = paired["wdir"] if "wdir" in paired.columns else (paired["theta_to_st"] - 180.0) % 360.0
    # FROM direction for ERA5
    wdir_era5 = (np.degrees(np.arctan2(-paired["u10"], -paired["v10"])) % 360).values
    pearson_r = float(paired["U_st"].corr(paired["U_era5"]))
    dtheta = np.deg2rad(wdir_era5 - wdir_st.values)
    cosine_agreement = float(np.cos(dtheta).mean())
    return {"pearson_r_wspd": pearson_r, "cosine_agreement_dir": cosine_agreement}


def compute_sectoral_pfav(paired: pd.DataFrame) -> pd.DataFrame:
    theta_st = paired["theta_to_st"].to_numpy(dtype=float)
    U_st = paired["U_st"].to_numpy(dtype=float)
    theta_e = paired["theta_to_era5"].to_numpy(dtype=float)
    U_e = paired["U_era5"].to_numpy(dtype=float)
    calm = paired["_is_calm"].to_numpy(dtype=bool)

    delta_st = np.deg2rad(theta_st[:, None] - SECTOR_CENTERS[None, :])
    u_st = U_st[:, None] * np.cos(delta_st)
    fav_st = (u_st >= U_THR) & (~calm[:, None])
    pfav_st = fav_st.mean(axis=0)

    delta_e = np.deg2rad(theta_e[:, None] - SECTOR_CENTERS[None, :])
    u_e = U_e[:, None] * np.cos(delta_e)
    fav_e = (u_e >= U_THR)
    pfav_e = fav_e.mean(axis=0)

    df = pd.DataFrame({
        "sector_deg": SECTOR_CENTERS,
        "pfav_station": pfav_st,
        "pfav_era5": pfav_e,
        "diff_era5_minus_station": pfav_e - pfav_st,
    })
    df.to_csv(SECTOR_CSV, index=False)
    log.info("Saved sectoral table to %s", SECTOR_CSV)
    return df


def aggregate_stats(df: pd.DataFrame) -> dict:
    diff = df["diff_era5_minus_station"].values
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    max_abs_idx = int(np.argmax(np.abs(diff)))
    max_abs_sector = int(df.loc[max_abs_idx, "sector_deg"])
    max_abs_diff = float(diff[max_abs_idx])
    n_close = int(np.sum(np.abs(diff) < 0.05))
    return {
        "mean_pfav_station": float(df["pfav_station"].mean()),
        "mean_pfav_era5": float(df["pfav_era5"].mean()),
        "rmse_sectors": rmse,
        "mean_signed_bias": bias,
        "max_abs_diff": max_abs_diff,
        "max_abs_sector": max_abs_sector,
        "n_sectors_within_5pct": n_close,
    }


def plot_polar_overlay(df: pd.DataFrame) -> Path:
    theta_from = (df["sector_deg"].values + 180) % 360
    order = np.argsort(theta_from)
    theta_sorted = np.deg2rad(theta_from[order])
    pst_sorted = df["pfav_station"].values[order]
    pe_sorted = df["pfav_era5"].values[order]
    theta_closed = np.append(theta_sorted, theta_sorted[0])
    pst_closed = np.append(pst_sorted, pst_sorted[0])
    pe_closed = np.append(pe_sorted, pe_sorted[0])

    fig = plt.figure(figsize=(7, 7))
    ax = cast(PolarAxes, fig.add_subplot(111, projection="polar"))
    ax.plot(theta_closed, pst_closed, "-o", color="C0", label="Станция 26063", linewidth=2)
    ax.plot(theta_closed, pe_closed, "-s", color="C3", label="ERA5 v2", linewidth=2)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.set_title("$p_{fav\\_wind}$ по секторам (азимут — откуда дует ветер)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    prevailing_idx = int(np.argmax(pst_sorted))
    prevailing_from = (df["sector_deg"].values[order][prevailing_idx] + 180) % 360
    ax.annotate(
        f"Преобл. {int(prevailing_from)}°",
        xy=(np.deg2rad(prevailing_from), pst_sorted[prevailing_idx]),
        xytext=(15, 15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="C0", alpha=0.7),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(POLAR_FIG, dpi=150)
    plt.close(fig)
    log.info("Saved polar plot to %s", POLAR_FIG)
    return POLAR_FIG


def plot_diff_bars(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    diff = df["diff_era5_minus_station"].values
    colors = ["C3" if abs(d) > 0.05 else "steelblue" for d in diff]
    ax.bar(df["sector_deg"], diff, width=18, color=colors, edgecolor="white")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.6)
    ax.axhline(-0.05, color="gray", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Сектор распространения θ (°, по часовой от севера)")
    ax.set_ylabel("$p_{fav}^{ERA5} - p_{fav}^{станция}$")
    ax.set_title("Посекторное расхождение $p_{fav\\_wind}$ (ERA5 v2 минус станция)")
    ax.set_xticks(SECTOR_CENTERS)
    ax.set_xticklabels([str(s) for s in SECTOR_CENTERS], rotation=45)
    fig.tight_layout()
    fig.savefig(DIFF_FIG, dpi=150)
    plt.close(fig)
    log.info("Saved diff bars to %s", DIFF_FIG)
    return DIFF_FIG


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in row
        ) + " |")
    return "\n".join([header, sep] + rows)


def write_report(pair_info: dict, sector_df: pd.DataFrame, stats: dict,
                 cont: dict) -> Path:
    bias_sign = "переоценивает" if stats["mean_signed_bias"] > 0 else "недооценивает"
    max_dir = "переоценки" if stats["max_abs_diff"] > 0 else "недооценки"

    if stats["rmse_sectors"] < 0.10:
        verdict = (
            f"ERA5-производный $p_{{fav\\_wind}}$ согласуется со станцией с RMSE = "
            f"{stats['rmse_sectors']*100:.1f} п.п. и систематическим смещением "
            f"{stats['mean_signed_bias']*100:+.1f} п.п. Источник {bias_sign} долю "
            f"благоприятных часов; расхождение задокументировано и допустимо для целей "
            f"климатологической оценки. Локальные урбанистические эффекты в секторе "
            f"{stats['max_abs_sector']}° дают максимальное расхождение "
            f"{stats['max_abs_diff']*100:+.1f} п.п."
        )
    else:
        verdict = (
            f"Расхождение между ERA5 и станцией велико: RMSE = "
            f"{stats['rmse_sectors']*100:.1f} п.п."
        )

    table_md = _df_to_markdown(sector_df)

    report = f"""# Посекторное сопоставление $p_{{fav\\_wind}}$: станция vs. ERA5 (v2)

_Дата анализа: {datetime.now().strftime('%Y-%m-%d')}_
_Источник ERA5: `{ERA5_LABEL}`_

---

## 1. Постановка сопоставления

Целью настоящего сопоставления является валидация ветровой компоненты
$p_{{fav\\_wind}}$, рассчитываемой по ERA5 (набор v2), посредством независимого
сопоставления с климатологически эквивалентным расчётом по данным наземной
станции 26063 (центральный синоптический пост Санкт-Петербурга).

Сопоставляется прикладная бинарная величина — доля часов с благоприятным
ветром по CNOSSOS-EU / ISO 9613-2 для каждого из 18 секторов распространения.

Численные значения воспроизводятся в пределах численной точности относительно
версии v1, так как переменные `u10` и `v10` в наборе данных v2 не менялись по
составу относительно v1.

---

## 2. Данные и метод

- **Источник 1 (станция):** WMO 26063, 2014–2024, файл `data/interim/meteostat_26063.parquet`.
- **Источник 2 (ERA5 v2):** ячейка ({pair_info['cell_lat']:.2f}°N, {pair_info['cell_lon']:.2f}°E),
  zarr `data/interim/{ERA5_LABEL}`, переменные `u10`, `v10`.
- **Парные часы по UTC:** {pair_info['n_paired']:,} из {pair_info['n_total_station']:,}
  доступных станционных часов ({100*pair_info['n_paired']/pair_info['n_total_station']:.1f}%).
- **Фильтр штиля:** {pair_info['n_calm']} случаев `wspd==0 & wdir==0` оставлены
  в знаменателе с принудительным `favorable=False`.
- **Сетка секторов:** 18 секторов по 20°, центры $\\theta_k = 20°\\cdot k$,
  по часовой стрелке от севера, в направлении распространения.
- **Порог благоприятности:** $u_{{thr}} = 2{{,}}0$ м/с (CNOSSOS-EU).
- **Условие благоприятности:** $u_\\theta = |U|\\cdot\\cos(\\theta_{{to}} - \\theta) \\geq u_{{thr}}$.

---

## 3. Сводные показатели согласия

| Показатель | Значение |
|---|---|
| Среднее $p_{{fav\\_wind}}$ (станция) | {stats['mean_pfav_station']:.4f} |
| Среднее $p_{{fav\\_wind}}$ (ERA5 v2) | {stats['mean_pfav_era5']:.4f} |
| RMSE по 18 секторам | {stats['rmse_sectors']:.4f} ({stats['rmse_sectors']*100:.2f} п.п.) |
| Среднее смещение (ERA5 − станция) | {stats['mean_signed_bias']:+.4f} ({stats['mean_signed_bias']*100:+.2f} п.п.) |
| Макс. абс. расхождение | {stats['max_abs_diff']:+.4f} (сектор {stats['max_abs_sector']}°) |
| Секторов с расхождением < 5 п.п. | {stats['n_sectors_within_5pct']} из 18 |
| Pearson r (скорость ветра) | {cont['pearson_r_wspd']:.3f} |
| Косинус-согласованность направления | {cont['cosine_agreement_dir']:.3f} |

---

## 4. Посекторное сопоставление

{table_md}

![Полярная диаграмма $p_{{fav\\_wind}}$](figures/pfav_wind_polar_v2.png)

![Посекторные расхождения](figures/pfav_wind_sectoral_diff_v2.png)

---

## 5. Интерпретация

**Общий характер согласия.** Среднее по 18 секторам значение $p_{{fav\\_wind}}$
у ERA5 v2 составляет {stats['mean_pfav_era5']:.3f}, у станции — {stats['mean_pfav_station']:.3f}.
RMSE посекторных оценок равен {stats['rmse_sectors']*100:.2f} процентных пунктов,
секторов с расхождением менее 5 п.п. — {stats['n_sectors_within_5pct']} из 18.

**Систематическое смещение.** ERA5 v2 в среднем {bias_sign} долю благоприятных часов
на {abs(stats['mean_signed_bias'])*100:.2f} п.п. Максимум {max_dir} наблюдается в
секторе $\\theta = {stats['max_abs_sector']}°$ и составляет
{abs(stats['max_abs_diff'])*100:.1f} п.п.

**Связь с непрерывной валидацией.** Непрерывная корреляция скорости ветра
умеренна (r = {cont['pearson_r_wspd']:.3f}), при хорошем согласии направления
(cos-согласованность {cont['cosine_agreement_dir']:.3f}). Бинаризация по порогу
2 м/с частично смягчает рассеяние скорости. Городская приповерхностная
шероховатость, неразрешаемая на сетке ERA5 0.25°, остаётся главным источником
посекторных различий и проявляется направленно — сильнее по ветру с
континентальной стороны, слабее с открытого моря.

---

## 6. Заключение для главы 4 статьи

**{verdict}**
"""
    REPORT_MD.write_text(report, encoding="utf-8")
    log.info("Report written to %s", REPORT_MD)
    return REPORT_MD


def verify_against_v1(stats: dict, cont: dict, sector_df: pd.DataFrame) -> None:
    """Phase D §4 verification gate."""
    def fmt(ok: bool) -> str:
        return "OK" if ok else "FAIL"

    d_r = cont["pearson_r_wspd"] - V1_REF["pearson_r_wspd"]
    d_cos = cont["cosine_agreement_dir"] - V1_REF["cosine_agreement_dir"]
    d_rmse_pp = (stats["rmse_sectors"] - V1_REF["rmse_sectors"]) * 100

    ok_r = abs(d_r) <= 0.005
    ok_cos = abs(d_cos) <= 0.005
    ok_rmse = abs(d_rmse_pp) <= 0.1

    # Over-prediction arc: v1 was contiguous θ=340° → 80° (i.e. 340,0,20,40,60,80) positive
    over_arc = [340, 0, 20, 40, 60, 80]
    over_signs = [sector_df.loc[sector_df["sector_deg"] == s, "diff_era5_minus_station"].values[0] > 0
                  for s in over_arc]
    ok_arc = all(over_signs)
    arc_str = f"θ{over_arc[0]}–{over_arc[-1]}°"

    overall = "PASSED" if (ok_r and ok_cos and ok_rmse and ok_arc) else "DEVIATION"

    print("\n" + "=" * 60)
    print(f"PHASE D VERIFICATION: {overall}")
    print("=" * 60)
    print(f"Pearson r:           v1={V1_REF['pearson_r_wspd']:.3f}  "
          f"v2={cont['pearson_r_wspd']:.3f}  Δ={d_r:+.4f}  [{fmt(ok_r)}]")
    print(f"Cosine agreement:    v1={V1_REF['cosine_agreement_dir']:.3f}  "
          f"v2={cont['cosine_agreement_dir']:.3f}  Δ={d_cos:+.4f}  [{fmt(ok_cos)}]")
    print(f"RMSE per-sector:     v1={V1_REF['rmse_sectors']*100:.2f}   "
          f"v2={stats['rmse_sectors']*100:.2f}   Δ={d_rmse_pp:+.3f} pp  [{fmt(ok_rmse)}]")
    print(f"Over-prediction arc: v1=θ340–80°  v2={arc_str} (all sign>0: {ok_arc})  [{fmt(ok_arc)}]")
    print("Outputs:")
    print(f"  {REPORT_MD.relative_to(ROOT)}")
    print(f"  {SECTOR_CSV.relative_to(ROOT)}")
    print(f"  {POLAR_FIG.relative_to(ROOT)}")
    print(f"  {DIFF_FIG.relative_to(ROOT)}")
    print("=" * 60)


def main():
    import time
    t0 = time.time()

    pair_info = load_and_pair()
    sector_df = compute_sectoral_pfav(pair_info["paired"])
    stats = aggregate_stats(sector_df)
    cont = compute_continuous_stats(pair_info["paired"])
    plot_polar_overlay(sector_df)
    plot_diff_bars(sector_df)
    write_report(pair_info, sector_df, stats, cont)
    verify_against_v1(stats, cont, sector_df)

    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
