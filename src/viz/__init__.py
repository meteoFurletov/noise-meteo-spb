"""Step 5: publication figures for the noise-meteo-spb paper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from matplotlib import patheffects
from matplotlib import patches
from matplotlib.lines import Line2D
from pyproj import Transformer

SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
PERIOD_ORDER = ("day", "evening", "night")
PERIOD_LABELS_RU = {"day": "День", "evening": "Вечер", "night": "Ночь"}
SEASON_LABELS_RU = {"DJF": "зима", "MAM": "весна", "JJA": "лето", "SON": "осень"}
COMPASS_LABELS = ("С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ")
VALIDATION_PALETTE = {
    "DJF": "#31588f",
    "MAM": "#2d8f68",
    "JJA": "#c58a22",
    "SON": "#8a4f9f",
}
CONTOUR_COLORS = {45: "#0B3D91", 55: "#000000", 60: "#7A0019"}
KAD_105_LAT = 59.999980
KAD_105_LON = 30.476457
KAD_105_CRS = "EPSG:32636"


def plotting_config() -> dict:
    """Return matplotlib rcParams for publication-quality figures."""
    return {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def to_display_angle(propagation_angle_deg: np.ndarray | float) -> np.ndarray | float:
    """Flip propagation-direction azimuths to meteorological direction-from labels."""
    angle = (np.asarray(propagation_angle_deg, dtype=float) + 180.0) % 360.0
    return float(angle) if angle.ndim == 0 else angle


CARDINAL_LABELS_RU = {0: "С", 45: "СВ", 90: "В", 135: "ЮВ",
                      180: "Ю", 225: "ЮЗ", 270: "З", 315: "СЗ"}

# True geographic center of Saint Petersburg (Дворцовая пл. vicinity),
# 59°56'15"N 30°18'31"E in decimal degrees.
SPB_CENTER_LAT = 59 + 56 / 60 + 15 / 3600
SPB_CENTER_LON = 30 + 18 / 60 + 31 / 3600


def cardinal_label_ru(theta_deg: float) -> str:
    """Nearest cardinal label (Russian) for a degree value."""
    nearest = min(
        CARDINAL_LABELS_RU.keys(),
        key=lambda c: min(abs(c - theta_deg), 360 - abs(c - theta_deg)),
    )
    return CARDINAL_LABELS_RU[nearest]


THERMAL_PRIMARY = "p_favorable_thermal_Ri"
CENTRAL_SPB_CELL_INDEX = 24  # (59.75°N, 30.25°E) — central SPb continental cell, Phase A
THERMAL_VARIANTS_ORDER = (
    ("p_favorable_thermal_Ri", "Ri + каскад"),
    ("p_favorable_thermal_Ri_strict", "Ri (строгий)"),
    ("p_favorable_thermal_PG", "Pasquill (PG)"),
    ("p_favorable_thermal_L_moderate", "1/L (умер.)"),
    ("p_favorable_thermal_L_strict", "1/L (строгий)"),
)


def _legacy_view(ds: xr.Dataset) -> xr.Dataset:
    """Adapt the v2 (latitude, longitude) schema to the legacy `cell`/`sector_az` API
    so existing figure functions keep working without per-figure rewrites."""
    if "cell" in ds.dims:
        return ds
    sector_az_vals = ds["sector_center_iso_deg"].values.astype(float)
    stacked = ds.stack(cell=("latitude", "longitude"))
    lat_vals = stacked["latitude"].values.astype(float)
    lon_vals = stacked["longitude"].values.astype(float)
    n_cells = stacked.sizes["cell"]
    out = stacked.reset_index("cell", drop=True).assign_coords(
        cell=("cell", np.arange(n_cells)),
        cell_lat=("cell", lat_vals),
        cell_lon=("cell", lon_vals),
        sector_az=("sector", sector_az_vals),
    )
    if "p_favorable_thermal" not in out.data_vars and THERMAL_PRIMARY in out.data_vars:
        out["p_favorable_thermal"] = out[THERMAL_PRIMARY]
    return out


def fig_two_component_favorable(
    climatology: xr.Dataset,
    output_path: Path,
    cell_index: int = CENTRAL_SPB_CELL_INDEX,
) -> None:
    """Fig. 1: wind, thermal, and combined favorable propagation roses."""
    import matplotlib.pyplot as plt

    climatology = _legacy_view(climatology)
    sector_az = climatology["sector_az"].values.astype(float)
    display_az = to_display_angle(sector_az)
    sort_idx = np.argsort(display_az)
    theta = np.deg2rad(display_az[sort_idx])

    weights = climatology["n_samples"].isel(cell=cell_index)
    wind = _weighted_period_mean(
        climatology["p_favorable_wind"].isel(cell=cell_index), weights
    ).values
    combined = _weighted_period_mean(
        climatology["p_favorable"].isel(cell=cell_index), weights
    ).values
    thermal_scalar = float(
        _weighted_period_mean(
            climatology["p_favorable_thermal"].isel(cell=cell_index), weights
        )
    )
    thermal = np.full_like(wind, thermal_scalar, dtype=float)

    panels = [
        ("(а) Ветровой компонент", wind, "#2b5c7a"),
        (f"(б) Термический компонент ({thermal_scalar:.2f})", thermal, "#6d6d6d"),
        ("(в) Совокупный", combined, "#2f6b4f"),
    ]

    with plt.rc_context(plotting_config()):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(8.4, 3.35),
            subplot_kw={"projection": "polar"},
            constrained_layout=True,
        )
        for panel_idx, (ax, (title, values, color)) in enumerate(
            zip(axes, panels, strict=True)
        ):
            _setup_polar_axis(ax)
            ordered = values[sort_idx]
            theta_closed = np.r_[theta, theta[0] + 2.0 * np.pi]
            values_closed = np.r_[ordered, ordered[0]]
            ax.plot(theta_closed, values_closed, color=color, lw=1.9)
            ax.fill(theta_closed, values_closed, color=color, alpha=0.10)
            peak_idx = int(np.nanargmax(values))
            peak_az_ru = float(to_display_angle(sector_az[peak_idx]))
            ax.plot(
                np.deg2rad(peak_az_ru),
                values[peak_idx],
                marker="o",
                ms=3.4,
                color="black",
            )
            if panel_idx == 2:
                ax.annotate(
                    f"max: {peak_az_ru:.0f}° ({cardinal_label_ru(peak_az_ru)})\n"
                    f"p = {values[peak_idx]:.2f}",
                    xy=(np.deg2rad(peak_az_ru), values[peak_idx]),
                    xytext=(np.deg2rad(peak_az_ru), values[peak_idx] + 0.18),
                    ha="center",
                    fontsize=7.5,
                    arrowprops={"arrowstyle": "->", "lw": 0.6, "color": "black"},
                )
            ax.set_ylim(0.0, 0.7)
            ax.set_yticks([0.2, 0.4, 0.6])
            ax.set_yticklabels(["0.2", "0.4", "0.6"])
            ax.set_title(title, pad=9)

        lat = float(climatology["cell_lat"].isel(cell=cell_index))
        lon = float(climatology["cell_lon"].isel(cell=cell_index))
        fig.suptitle(f"Ячейка {cell_index} ({lat:.2f}° с.ш., {lon:.2f}° в.д.)", y=1.04)
        fig.supxlabel("Азимут, град (откуда дует ветер)", y=0.02)
        _save_figure(fig, output_path)


def fig_seasonal_thermal(climatology: xr.Dataset, output_path: Path) -> None:
    """Fig. 2: seasonal/diurnal thermal-favorable contrast (kept as supplementary
    full-grid view; the headline §4.6.3 figure is `fig_seasonal_diurnal_bars`)."""
    import matplotlib.pyplot as plt

    climatology = _legacy_view(climatology)
    heatmap = (
        climatology["p_favorable_thermal"]
        .sel(season=list(SEASON_ORDER), period=list(PERIOD_ORDER))
        .mean("cell")
    )
    values = heatmap.values.astype(float)

    with plt.rc_context(plotting_config()):
        fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
        im = ax.imshow(values, cmap="viridis", vmin=0.10, vmax=0.85, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.08, pad=0.14)
        cbar.set_label(
            "p_favorable_thermal, доля часов\nс термически благоприятными условиями",
            fontsize=9,
        )

        ax.set_xticks(
            np.arange(len(PERIOD_ORDER)), labels=[PERIOD_LABELS_RU[p] for p in PERIOD_ORDER]
        )
        ax.set_yticks(
            np.arange(len(SEASON_ORDER)),
            labels=[f"{s} ({SEASON_LABELS_RU[s]})" for s in SEASON_ORDER],
        )
        ax.set_xlabel("Период суток")
        ax.set_ylabel("Сезон")
        ax.set_title("Термически благоприятные условия")

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.8},
                )

        _save_figure(fig, output_path)


def fig_case_study_kad(
    climatology: xr.Dataset,
    output_path: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Fig. 3: KAD km 105 buffer-zone comparison."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    from src.case_study import run_case_study

    climatology = _legacy_view(climatology)
    run_config = dict(config or {})
    case_config = dict(run_config.get("case_study", {}))
    case_config["p_fav_dataset"] = climatology
    run_config["case_study"] = case_config
    result = run_case_study(run_config)

    metadata = result["metadata"]
    x = metadata["x_coords"]
    y = metadata["y_coords"]
    xx, yy = np.meshgrid(x, y)
    shape = metadata["grid_shape"]
    road_x, road_y = result["road_line"].xy

    modes = [
        ("static", "(а) Статический\nМГСУ без $C_{met}$"),
        ("conservative", "(б) Консервативный\nМГСУ + $C_{met}$, $C_0=0$"),
        ("climatology", "(в) Климатология\nМГСУ + секторный $C_0$"),
    ]
    levels = [45, 55, 60]
    utm_crs = ccrs.UTM(36)
    geodetic = ccrs.PlateCarree()

    with plt.rc_context(plotting_config()):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(8.2, 3.25),
            subplot_kw={"projection": utm_crs},
            constrained_layout=True,
        )
        for ax, (mode, title) in zip(axes, modes, strict=True):
            z = result["levels"][mode].reshape(shape)
            ax.set_facecolor("white")
            ax.set_extent([float(x.min()), float(x.max()), float(y.min()), float(y.max())], crs=utm_crs)
            ax.plot(
                road_x,
                road_y,
                color="black",
                lw=1.8,
                solid_capstyle="round",
                transform=utm_crs,
                zorder=3,
            )

            available_levels = [level for level in levels if np.nanmin(z) <= level <= np.nanmax(z)]
            if available_levels:
                contour = ax.contour(
                    xx,
                    yy,
                    z,
                    levels=available_levels,
                    colors=[CONTOUR_COLORS[level] for level in available_levels],
                    linewidths=1.35,
                    transform=utm_crs,
                )
                ax.clabel(contour, inline=True, fmt=lambda value: f"{value:.0f}", fontsize=7)

            ax.set_title(title)
            gl = ax.gridlines(
                crs=geodetic,
                draw_labels=True,
                linewidth=0.45,
                color="0.74",
                alpha=0.8,
                linestyle="-",
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 7}
            gl.ylabel_style = {"size": 7}
            _add_north_arrow(ax)
            _add_scale_bar(ax, 500)

        _annotate_case_asymmetry(axes[2], result)

        legend_handles = [
            Line2D([0], [0], color=CONTOUR_COLORS[level], lw=1.5, label=f"{level} дБА")
            for level in levels
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, -0.03),
        )
        _save_figure(fig, output_path)


DEFAULT_KAD_PANELS = (
    ("DJF", "night", "Зимняя ночь\nDJF, 23:00-07:00", "climatology"),
    ("MAM", "day", "Весенний день\nMAM, 07:00-19:00", "climatology"),
    ("JJA", "day", "Летний день\nJJA, 07:00-19:00", "climatology"),
    (None, None, "Статический расчёт\nМГСУ без C_met", "static"),
)
KAD_DAY_PANELS = (
    ("DJF", "day", "Зимний день\nDJF, 07:00-19:00", "climatology"),
    ("MAM", "day", "Весенний день\nMAM, 07:00-19:00", "climatology"),
    ("JJA", "day", "Летний день\nJJA, 07:00-19:00", "climatology"),
    (None, None, "Статический расчёт\nМГСУ без C_met", "static"),
)
KAD_NIGHT_PANELS = (
    ("DJF", "night", "Зимняя ночь\nDJF, 23:00-07:00", "climatology"),
    ("MAM", "night", "Весенняя ночь\nMAM, 23:00-07:00", "climatology"),
    ("JJA", "night", "Летняя ночь\nJJA, 23:00-07:00", "climatology"),
    (None, None, "Статический расчёт\nМГСУ без C_met", "static"),
)


def fig_case_study_hero_kad(
    climatology: xr.Dataset,
    output_path: Path,
    config: dict[str, Any] | None = None,
    panels: tuple = DEFAULT_KAD_PANELS,
    half_width_m: float = 400.0,
    contour_levels: tuple = (45, 55, 60),
    regulatory_level: float = 45.0,
    show_pfav_colorbar: bool = True,
    figsize: tuple = (8.6, 7.4),
    scale_bar_m: float = 200.0,
    basemap_zoom: int = 17,
) -> None:
    """Fig. 3 hero: directional climatological buffer zones at KAD km 105.
    Parameterized to also serve day-period (`fig_case_study_kad_day`) and
    night-period (`fig_case_study_kad_night`) variants for thesis §4.5."""
    import contextily as ctx
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from src.case_study.propagation import (
        compute_attenuation,
        compute_cmet,
        compute_emission_level,
        derive_c0_from_pfav,
    )

    climatology = _legacy_view(climatology)
    case_config = dict((config or {}).get("case_study", {}))
    source_params = {
        "intensity_vph": 3500,
        "hgv_fraction": 0.18,
        "speed_kmh": 90,
        "surface": "asphalt",
        **case_config.get("source_params", {}),
    }
    atmospheric_params = {
        "ground_factor": 0.5,
        "freq_hz": 1000,
        "T_celsius": 10,
        "RH_pct": 75,
        "p_kPa": 101.3,
        **case_config.get("atmospheric_params", {}),
    }

    transformer = Transformer.from_crs("EPSG:4326", KAD_105_CRS, always_xy=True)
    center_x, center_y = transformer.transform(KAD_105_LON, KAD_105_LAT)
    road_cell_idx = _nearest_climatology_cell(climatology, KAD_105_LAT, KAD_105_LON)
    sector_az = climatology["sector_az"].values.astype(float)

    grid_spacing_m = 5.0
    xx, yy, distance, azimuth = _hero_receiver_grid(
        center_x, center_y, half_width_m, grid_spacing_m
    )
    emission = compute_emission_level(**source_params)
    attenuation = compute_attenuation(distance, **atmospheric_params)
    static_levels = emission - attenuation

    cmap = plt.get_cmap("Reds")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)

    with plt.rc_context(plotting_config()):
        fig, axes_grid = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        axes = axes_grid.ravel()
        for ax, (season, period, title, mode) in zip(axes, panels, strict=True):
            ax.set_xlim(center_x - half_width_m, center_x + half_width_m)
            ax.set_ylim(center_y - half_width_m, center_y + half_width_m)
            ctx.add_basemap(
                ax,
                source=ctx.providers.CartoDB.Positron,
                crs=KAD_105_CRS,
                zoom=basemap_zoom,
                alpha=1.0,
                attribution_size=5,
            )
            if mode == "climatology":
                p_sector = (
                    climatology["p_favorable"]
                    .isel(cell=road_cell_idx)
                    .sel(season=season, period=period)
                    .values.astype(float)
                )
                sector_wedge_radii = np.asarray([
                    _distance_for_climatology_level(
                        regulatory_level,
                        emission,
                        atmospheric_params,
                        p_value,
                    )
                    for p_value in p_sector
                ])
                sector_idx = _nearest_sector_indices(azimuth, sector_az)
                c0_receivers = derive_c0_from_pfav(p_sector[sector_idx], distance)
                levels = static_levels - compute_cmet(distance, c0_db=c0_receivers)
                _draw_sector_wedges(
                    ax,
                    center_x,
                    center_y,
                    sector_az,
                    p_sector,
                    sector_wedge_radii,
                    cmap,
                    norm,
                )
            else:
                levels = static_levels

            _draw_level_contours(ax, xx, yy, levels.reshape(xx.shape), list(contour_levels))
            ax.scatter(
                [center_x],
                [center_y],
                s=24,
                marker="o",
                color="black",
                edgecolor="white",
                linewidth=0.7,
                zorder=8,
            )
            mean_r = _mean_radius_to_level(
                distance, azimuth, levels, float(regulatory_level)
            )
            r_text = (f"Ср. радиус {regulatory_level:.0f} дБА: {mean_r:.0f} м"
                      if np.isfinite(mean_r)
                      else f"Ср. радиус {regulatory_level:.0f} дБА: вне сетки")
            ax.text(
                0.03, 0.04, r_text,
                transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8.5, weight="bold",
                color="#222",
                bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.9, "pad": 2.2},
                zorder=10,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title)
            _format_latlon_ticks(ax, center_x, center_y, half_width_m, KAD_105_CRS)
            _add_north_arrow(ax)
            _add_scale_bar(ax, scale_bar_m)

        fig.suptitle(
            "Буферные зоны автомагистрали КАД (км 105) — комбинации сезон × период суток.\n"
            "Заливка — секторное p_favorable; изолинии — расчётный уровень L_Aeq, дБА; "
            f"оценка среднего радиуса даётся по контуру {regulatory_level:.0f} дБА.",
            fontsize=10,
        )

        if show_pfav_colorbar:
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.035, pad=0.02)
            cbar.set_label("p_favorable, доля часов с благоприятными условиями")

        fig.legend(
            handles=_hero_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=5,
            frameon=False,
        )
        _save_figure(fig, output_path)


def fig_validation(
    output_path: Path,
    pairs_path: Path = Path("data/processed/voeikovo_validation.nc"),
) -> None:
    """Fig. 4: bounded sounding validation composite. Reads Phase B's
    voeikovo_validation.nc by default (use raw v1 pairs.zarr only as fallback)."""
    import matplotlib.pyplot as plt

    if pairs_path.suffix == ".nc":
        pairs = xr.open_dataset(pairs_path)
        mask = pairs["paired_valid"].values.astype(bool)
        pairs = pairs.isel(time=mask)
    else:
        pairs = xr.open_zarr(pairs_path, consolidated=False)
    full_stats = _agreement_statistics_v2(pairs)
    bounded_mask = (np.abs(pairs["ri_sounding"]) <= 10.0) & (np.abs(pairs["ri_era5"]) <= 10.0)
    bounded = pairs.isel(time=bounded_mask.values)
    bounded_stats = _agreement_statistics_v2(bounded)

    valid = np.isfinite(bounded["ri_sounding"].values) & np.isfinite(bounded["ri_era5"].values)
    ri_sounding = bounded["ri_sounding"].values[valid]
    ri_era5 = bounded["ri_era5"].values[valid]

    with plt.rc_context(plotting_config()):
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), constrained_layout=True)
        _plot_validation_confusion(axes[0], full_stats)
        _plot_validation_scatter(
            axes[1],
            ri_sounding,
            ri_era5,
            bounded["time"].values[valid],
            bounded_stats,
        )
        _plot_validation_bars(axes[2], full_stats)
        _save_figure(fig, output_path)


def fig_pipeline_schematic(output_path: Path) -> None:
    """Fig. 5: vector methodology pipeline schematic."""
    import matplotlib.pyplot as plt

    steps = [
        "ERA5\n2014-2024",
        "Классификация устойчивости\nRi_bulk; проверка Pasquill",
        "Секторный флаг благоприятности\nветер ИЛИ термика; ISO 9613-2",
        "Климатологическое агрегирование\nячейка x сектор x сезон x период",
        "p_favorable_spb.nc",
        "Расчет C0\nдвухсценарная модель",
        "Буферные зоны\nМГСУ + C_met",
    ]

    with plt.rc_context(plotting_config()):
        fig, ax = plt.subplots(figsize=(4.7, 6.8), constrained_layout=True)
        ax.set_axis_off()
        y_positions = np.linspace(0.91, 0.10, len(steps))
        box_x = 0.14
        box_w = 0.72
        box_h = 0.087

        for i, (label, y) in enumerate(zip(steps, y_positions, strict=True)):
            rect = patches.Rectangle(
                (box_x, y - box_h / 2),
                box_w,
                box_h,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="black",
                linewidth=0.9,
            )
            ax.add_patch(rect)
            ax.text(0.5, y, label, transform=ax.transAxes, ha="center", va="center")
            if i < len(steps) - 1:
                next_y = y_positions[i + 1]
                ax.annotate(
                    "",
                    xy=(0.5, next_y + box_h / 2),
                    xytext=(0.5, y - box_h / 2),
                    xycoords=ax.transAxes,
                    arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "black"},
                )

        _save_figure(fig, output_path)


def fig_polar_p_favorable(
    climatology: xr.Dataset,
    cell_index: int,
    output_path: Path,
) -> None:
    """Backward-compatible wrapper for the new Fig. 1."""
    fig_two_component_favorable(climatology, output_path, cell_index)


def fig_map_p_favorable(
    climatology: xr.Dataset,
    sector: int,
    season: str,
    period: str,
    output_path: Path,
) -> None:
    """Simple spatial diagnostic retained for the original Step 5 API."""
    import matplotlib.pyplot as plt

    climatology = _legacy_view(climatology)
    values = climatology["p_favorable"].sel(season=season, period=period).isel(sector=sector)
    with plt.rc_context(plotting_config()):
        fig, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
        scatter = ax.scatter(
            climatology["cell_lon"],
            climatology["cell_lat"],
            c=values,
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=35,
        )
        fig.colorbar(scatter, ax=ax, label="p_favorable")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Sector {sector}, {season} {period}")
        _save_figure(fig, output_path)


def fig_seasonal_cycle(climatology: xr.Dataset, output_path: Path) -> None:
    """Season-by-period summary retained for the original Step 5 API."""
    fig_seasonal_thermal(climatology, output_path)


def fig_sector_season_heatmap(climatology: xr.Dataset, output_path: Path) -> None:
    """Sector x season heatmap retained for the original Step 5 API."""
    import matplotlib.pyplot as plt

    climatology = _legacy_view(climatology)
    values = (
        climatology["p_favorable"]
        .sel(season=list(SEASON_ORDER))
        .mean(("cell", "period"))
        .transpose("season", "sector")
    )
    with plt.rc_context(plotting_config()):
        fig, ax = plt.subplots(figsize=(6.2, 3.0), constrained_layout=True)
        im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, label="p_favorable")
        ax.set_yticks(np.arange(len(SEASON_ORDER)), labels=SEASON_ORDER)
        ax.set_xticks(
            np.arange(values.sizes["sector"]),
            labels=[f"{int(az):d}" for az in climatology["sector_az"].values],
            rotation=90,
        )
        ax.set_xlabel("Propagation sector azimuth")
        ax.set_ylabel("Season")
        _save_figure(fig, output_path)


def fig_case_study_kad_day(
    climatology: xr.Dataset,
    output_path: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Fig. 4.5a: daytime buffer zones at KAD km 105, sized to the 55 dBA
    regulatory contour (СН 2.2.4/2.1.8.562-96 / СанПиН 1.2.3685-21:
    L_Aeq <= 55 дБА for residential zones, day 07:00-23:00)."""
    fig_case_study_hero_kad(
        climatology, output_path, config,
        panels=KAD_DAY_PANELS,
        half_width_m=180.0,
        contour_levels=(55, 60),
        regulatory_level=55.0,
        show_pfav_colorbar=False,
        figsize=(8.6, 8.4),
        scale_bar_m=50.0,
        basemap_zoom=18,
    )


def fig_case_study_kad_night(
    climatology: xr.Dataset,
    output_path: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Fig. 4.5b: nighttime buffer zones at KAD km 105, sized to the 45 dBA
    regulatory contour (СН 2.2.4/2.1.8.562-96 / СанПиН 1.2.3685-21:
    L_Aeq <= 45 дБА for residential zones, night 23:00-07:00)."""
    fig_case_study_hero_kad(
        climatology, output_path, config,
        panels=KAD_NIGHT_PANELS,
        half_width_m=600.0,
        contour_levels=(45, 55),
        regulatory_level=45.0,
        show_pfav_colorbar=False,
        figsize=(8.6, 8.4),
        scale_bar_m=200.0,
        basemap_zoom=16,
    )


def fig_polar_seasonal_diurnal(
    climatology: xr.Dataset,
    output_path: Path,
    cell_index: int = CENTRAL_SPB_CELL_INDEX,
) -> None:
    """Fig. 4.2: 2x2 polar p_favorable for (DJF, JJA) x (night, day) at central SPb."""
    import matplotlib.pyplot as plt

    climatology = _legacy_view(climatology)
    sector_az = climatology["sector_az"].values.astype(float)
    display_az = to_display_angle(sector_az)
    sort_idx = np.argsort(display_az)
    theta = np.deg2rad(display_az[sort_idx])
    theta_closed = np.r_[theta, theta[0] + 2.0 * np.pi]

    panels = [
        ("DJF", "night", "Зима, ночь (DJF, 23-07)", "#2b3f7a"),
        ("DJF", "day", "Зима, день (DJF, 07-19)", "#5a78c9"),
        ("JJA", "night", "Лето, ночь (JJA, 23-07)", "#a93030"),
        ("JJA", "day", "Лето, день (JJA, 07-19)", "#e07b1c"),
    ]

    values_by_panel = []
    for season, period, _, _ in panels:
        v = (
            climatology["p_favorable"]
            .isel(cell=cell_index)
            .sel(season=season, period=period)
            .values.astype(float)
        )
        values_by_panel.append(v)
    radial_max = max(0.7, float(np.nanmax(values_by_panel)) * 1.05)

    with plt.rc_context(plotting_config()):
        fig, axes_grid = plt.subplots(
            2, 2, figsize=(8.4, 8.0),
            subplot_kw={"projection": "polar"},
            constrained_layout=True,
        )
        axes = axes_grid.ravel()
        for ax, (season, period, title, color), values in zip(
            axes, panels, values_by_panel, strict=True
        ):
            _setup_polar_axis(ax)
            ordered = values[sort_idx]
            values_closed = np.r_[ordered, ordered[0]]
            ax.plot(theta_closed, values_closed, color=color, lw=1.9)
            ax.fill(theta_closed, values_closed, color=color, alpha=0.15)
            peak_idx = int(np.nanargmax(values))
            peak_az_ru = float(to_display_angle(sector_az[peak_idx]))
            ax.plot(np.deg2rad(peak_az_ru), values[peak_idx],
                    marker="o", ms=4.0, color="black")
            ax.set_ylim(0.0, radial_max)
            ax.set_yticks([0.2, 0.4, 0.6])
            ax.set_yticklabels(["0.2", "0.4", "0.6"])
            ax.set_title(
                f"{title}\nmax {peak_az_ru:.0f}° "
                f"({cardinal_label_ru(peak_az_ru)}), p={values[peak_idx]:.2f}",
                pad=10, fontsize=10,
            )

        lat = float(climatology["cell_lat"].isel(cell=cell_index))
        lon = float(climatology["cell_lon"].isel(cell=cell_index))
        fig.suptitle(
            f"p_favorable — сезонно-суточная динамика, ячейка {cell_index} "
            f"({lat:.2f}° с.ш., {lon:.2f}° в.д.)",
            y=1.015,
        )
        fig.supxlabel("Азимут, град (откуда дует ветер)", y=-0.01)
        _save_figure(fig, output_path)


def fig_heatmap_p_favorable_annual(
    climatology: xr.Dataset,
    output_path: Path,
    center_cell_index: int = CENTRAL_SPB_CELL_INDEX,
) -> None:
    """Fig. 4.3: lat x lon heatmap of annual-mean p_favorable averaged over
    (sector, season, period), drawn over a cartopy PlateCarree map with
    Natural Earth coastlines (Gulf of Finland visible)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    ds = climatology if "latitude" in climatology.dims else None
    if ds is None:
        raise ValueError("fig_heatmap_p_favorable_annual requires the native "
                         "(latitude, longitude) schema.")
    grid = ds["p_favorable"].mean(("sector", "season", "period")).transpose("latitude", "longitude")
    values = grid.values.astype(float)
    lats = ds["latitude"].values.astype(float)
    lons = ds["longitude"].values.astype(float)

    legacy = _legacy_view(ds)
    center_lat = float(legacy["cell_lat"].isel(cell=center_cell_index))
    center_lon = float(legacy["cell_lon"].isel(cell=center_cell_index))

    dlat = float(np.median(np.diff(np.sort(lats))))
    dlon = float(np.median(np.diff(np.sort(lons))))
    west = float(lons.min() - dlon / 2)
    east = float(lons.max() + dlon / 2)
    south = float(lats.min() - dlat / 2)
    north = float(lats.max() + dlat / 2)

    vmin = float(np.floor(np.nanmin(values) * 100) / 100)
    vmax = float(np.ceil(np.nanmax(values) * 100) / 100)
    if vmax - vmin < 0.02:
        vmin -= 0.01
        vmax += 0.01

    proj = ccrs.PlateCarree()
    with plt.rc_context(plotting_config()):
        fig = plt.figure(figsize=(8.4, 6.0), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([west, east, south, north], crs=proj)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                       facecolor="#cfe4f2", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("10m"),
                       facecolor="#f7f2e6", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                       edgecolor="#3a3a3a", linewidth=0.7, zorder=3)
        ax.add_feature(cfeature.RIVERS.with_scale("10m"),
                       edgecolor="#6996b3", linewidth=0.5, zorder=3)

        # Order rows to match imshow's origin convention.
        if lats[0] > lats[-1]:
            values_plot = values
            origin = "upper"
        else:
            values_plot = values
            origin = "lower"
        im = ax.imshow(
            values_plot,
            cmap="viridis", vmin=vmin, vmax=vmax,
            extent=[west, east, south, north],
            origin=origin, transform=proj,
            alpha=0.62, zorder=2,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.04)
        cbar.set_label("p_favorable (среднегодовое)")

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                ax.text(
                    float(lon), float(lat), f"{values[i, j]:.2f}",
                    transform=proj,
                    ha="center", va="center", color="black",
                    fontsize=8.5, weight="bold",
                    path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")],
                    zorder=4,
                )

        ax.plot(center_lon, center_lat, marker="*", ms=16,
                color="#fff200", markeredgecolor="black", markeredgewidth=0.9,
                transform=proj, zorder=6, label="центральная ячейка анализа")
        ax.plot(SPB_CENTER_LON, SPB_CENTER_LAT, marker="o", ms=8,
                color="#d62728", markeredgecolor="white", markeredgewidth=0.8,
                transform=proj, zorder=7, label="центр Санкт-Петербурга")

        gl = ax.gridlines(crs=proj, draw_labels=True,
                          linewidth=0.4, color="0.6", alpha=0.6, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

        ax.set_title(
            "Среднегодовая вероятность благоприятного распространения\n"
            "(домен Санкт-Петербурга, усреднение по секторам/сезонам/периодам)"
        )
        ax.legend(loc="lower left", framealpha=0.92, fontsize=8)
        _save_figure(fig, output_path)


def fig_seasonal_diurnal_bars(
    climatology: xr.Dataset,
    output_path: Path,
    cell_index: int = CENTRAL_SPB_CELL_INDEX,
    variable: str = THERMAL_PRIMARY,
) -> None:
    """Fig. 4.4: grouped bar chart of p_favorable_thermal_Ri by season x period
    at the central SPb cell. Headline figure for §4.6.3."""
    import matplotlib.pyplot as plt

    climatology = _legacy_view(climatology)
    da = climatology[variable].isel(cell=cell_index)
    values = (
        da.sel(season=list(SEASON_ORDER), period=list(PERIOD_ORDER))
        .transpose("season", "period")
        .values.astype(float)
    )

    djf_night = float(da.sel(season="DJF", period="night"))
    jja_day = float(da.sel(season="JJA", period="day"))
    ratio = djf_night / jja_day if jja_day > 0 else float("nan")

    period_colors = {"day": "#e0a040", "evening": "#7f6b9c", "night": "#2b4d7a"}
    n_seasons = len(SEASON_ORDER)
    n_periods = len(PERIOD_ORDER)
    bar_w = 0.26
    x = np.arange(n_seasons)

    with plt.rc_context(plotting_config()):
        fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
        for k, period in enumerate(PERIOD_ORDER):
            offsets = (k - (n_periods - 1) / 2) * bar_w
            ax.bar(
                x + offsets,
                values[:, k],
                bar_w,
                color=period_colors[period],
                edgecolor="black",
                linewidth=0.5,
                label={"day": "День 07-19", "evening": "Вечер 19-23",
                       "night": "Ночь 23-07"}[period],
            )
            for xi, val in zip(x + offsets, values[:, k], strict=True):
                ax.text(xi, val + 0.012, f"{val:.2f}", ha="center", va="bottom",
                        fontsize=7.5)

        ax.set_xticks(x, labels=[f"{s}\n({SEASON_LABELS_RU[s]})" for s in SEASON_ORDER])
        ax.set_xlabel("Сезон")
        ax.set_ylabel(f"{variable}, доля часов")
        ax.set_ylim(0, 1.10)
        ax.set_title(
            "Сезонно-суточная динамика термически благоприятных условий\n"
            f"(центральная ячейка СПб, {variable})"
        )
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

        # Callout in upper-left empty area with a diagonal arrow pointing from the
        # JJA-day bar (the smaller one) up to the DJF-night bar (the larger one).
        text_x, text_y = 0.10, 0.98
        ax.text(
            text_x, text_y,
            f"DJF-ночь / JJA-день  ≈ {ratio:.2f}×",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color="#a02020", weight="bold",
            bbox={"facecolor": "white", "edgecolor": "#a02020", "lw": 0.8, "pad": 3.0},
        )
        _save_figure(fig, output_path)


def fig_classifier_comparison(
    climatology: xr.Dataset,
    output_path: Path,
    voeikovo_path: Path = Path("data/processed/voeikovo_validation.nc"),
) -> None:
    """Fig. 4.5: horizontal bar comparison of the 5 thermal-favorable classifier
    variants (domain means) with the Voeikovo sounding reference line."""
    import matplotlib.pyplot as plt

    ds = _legacy_view(climatology)
    weights = ds["n_samples"]

    labels = []
    means = []
    for var, label in THERMAL_VARIANTS_ORDER:
        if var not in ds.data_vars:
            continue
        da = ds[var]
        total = (da * weights).sum() / weights.sum()
        means.append(float(total))
        labels.append(label)
    means = np.asarray(means)

    voeikovo_ref = None
    voeikovo_n = None
    try:
        voe = xr.open_dataset(voeikovo_path)
        mask = voe["paired_valid"].values.astype(bool)
        stable = voe["stable_sounding"].values.astype(bool)
        valid_stable = stable[mask]
        voeikovo_n = int(mask.sum())
        if voeikovo_n > 0:
            voeikovo_ref = float(valid_stable.mean())
    except (FileNotFoundError, KeyError):
        pass

    highlight = {"Ri + каскад", "Pasquill (PG)"}
    colors = ["#a02020" if lbl in highlight else "#4a6c8a" for lbl in labels]

    with plt.rc_context(plotting_config()):
        fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
        y = np.arange(len(labels))
        ax.barh(y, means, color=colors, edgecolor="black", linewidth=0.5)
        for yi, val in zip(y, means, strict=True):
            ax.text(val + 0.008, yi, f"{val*100:.1f} %",
                    va="center", fontsize=9)
        ax.set_yticks(y, labels=labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(0.65, float(np.nanmax(means)) * 1.20))
        ax.set_xlabel("Среднедоменная доля часов с термически\n"
                      "благоприятной (устойчивой) стратификацией")
        ax.set_title("Сравнение пяти классификаторов термической благоприятности")

        if voeikovo_ref is not None:
            ax.axvline(voeikovo_ref, color="#1d6b3a", lw=1.8, ls="--",
                       label=f"Воейково (зонд, n={voeikovo_n})")
            ax.text(voeikovo_ref + 0.006, len(labels) - 0.4,
                    f"{voeikovo_ref*100:.1f} %",
                    color="#1d6b3a", fontsize=9, weight="bold")
            ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9)
        _save_figure(fig, output_path)


def generate_all_figures(
    climatology: xr.Dataset,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Produce the paper figures for Step 5 (PDFs)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_two_component_favorable(climatology, output_dir / "fig1_two_component.pdf", CENTRAL_SPB_CELL_INDEX)
    fig_polar_seasonal_diurnal(climatology, output_dir / "fig1b_polar_seasonal_diurnal.pdf", CENTRAL_SPB_CELL_INDEX)
    fig_heatmap_p_favorable_annual(climatology, output_dir / "fig2_heatmap_annual.pdf")
    fig_seasonal_diurnal_bars(climatology, output_dir / "fig2b_seasonal_diurnal_bars.pdf", CENTRAL_SPB_CELL_INDEX)
    fig_classifier_comparison(climatology, output_dir / "fig2c_classifier_comparison.pdf")
    fig_seasonal_thermal(climatology, output_dir / "fig2_seasonal_thermal.pdf")
    fig_case_study_hero_kad(climatology, output_dir / "fig3_hero_kad.pdf", config)
    fig_case_study_kad_day(climatology, output_dir / "kad_buffer_day_55dBA.pdf", config)
    fig_case_study_kad_night(climatology, output_dir / "kad_buffer_night_45dBA.pdf", config)
    fig_validation(output_dir / "fig4_validation.pdf")
    fig_pipeline_schematic(output_dir / "fig5_pipeline.pdf")


def generate_paper_png_figures(
    climatology: xr.Dataset,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Produce the Russian-labeled climatology-derived PNG paper figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_two_component_favorable(climatology, output_dir / "fig1_two_component.png", CENTRAL_SPB_CELL_INDEX)
    fig_polar_seasonal_diurnal(climatology, output_dir / "fig1b_polar_seasonal_diurnal.png", CENTRAL_SPB_CELL_INDEX)
    fig_heatmap_p_favorable_annual(climatology, output_dir / "fig2_heatmap_annual.png")
    fig_seasonal_diurnal_bars(climatology, output_dir / "fig2b_seasonal_diurnal_bars.png", CENTRAL_SPB_CELL_INDEX)
    fig_classifier_comparison(climatology, output_dir / "fig2c_classifier_comparison.png")
    fig_seasonal_thermal(climatology, output_dir / "fig2_seasonal_thermal.png")
    fig_case_study_hero_kad(climatology, output_dir / "fig3_hero_kad.png", config)
    fig_case_study_kad_day(climatology, output_dir / "kad_buffer_day_55dBA.png", config)
    fig_case_study_kad_night(climatology, output_dir / "kad_buffer_night_45dBA.png", config)
    fig_validation(output_dir / "fig4_validation.png")


def _weighted_period_mean(da: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    return (da * weights).sum(("season", "period")) / weights.sum(("season", "period"))


def _nearest_climatology_cell(ds: xr.Dataset, lat: float, lon: float) -> int:
    distance_sq = (ds["cell_lat"].values - lat) ** 2 + (ds["cell_lon"].values - lon) ** 2
    return int(np.argmin(distance_sq))


def _hero_receiver_grid(
    center_x: float,
    center_y: float,
    half_width_m: float,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(center_x - half_width_m, center_x + half_width_m + spacing_m, spacing_m)
    y = np.arange(center_y - half_width_m, center_y + half_width_m + spacing_m, spacing_m)
    xx, yy = np.meshgrid(x, y)
    dx = xx - center_x
    dy = yy - center_y
    distance = np.maximum(np.hypot(dx, dy).ravel(), 1.0)
    azimuth = (np.degrees(np.arctan2(dx, dy)).ravel() + 360.0) % 360.0
    return xx, yy, distance, azimuth


def _nearest_sector_indices(azimuths: np.ndarray, sector_azimuths: np.ndarray) -> np.ndarray:
    az = np.asarray(azimuths, dtype=float)[:, np.newaxis]
    sectors = np.asarray(sector_azimuths, dtype=float)[np.newaxis, :]
    diff = np.abs(((az - sectors + 180.0) % 360.0) - 180.0)
    return np.argmin(diff, axis=1)


def _draw_sector_wedges(
    ax: Any,
    center_x: float,
    center_y: float,
    sector_azimuths: np.ndarray,
    p_sector: np.ndarray,
    radii_m: np.ndarray,
    cmap: Any,
    norm: Any,
) -> None:
    sector_width = _sector_width_deg(sector_azimuths)
    for azimuth, p_value, radius_m in zip(sector_azimuths, p_sector, radii_m, strict=True):
        theta1 = 90.0 - (azimuth + sector_width / 2.0)
        theta2 = 90.0 - (azimuth - sector_width / 2.0)
        wedge = patches.Wedge(
            (center_x, center_y),
            float(radius_m),
            theta1,
            theta2,
            facecolor=cmap(norm(p_value)),
            edgecolor="white",
            linewidth=0.45,
            alpha=0.33,
            zorder=3,
        )
        ax.add_patch(wedge)


def _draw_level_contours(
    ax: Any,
    xx: np.ndarray,
    yy: np.ndarray,
    levels_grid: np.ndarray,
    levels: list[int],
) -> None:
    available = [
        level for level in levels if np.nanmin(levels_grid) <= level <= np.nanmax(levels_grid)
    ]
    if not available:
        return
    contour = ax.contour(
        xx,
        yy,
        levels_grid,
        levels=available,
        colors=[CONTOUR_COLORS[level] for level in available],
        linewidths=2.0,
        zorder=5,
    )
    contour.set_path_effects([patheffects.withStroke(linewidth=3.4, foreground="white")])
    for collection in getattr(contour, "collections", []):
        collection.set_path_effects([
            patheffects.withStroke(linewidth=3.4, foreground="white")
        ])
    manual_positions = [
        _contour_label_position(path, int(level))
        for level, path in zip(contour.levels, contour.get_paths(), strict=True)
    ]
    labels = ax.clabel(
        contour,
        levels=contour.levels,
        inline=True,
        manual=manual_positions,
        fmt=lambda value: f"{value:.0f} дБА",
        fontsize=8,
        colors="black",
    )
    for label in labels:
        label.set_path_effects([
            patheffects.withStroke(linewidth=3.0, foreground="white")
        ])


def _draw_reference_circle(
    ax: Any,
    center_x: float,
    center_y: float,
    radius_m: float,
    color: str,
    linewidth: float,
) -> None:
    circle = patches.Circle(
        (center_x, center_y),
        radius_m,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=(0, (5, 3)),
        zorder=4,
    )
    ax.add_patch(circle)


def _contour_label_position(path: Any, level: int) -> tuple[float, float]:
    vertices = path.vertices
    center = np.nanmean(vertices, axis=0)
    target_angles = {45: 140.0, 55: 145.0, 60: 150.0}
    theta = np.deg2rad(target_angles.get(level, 145.0))
    target = np.asarray([np.cos(theta), np.sin(theta)])
    vector = vertices - center
    norm = np.linalg.norm(vector, axis=1)
    valid = norm > 0
    if not np.any(valid):
        return tuple(center)
    unit = vector[valid] / norm[valid, np.newaxis]
    idx = int(np.argmax(unit @ target))
    return tuple(vertices[valid][idx])


def _distance_for_level(
    target_level: float,
    emission_level: float,
    atmospheric_params: dict[str, Any],
    c0_db: float,
) -> float:
    from src.case_study.propagation import compute_attenuation, compute_cmet

    distance = np.linspace(1.0, 4000.0, 4000)
    levels = emission_level - compute_attenuation(distance, **atmospheric_params)
    levels = levels - compute_cmet(distance, c0_db=c0_db)
    order = np.argsort(levels)
    return float(np.interp(target_level, levels[order], distance[order]))


def _distance_for_climatology_level(
    target_level: float,
    emission_level: float,
    atmospheric_params: dict[str, Any],
    p_favorable: float,
) -> float:
    from src.case_study.propagation import (
        compute_attenuation,
        compute_cmet,
        derive_c0_from_pfav,
    )

    distance = np.linspace(1.0, 4000.0, 4000)
    c0 = derive_c0_from_pfav(float(p_favorable), distance)
    levels = emission_level - compute_attenuation(distance, **atmospheric_params)
    levels = levels - compute_cmet(distance, c0_db=c0)
    order = np.argsort(levels)
    return float(np.interp(target_level, levels[order], distance[order]))


def _format_latlon_ticks(ax: Any, center_x: float, center_y: float,
                         half_width_m: float, crs_string: str) -> None:
    """Replace UTM-metre tick labels with WGS84 lat/lon (degrees) labels."""
    inv = Transformer.from_crs(crs_string, "EPSG:4326", always_xy=True)
    n_ticks = 5
    xs = np.linspace(center_x - half_width_m, center_x + half_width_m, n_ticks)
    ys = np.linspace(center_y - half_width_m, center_y + half_width_m, n_ticks)
    lons_at_xs = [inv.transform(x, center_y)[0] for x in xs]
    lats_at_ys = [inv.transform(center_x, y)[1] for y in ys]
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xticklabels([f"{lon:.4f}°" for lon in lons_at_xs], fontsize=7)
    ax.set_yticklabels([f"{lat:.4f}°" for lat in lats_at_ys], fontsize=7)
    ax.set_xlabel("Долгота, °в.д.", fontsize=8)
    ax.set_ylabel("Широта, °с.ш.", fontsize=8)


def _mean_radius_to_level(distance: np.ndarray, azimuth: np.ndarray,
                          levels: np.ndarray, threshold_db: float,
                          n_bins: int = 36) -> float:
    """Mean directional radius (m) of the contour at which level == threshold_db.
    For each azimuth bin, take the maximum distance where level >= threshold;
    average over bins where the contour exists."""
    levels = np.asarray(levels).ravel()
    above = levels >= threshold_db
    if not above.any():
        return float("nan")
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    az = np.asarray(azimuth).ravel()
    dist = np.asarray(distance).ravel()
    radii = []
    for k in range(n_bins):
        m = above & (az >= edges[k]) & (az < edges[k + 1])
        if m.any():
            radii.append(float(dist[m].max()))
    return float(np.mean(radii)) if radii else float("nan")


def _format_utm_ticks(ax: Any) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xticks = np.linspace(xmin, xmax, 5)
    yticks = np.linspace(ymin, ymax, 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{tick:.0f}" for tick in xticks], rotation=0)
    ax.set_yticklabels([f"{tick:.0f}" for tick in yticks])
    ax.tick_params(labelsize=7)


def _hero_legend_handles() -> list[Any]:
    return [
        Line2D([0], [0], color=CONTOUR_COLORS[45], lw=2.0, label="45 дБА"),
        Line2D([0], [0], color=CONTOUR_COLORS[55], lw=2.0, label="55 дБА"),
        Line2D([0], [0], color=CONTOUR_COLORS[60], lw=2.0, label="60 дБА"),
    ]


def _sector_width_deg(sector_az: np.ndarray) -> float:
    if len(sector_az) < 2:
        return 360.0
    wrapped = np.r_[sector_az, sector_az[0] + 360.0]
    return float(np.median(np.diff(wrapped)))


def _setup_polar_axis(ax: Any) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=COMPASS_LABELS)
    ax.set_rlabel_position(22.5)
    ax.grid(True, alpha=0.35, linewidth=0.6)


def _add_north_arrow(ax: Any) -> None:
    ax.annotate(
        "",
        xy=(0.92, 0.91),
        xytext=(0.92, 0.77),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "black"},
    )
    ax.text(0.92, 0.94, "С", transform=ax.transAxes, ha="center", va="bottom", fontsize=8)


def _add_scale_bar(ax: Any, length_m: float) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.08 * (xmax - xmin)
    y0 = ymin + 0.07 * (ymax - ymin)
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", lw=1.2)
    ax.plot([x0, x0], [y0 - 18, y0 + 18], color="black", lw=1.0)
    ax.plot([x0 + length_m, x0 + length_m], [y0 - 18, y0 + 18], color="black", lw=1.0)
    ax.text(x0 + length_m / 2, y0 + 35, f"{int(length_m)} м", ha="center", va="bottom")


def _annotate_case_asymmetry(ax: Any, result: dict[str, Any]) -> None:
    midpoint = result["road_line"].interpolate(0.5, normalized=True)
    peak = float(result["metadata"]["peak_sector_azimuth_deg"])
    theta = np.deg2rad(peak)
    long_end = (midpoint.x + np.sin(theta) * 650.0, midpoint.y + np.cos(theta) * 650.0)
    short_end = (
        midpoint.x - np.sin(theta) * 380.0,
        midpoint.y - np.cos(theta) * 380.0,
    )
    ax.annotate(
        "выше p_fav",
        xy=long_end,
        xytext=(midpoint.x + 120.0, midpoint.y + 760.0),
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "0.2"},
        ha="left",
        va="bottom",
        fontsize=8,
    )
    ax.annotate(
        "",
        xy=short_end,
        xytext=(midpoint.x, midpoint.y),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.45", "linestyle": "--"},
    )


def _format_count_ru(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def _plot_validation_confusion(ax: Any, stats: dict[str, Any]) -> None:
    confusion = stats["confusion_matrix"]
    im = ax.imshow(confusion, cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Количество парных\nнаблюдений", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    labels = ["Неустойчиво", "Устойчиво"]
    ax.set_xticks([0, 1], labels=labels, rotation=25, ha="right")
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("ERA5")
    ax.set_ylabel("Зондирование")
    ax.set_title("(а) Классификация устойчивости")
    threshold = confusion.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if confusion[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                _format_count_ru(confusion[i, j]),
                ha="center",
                va="center",
                color=color,
            )


def _plot_validation_scatter(
    ax: Any,
    ri_sounding: np.ndarray,
    ri_era5: np.ndarray,
    times: np.ndarray,
    stats: dict[str, Any],
) -> None:
    import pandas as pd

    months = pd.DatetimeIndex(times).month
    seasons = np.asarray([_season_for_month(int(month)) for month in months])
    for season in SEASON_ORDER:
        mask = seasons == season
        ax.scatter(
            ri_sounding[mask],
            ri_era5[mask],
            s=9,
            alpha=0.42,
            edgecolors="none",
            color=VALIDATION_PALETTE[season],
            label=f"{season} ({SEASON_LABELS_RU[season]})",
        )
    ax.plot([-10, 10], [-10, 10], color="0.25", lw=0.9)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ri_b зондирование")
    ax.set_ylabel("ri_b ERA5")
    ax.set_title("(б) Согласие ri_b, |ri_b| <= 10")
    ax.text(
        0.04,
        0.96,
        f"N = {_format_count_ru(stats['n'])}\n"
        f"Pearson = {stats['pearson_r']:.2f}\n"
        f"RMSE = {stats['rmse']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.75", "pad": 2.5},
        fontsize=8,
    )
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="0.75",
        loc="lower right",
        fontsize=8,
        markerscale=1.5,
        handletextpad=0.2,
        borderpad=0.3,
    )


def _plot_validation_bars(ax: Any, stats: dict[str, Any]) -> None:
    x = np.arange(len(SEASON_ORDER))
    width = 0.36
    hit_rates = [stats["seasonal"][season]["hit_rate"] for season in SEASON_ORDER]
    false_alarm_rates = [stats["seasonal"][season]["false_alarm_rate"] for season in SEASON_ORDER]
    ax.bar(x - width / 2, hit_rates, width, label="Доля попадания", color="#2d8f68")
    ax.bar(
        x + width / 2,
        false_alarm_rates,
        width,
        label="Доля ложных тревог",
        color="#9c4f3f",
    )
    ax.set_xticks(x, labels=[f"{season}\n({SEASON_LABELS_RU[season]})" for season in SEASON_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Доля")
    ax.set_title("(в) Метрики по сезонам")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=2,
        handlelength=1.4,
        columnspacing=0.9,
        fontsize=8,
    )
    for xpos, value in zip(x - width / 2, hit_rates, strict=True):
        ax.text(xpos, value + 0.025, f"{value:.0%}", ha="center", va="bottom", fontsize=7)


def _agreement_statistics_v2(ds: xr.Dataset, stable_threshold: float = 0.1) -> dict[str, Any]:
    """Build the stats dict shape `_plot_validation_*` expects from a v2 Voeikovo
    paired dataset (`ri_sounding`, `ri_era5`, `time`)."""
    import pandas as pd

    return {
        **_stats_from_arrays(ds["ri_sounding"].values, ds["ri_era5"].values, stable_threshold),
        "stable_threshold": float(stable_threshold),
        "seasonal": {
            season: _stats_from_arrays(
                ds["ri_sounding"].values[np.isin(pd.DatetimeIndex(ds["time"].values).month, months)],
                ds["ri_era5"].values[np.isin(pd.DatetimeIndex(ds["time"].values).month, months)],
                stable_threshold,
            )
            for season, months in {
                "DJF": (12, 1, 2), "MAM": (3, 4, 5),
                "JJA": (6, 7, 8), "SON": (9, 10, 11),
            }.items()
        },
    }


def _stats_from_arrays(sounding: np.ndarray, era5: np.ndarray,
                       stable_threshold: float) -> dict[str, Any]:
    sounding = np.asarray(sounding, dtype=float)
    era5 = np.asarray(era5, dtype=float)
    valid = np.isfinite(sounding) & np.isfinite(era5)
    sounding, era5 = sounding[valid], era5[valid]
    n = int(len(sounding))
    if n >= 2:
        pearson_r = float(np.corrcoef(sounding, era5)[0, 1])
    else:
        pearson_r = float("nan")
    if n:
        diff = era5 - sounding
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        bias = float(np.mean(diff))
    else:
        rmse = float("nan")
        bias = float("nan")
    s_stable = sounding > stable_threshold
    e_stable = era5 > stable_threshold
    tn = int((~s_stable & ~e_stable).sum())
    fp = int((~s_stable & e_stable).sum())
    fn = int((s_stable & ~e_stable).sum())
    tp = int((s_stable & e_stable).sum())
    hit_rate = tp / (tp + fn) if (tp + fn) else float("nan")
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "n": n,
        "pearson_r": pearson_r,
        "rmse": rmse,
        "bias": bias,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
        "hit_rate": float(hit_rate),
        "false_alarm_rate": float(false_alarm_rate),
    }


def _season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _save_figure(fig: Any, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path, pil_kwargs={"compress_level": 9, "optimize": True})
    else:
        fig.savefig(output_path)
    plt.close(fig)
