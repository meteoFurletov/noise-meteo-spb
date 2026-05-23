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


def fig_two_component_favorable(
    climatology: xr.Dataset,
    output_path: Path,
    cell_index: int = 16,
) -> None:
    """Fig. 1: wind, thermal, and combined favorable propagation roses."""
    import matplotlib.pyplot as plt

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
        for ax, (title, values, color) in zip(axes, panels, strict=True):
            _setup_polar_axis(ax)
            ordered = values[sort_idx]
            theta_closed = np.r_[theta, theta[0] + 2.0 * np.pi]
            values_closed = np.r_[ordered, ordered[0]]
            ax.plot(theta_closed, values_closed, color=color, lw=1.9)
            ax.fill(theta_closed, values_closed, color=color, alpha=0.10)
            peak_idx = int(np.nanargmax(values))
            ax.plot(
                np.deg2rad(to_display_angle(sector_az[peak_idx])),
                values[peak_idx],
                marker="o",
                ms=3.4,
                color="black",
            )
            ax.set_ylim(0.0, 0.7)
            ax.set_yticks([0.2, 0.4, 0.6])
            ax.set_yticklabels(["0.2", "0.4", "0.6"])
            ax.set_title(title, pad=9)

        lat = float(climatology["cell_lat"].isel(cell=cell_index))
        lon = float(climatology["cell_lon"].isel(cell=cell_index))
        fig.suptitle(f"Ячейка {cell_index} ({lat:.1f}° с.ш., {lon:.1f}° в.д.)", y=1.04)
        fig.supxlabel("Азимут, град (откуда дует ветер)", y=0.02)
        _save_figure(fig, output_path)


def fig_seasonal_thermal(climatology: xr.Dataset, output_path: Path) -> None:
    """Fig. 2: seasonal/diurnal thermal-favorable contrast."""
    import matplotlib.pyplot as plt

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


def fig_case_study_hero_kad(
    climatology: xr.Dataset,
    output_path: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Fig. 3 hero: directional climatological buffer zones at KAD km 105."""
    import contextily as ctx
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from src.case_study.propagation import (
        compute_attenuation,
        compute_cmet,
        compute_emission_level,
        derive_c0_from_pfav,
    )

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
    panels = [
        ("DJF", "night", "Зимняя ночь\nDJF, 23:00-07:00", "climatology"),
        ("MAM", "day", "Весенний день\nMAM, 07:00-19:00", "climatology"),
        ("JJA", "day", "Летний день\nJJA, 07:00-19:00", "climatology"),
        (None, None, "Статический расчёт\nМГСУ без C_met", "static"),
    ]

    half_width_m = 400.0
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
        fig, axes_grid = plt.subplots(2, 2, figsize=(8.6, 7.4), constrained_layout=True)
        axes = axes_grid.ravel()
        for ax, (season, period, title, mode) in zip(axes, panels, strict=True):
            ax.set_xlim(center_x - half_width_m, center_x + half_width_m)
            ax.set_ylim(center_y - half_width_m, center_y + half_width_m)
            ctx.add_basemap(
                ax,
                source=ctx.providers.CartoDB.Positron,
                crs=KAD_105_CRS,
                zoom=17,
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
                sector_45_radii = np.asarray([
                    _distance_for_climatology_level(
                        45.0,
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
                    sector_45_radii,
                    cmap,
                    norm,
                )
            else:
                levels = static_levels

            _draw_level_contours(ax, xx, yy, levels.reshape(xx.shape), [45, 55, 60])
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
            ax.annotate(
                "КАД 105 км",
                xy=(center_x, center_y),
                xytext=(center_x + 80, center_y - 125),
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.25"},
                fontsize=7.5,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.75, "pad": 1.4},
                zorder=9,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title)
            ax.set_xlabel("Восточное смещение UTM 36N, м")
            ax.set_ylabel("Северное смещение UTM 36N, м")
            _format_utm_ticks(ax)
            _add_north_arrow(ax)
            _add_scale_bar(ax, 200)

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
    pairs_path: Path = Path("data/interim/sounding_era5_pairs.zarr"),
) -> None:
    """Fig. 4: bounded sounding validation composite."""
    import matplotlib.pyplot as plt

    from src.stability.validation import agreement_statistics

    pairs = xr.open_zarr(pairs_path, consolidated=False)
    full_stats = agreement_statistics(pairs)
    bounded = pairs.where(
        (np.abs(pairs["ri_sounding"]) <= 10.0) & (np.abs(pairs["ri_era5"]) <= 10.0),
        drop=True,
    )
    bounded_stats = agreement_statistics(bounded)

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


def generate_all_figures(
    climatology: xr.Dataset,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Produce the five paper figures requested for Step 5."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_two_component_favorable(climatology, output_dir / "fig1_two_component.pdf", 16)
    fig_seasonal_thermal(climatology, output_dir / "fig2_seasonal_thermal.pdf")
    fig_case_study_hero_kad(climatology, output_dir / "fig3_hero_kad.pdf", config)
    fig_validation(output_dir / "fig4_validation.pdf")
    fig_pipeline_schematic(output_dir / "fig5_pipeline.pdf")


def generate_paper_png_figures(
    climatology: xr.Dataset,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Produce the Russian-labeled climatology-derived PNG paper figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_two_component_favorable(climatology, output_dir / "fig1_two_component.png", 16)
    fig_seasonal_thermal(climatology, output_dir / "fig2_seasonal_thermal.png")
    fig_case_study_hero_kad(climatology, output_dir / "fig3_hero_kad.png", config)
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
