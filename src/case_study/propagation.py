"""Minimal propagation model for the KAD km 105 case study."""

from __future__ import annotations

import numpy as np
import xarray as xr

MGSU_INTENSITY_VPH = np.asarray([50, 100, 230, 500, 880, 1650, 3000], dtype=float)
MGSU_SPEED_KMH = np.asarray([30, 40, 50, 60, 70], dtype=float)
MGSU_LAEQ_TABLE = np.asarray(
    [
        [63.5, 65.0, 66.5, 68.0, 69.5], [66.5, 68.0, 69.5, 71.0, 72.5],
        [69.5, 71.0, 72.5, 74.0, 75.5], [72.5, 74.0, 75.5, 77.0, 78.5],
        [75.5, 76.0, 77.5, 79.0, 80.5], [76.5, 78.0, 79.5, 81.0, 82.5],
        [78.5, 80.0, 81.5, 83.0, 84.5],
    ],
    dtype=float,
)
HGV_PERCENT = np.asarray([0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100], dtype=float)
HGV_CORRECTION_DB = np.asarray(
    [-6.5, -4.9, -3.7, -2.8, -2.1, -1.5, -0.9, -0.4, 0.0, 0.8, 1.4, 2.0, 2.5, 3.3],
    dtype=float,
)
SURFACE_CORRECTION_DB = {"asphalt": 0.0, "asphalt_concrete": 0.0, "cement": 2.0, "cement_concrete": 2.0, "concrete": 2.0, "cobblestone": 5.0}
DELTA_L_A_DB = 3.0
DELTA_L_B_DB = 6.0
DELTA_L_D0_M = 50.0
DELTA_L_MAX_DB = 15.0
DEFAULT_ROAD_LENGTH_M = 2000.0


def compute_emission_level(
    intensity_vph: float = 3500,
    hgv_fraction: float = 0.18,
    speed_kmh: float = 90,
    surface: str = "asphalt",
) -> float:
    """Return the MGSU-2015 traffic-flow noise characteristic, dBA."""
    if intensity_vph <= 0:
        raise ValueError("intensity_vph must be positive")
    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be positive")

    log_intensity = np.log10(float(intensity_vph))
    log_table = np.log10(MGSU_INTENSITY_VPH)
    by_speed = np.asarray([
        _interp_extrapolate(log_intensity, log_table, MGSU_LAEQ_TABLE[:, j])
        for j in range(MGSU_LAEQ_TABLE.shape[1])
    ])
    level = _interp_extrapolate(float(speed_kmh), MGSU_SPEED_KMH, by_speed)

    hgv_pct = float(hgv_fraction) * 100.0 if hgv_fraction <= 1.0 else float(hgv_fraction)
    level += _interp_extrapolate(hgv_pct, HGV_PERCENT, HGV_CORRECTION_DB)
    level += SURFACE_CORRECTION_DB.get(surface, 0.0)
    return float(level)


def compute_attenuation(
    distance_m: np.ndarray | float,
    ground_factor: float = 0.5,
    freq_hz: float = 1000,
    T_celsius: float = 10,
    RH_pct: float = 75,
    p_kPa: float = 101.3,
    road_length_m: float = DEFAULT_ROAD_LENGTH_M,
) -> np.ndarray:
    """Return ``A_div + A_atm + A_gr`` in dB."""
    distance = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    a_div = compute_geometric_divergence(distance, road_length_m=road_length_m)
    alpha_db_per_m = _atmospheric_absorption_db_per_m(freq_hz, T_celsius, RH_pct, p_kPa)
    a_atm = alpha_db_per_m * distance
    g = np.clip(float(ground_factor), 0.0, 1.0)
    a_gr = (1.5 + 3.0 * g) * (1.0 - np.exp(-distance / 300.0))
    return a_div + a_atm + a_gr


def compute_geometric_divergence(
    distance_m: np.ndarray | float,
    road_length_m: float = DEFAULT_ROAD_LENGTH_M,
) -> np.ndarray:
    """Return finite-line-source divergence in dB.

    The ``1 + 4d/L`` term is a compact finite-line approximation inspired by
    GOST 31295.2 / NMPB-style line-source propagation: cylindrical near the
    road and increasingly point-source-like far beyond the segment length.
    """
    distance = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    length = max(float(road_length_m), 1.0)
    return 10.0 * np.log10(2.0 * np.pi * distance * (1.0 + 4.0 * distance / length))


def compute_cmet(
    distance_m: np.ndarray | float,
    hs: float = 0.5,
    hr: float = 1.5,
    c0_db: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Return ``C_met`` from GOST 31295.2 formula (22)."""
    distance = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    c0 = np.asarray(c0_db, dtype=float)
    threshold = 10.0 * (float(hs) + float(hr))
    return np.where(distance <= threshold, 0.0, c0 * (1.0 - threshold / distance))


def delta_l_distance_dependent(
    distance_m: np.ndarray | float,
    a_db: float = DELTA_L_A_DB,
    b_db: float = DELTA_L_B_DB,
    d0_m: float = DELTA_L_D0_M,
    max_db: float = DELTA_L_MAX_DB,
) -> np.ndarray:
    """Return distance-dependent favorable-vs-homogeneous contrast.

    Approximation of NMPB-2008 tabulated behavior, tuned for typical urban
    distances in this case study.
    """
    distance = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    d0 = max(float(d0_m), 1.0)
    delta = float(a_db) + float(b_db) * np.log10(distance / d0)
    return np.clip(delta, 0.0, float(max_db))


def derive_c0_from_pfav(
    p_favorable: np.ndarray | float,
    distance_m: np.ndarray | float,
    delta_l_A_db: float = DELTA_L_A_DB,
    delta_l_B_db: float = DELTA_L_B_DB,
    delta_l_d0_m: float = DELTA_L_D0_M,
    delta_l_max_db: float = DELTA_L_MAX_DB,
) -> np.ndarray:
    """Derive ``C_0`` from a distance-dependent two-state mixture."""
    p = np.clip(np.asarray(p_favorable, dtype=float), 0.0, 1.0)
    delta_l = delta_l_distance_dependent(
        distance_m,
        a_db=delta_l_A_db,
        b_db=delta_l_B_db,
        d0_m=delta_l_d0_m,
        max_db=delta_l_max_db,
    )
    p, delta_l = np.broadcast_arrays(p, delta_l)
    homogeneous_energy = 10.0 ** (-delta_l / 10.0)
    long_term_energy = p + (1.0 - p) * homogeneous_energy
    return -10.0 * np.log10(long_term_energy)


def compute_levels_three_modes(
    grid_distances: np.ndarray,
    grid_azimuths: np.ndarray,
    road_cell_idx: int,
    p_fav_dataset: xr.Dataset,
    source_params: dict | None = None,
    atmospheric_params: dict | None = None,
) -> dict[str, np.ndarray]:
    """Compute ``L_Aeq`` receiver levels for the three case-study modes."""
    source_params = dict(source_params or {})
    atmospheric_params = dict(atmospheric_params or {})

    emission = compute_emission_level(**source_params)
    attenuation = compute_attenuation(grid_distances, **atmospheric_params)
    static = emission - attenuation

    sector_azimuths, p_sector = _sector_probability(p_fav_dataset, road_cell_idx)
    sector_idx = _nearest_sector_indices(grid_azimuths, sector_azimuths)
    p_receivers = p_sector[sector_idx]
    c0_receivers = derive_c0_from_pfav(p_receivers, grid_distances)
    c0_sector = derive_c0_from_pfav(p_sector, 500.0)

    conservative = static - compute_cmet(grid_distances, c0_db=0.0)
    climatology = static - compute_cmet(grid_distances, c0_db=c0_receivers)

    return {
        "static": static.astype(float),
        "conservative": conservative.astype(float),
        "climatology": climatology.astype(float),
        "c0_by_receiver": c0_receivers.astype(float),
        "sector_azimuths": sector_azimuths.astype(float),
        "p_favorable_sector": p_sector.astype(float),
        "c0_sector": c0_sector.astype(float),
        "emission_level_dba": np.asarray(emission, dtype=float),
    }


def _sector_probability(ds: xr.Dataset, road_cell_idx: int) -> tuple[np.ndarray, np.ndarray]:
    selected = ds["p_favorable"].isel(cell=road_cell_idx)
    if {"season", "period"}.issubset(selected.dims):
        weights = ds["n_samples"].isel(cell=road_cell_idx)
        weighted = (selected * weights).sum(("season", "period")) / weights.sum(("season", "period"))
    else:
        weighted = selected

    if "sector_az" in ds.coords:
        azimuths = ds["sector_az"].values
    else:
        azimuths = np.arange(weighted.sizes["sector"], dtype=float) * (360.0 / weighted.sizes["sector"])
    return np.asarray(azimuths, dtype=float), np.asarray(weighted.values, dtype=float)


def _nearest_sector_indices(azimuths: np.ndarray, sector_azimuths: np.ndarray) -> np.ndarray:
    az = np.asarray(azimuths, dtype=float)[:, np.newaxis]
    sectors = np.asarray(sector_azimuths, dtype=float)[np.newaxis, :]
    diff = np.abs(((az - sectors + 180.0) % 360.0) - 180.0)
    return np.argmin(diff, axis=1)


def _interp_extrapolate(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    if x <= xp[0]:
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        return float(fp[0] + slope * (x - xp[0]))
    if x >= xp[-1]:
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return float(fp[-1] + slope * (x - xp[-1]))
    return float(np.interp(x, xp, fp))


def _atmospheric_absorption_db_per_m(
    freq_hz: float,
    T_celsius: float,
    RH_pct: float,
    p_kPa: float,
) -> float:
    temperature = float(T_celsius) + 273.15
    pressure_ratio = float(p_kPa) / 101.325
    reference_temperature = 293.15
    triple_point = 273.16
    relative_humidity = np.clip(float(RH_pct) / 100.0, 0.0, 1.0)
    saturation_pressure_ratio = 10.0 ** (
        -6.8346 * (triple_point / temperature) ** 1.261 + 4.6151
    )
    molar_humidity = relative_humidity * saturation_pressure_ratio / pressure_ratio

    fr_o = pressure_ratio * (
        24.0
        + 4.04e4
        * molar_humidity
        * (0.02 + molar_humidity)
        / (0.391 + molar_humidity)
    )
    fr_n = (
        pressure_ratio
        * (temperature / reference_temperature) ** -0.5
        * (
            9.0
            + 280.0
            * molar_humidity
            * np.exp(-4.17 * ((temperature / reference_temperature) ** (-1.0 / 3.0) - 1.0))
        )
    )
    freq = float(freq_hz)
    alpha = 8.686 * freq**2 * (
        1.84e-11 * (1.0 / pressure_ratio) * np.sqrt(temperature / reference_temperature)
        + (temperature / reference_temperature) ** -2.5
        * (
            0.01275 * np.exp(-2239.1 / temperature) / (fr_o + freq**2 / fr_o)
            + 0.1068 * np.exp(-3352.0 / temperature) / (fr_n + freq**2 / fr_n)
        )
    )
    return float(alpha)
