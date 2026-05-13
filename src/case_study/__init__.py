"""KAD km 105 case-study entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from src.case_study.grid import make_receiver_grid
from src.case_study.propagation import compute_levels_three_modes
from src.case_study.road import define_road_segment

DEFAULT_CASE_STUDY = {
    "center_lat": 59.999980,
    "center_lon": 30.476457,
    "azimuth_deg": 90.0,
    "length_m": 2000.0,
    "utm_zone": "EPSG:32636",
    "half_width_m": 1000.0,
    "spacing_m": 20.0,
    "source_params": {"intensity_vph": 3500, "hgv_fraction": 0.18, "speed_kmh": 90, "surface": "asphalt"},
    "atmospheric_params": {"ground_factor": 0.5, "freq_hz": 1000, "T_celsius": 10, "RH_pct": 75, "p_kPa": 101.3},
}


def run_case_study(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the KAD km 105 receiver-grid calculation."""
    config = dict(config or {})
    case_config = {**DEFAULT_CASE_STUDY, **config.get("case_study", {})}
    case_config["source_params"] = {
        **DEFAULT_CASE_STUDY["source_params"],
        **case_config.get("source_params", {}),
    }
    case_config["atmospheric_params"] = {
        **DEFAULT_CASE_STUDY["atmospheric_params"],
        **case_config.get("atmospheric_params", {}),
    }

    climatology = case_config.get("p_fav_dataset")
    if climatology is None:
        output_config = config.get("output", {})
        path = Path(output_config.get("processed_path", "data/processed/p_favorable_spb.nc"))
        climatology = xr.open_dataset(path)

    road_line = define_road_segment(
        case_config["center_lat"], case_config["center_lon"],
        case_config["azimuth_deg"], case_config["length_m"], case_config["utm_zone"],
    )
    grid_points, distances, azimuths = make_receiver_grid(
        road_line, case_config["half_width_m"], case_config["spacing_m"],
    )

    configured_cell = case_config.get("road_cell_idx")
    road_cell_idx = int(configured_cell) if configured_cell is not None else _nearest_climatology_cell(
        climatology, case_config["center_lat"], case_config["center_lon"]
    )
    level_result = compute_levels_three_modes(
        distances, azimuths, road_cell_idx, climatology,
        case_config["source_params"], case_config["atmospheric_params"],
    )
    levels = {name: level_result[name] for name in ("static", "conservative", "climatology")}

    x_coords = np.unique(grid_points[:, 0])
    y_coords = np.unique(grid_points[:, 1])
    peak_idx = int(np.argmax(level_result["p_favorable_sector"]))
    metadata = {
        "case_name": "KAD km 105",
        "utm_zone": case_config["utm_zone"],
        "road_cell_idx": road_cell_idx,
        "road_cell_lat": float(climatology["cell_lat"].isel(cell=road_cell_idx)),
        "road_cell_lon": float(climatology["cell_lon"].isel(cell=road_cell_idx)),
        "grid_shape": (len(y_coords), len(x_coords)),
        "x_coords": x_coords,
        "y_coords": y_coords,
        "source_params": case_config["source_params"],
        "atmospheric_params": case_config["atmospheric_params"],
        "sector_azimuths": level_result["sector_azimuths"],
        "p_favorable_sector": level_result["p_favorable_sector"],
        "c0_sector": level_result["c0_sector"],
        "emission_level_dba": float(level_result["emission_level_dba"]),
        "peak_sector_azimuth_deg": float(level_result["sector_azimuths"][peak_idx]),
        "conservative_note": "C0=0 gives C_met=0, the GOST conservative downwind bound.",
    }
    return {
        "grid_points": grid_points,
        "distances": distances,
        "azimuths": azimuths,
        "road_line": road_line,
        "levels": levels,
        "metadata": metadata,
    }


def _nearest_climatology_cell(ds: xr.Dataset, lat: float, lon: float) -> int:
    squared_distance = (ds["cell_lat"].values - lat) ** 2 + (ds["cell_lon"].values - lon) ** 2
    return int(np.argmin(squared_distance))


__all__ = [
    "run_case_study",
    "define_road_segment",
    "make_receiver_grid",
    "compute_levels_three_modes",
]
