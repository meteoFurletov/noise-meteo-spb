"""
Step 4 v2: Climatological aggregation — THE DELIVERABLE.

Collapses the per-hour favorable arrays from Step 3 v2 into the climatological
lookup table that is the scientific deliverable of this project. Every number
cited in § 4.6, the abstract, and § 5.2 of the thesis — and every number
consumed by the React calculator data layer — comes from this step.

────────────────────────────────────────────────────────────────────────────
Input contract — data/interim/favorable_v2.zarr
────────────────────────────────────────────────────────────────────────────

Variables (booleans):
    favorable                       (time, latitude, longitude, sector)
    favorable_wind                  (time, latitude, longitude, sector)
    favorable_thermal_Ri            (time, latitude, longitude)
    favorable_thermal_Ri_strict     (time, latitude, longitude)
    favorable_thermal_L_strict      (time, latitude, longitude)
    favorable_thermal_L_moderate    (time, latitude, longitude)
    favorable_thermal_PG            (time, latitude, longitude)

Coordinates:
    time       UTC, hourly, 2014-01-01..2024-12-31  (96 432 values)
    latitude   5 values   (60.5°N to 59.5°N, 0.25° spacing)
    longitude  7 values   (29.5°E to 31.0°E,  0.25° spacing)
    sector     18 values, ISO 9613-2 azimuth indices 0..17

────────────────────────────────────────────────────────────────────────────
Output contract — data/processed/p_favorable_spb.nc  (NetCDF-4, CF-1.10)
────────────────────────────────────────────────────────────────────────────

Variables:
    p_favorable                     (latitude, longitude, sector, season, period)  float32
    p_favorable_wind                (latitude, longitude, sector, season, period)  float32
    p_favorable_thermal_Ri          (latitude, longitude,         season, period)  float32
    p_favorable_thermal_Ri_strict   (latitude, longitude,         season, period)  float32
    p_favorable_thermal_L_strict    (latitude, longitude,         season, period)  float32
    p_favorable_thermal_L_moderate  (latitude, longitude,         season, period)  float32
    p_favorable_thermal_PG          (latitude, longitude,         season, period)  float32
    n_samples                       (latitude, longitude,         season, period)  int32

Coordinates:
    latitude, longitude          float32, CF units
    sector                       int8, 0..17
    sector_center_iso_deg        float32 auxiliary (0°, 20°, ..., 340°)
    season                       string  ["DJF", "MAM", "JJA", "SON"]
    period                       string  ["day", "evening", "night"]

────────────────────────────────────────────────────────────────────────────
Methodology — locked decisions (HANDOFF §4, no relitigation)
────────────────────────────────────────────────────────────────────────────

- Seasons:  DJF / MAM / JJA / SON (calendar-meteorological).
- Periods:  day 07-19, evening 19-23, night 23-07 in LOCAL TIME (UTC+3 year
            round; Russia has not observed DST since 2014). UTC→local is a
            fixed +3 h offset; no tz-database lookup is needed.
- Sectors:  18 of 20° each, stored in ISO 9613-2 "toward" convention.
            A user wanting the meteorological "where wind comes from"
            convention adds 180°.

────────────────────────────────────────────────────────────────────────────
What this module does NOT do
────────────────────────────────────────────────────────────────────────────

- Produces no figures (Phase F handles visualisation).
- Does not flip sector convention. ISO is the storage standard.
- Does not re-derive favorable flags or re-fetch ERA5.
- Does not validate against ground stations (a separate exploratory module).
"""

from src.aggregate.aggregator import aggregate_to_climatology
from src.aggregate.cnossos_combiner import (
    aggregate_cnossos_to_climatology,
    build_cnossos_favorable,
)
from src.aggregate.time_bins import (
    PERIOD_ORDER,
    SEASON_ORDER,
    assign_period,
    assign_season,
)

__all__ = [
    "PERIOD_ORDER",
    "SEASON_ORDER",
    "aggregate_cnossos_to_climatology",
    "aggregate_to_climatology",
    "assign_period",
    "assign_season",
    "build_cnossos_favorable",
]
