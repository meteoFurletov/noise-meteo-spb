"""
Step 3 v2: Per-(hour, cell, sector) favorable-propagation flag.

This module implements the ISO 9613-2 / CNOSSOS-EU favorable-propagation
criterion. It consumes the v2 ERA5 cache (for 10 m wind) and the v2
stability cache (for thermal stratification flags), and produces the
per-sector favorable-propagation flag that Step 4 aggregates into the
climatological lookup table.

────────────────────────────────────────────────────────────────────────────
Inputs
────────────────────────────────────────────────────────────────────────────

    data/interim/era5_spb_v2.zarr      — u10, v10 only
    data/interim/stability_v2.zarr     — Ri_b, inv_L, pasquill_class,
                                          stable_Ri, stable_PG, qc_low_ustar

────────────────────────────────────────────────────────────────────────────
Output
────────────────────────────────────────────────────────────────────────────

    data/interim/favorable_v2.zarr

    favorable_wind                bool (time, lat, lon, sector)
    favorable_thermal_Ri          bool (time, lat, lon)   primary, with cascade
    favorable_thermal_Ri_strict   bool (time, lat, lon)   no cascade
    favorable_thermal_L_strict    bool (time, lat, lon)   inv_L >= 0.05
    favorable_thermal_L_moderate  bool (time, lat, lon)   inv_L >= 0.01
    favorable_thermal_PG          bool (time, lat, lon)   E + F + G
    favorable                     bool (time, lat, lon, sector)
                                    = favorable_wind OR favorable_thermal_Ri
    cascade_used                  bool (time, lat, lon)   stable_Ri NaN -> PG

    Coordinate sector: int 0..17, attr `sector_centers_iso_deg` = 0..340.

────────────────────────────────────────────────────────────────────────────
Methodological choices (locked)
────────────────────────────────────────────────────────────────────────────

  * Wind threshold u_thr = 2.0 m/s (CNOSSOS-EU practice).
  * 18 azimuth sectors of 20°, ISO 9613-2 convention: azimuth = direction
    noise propagates TOWARD. Storage convention is ISO; the meteorological
    "от" (from) convention is applied in figures only (Phase F).
  * Thermal primary = bulk Richardson with cascade to Pasquill-Turner when
    Ri_b is NaN (low-shear masked, ~11.2 % of hours). Cascade does NOT fall
    through to 1/L because 1/L is ill-behaved in the calm conditions that
    cause the Ri NaN regime.
  * Four additional thermal variants stored for thesis §4.6 sensitivity
    reporting; only Ri (with cascade) feeds the `favorable` deliverable.

────────────────────────────────────────────────────────────────────────────
What this module does NOT do
────────────────────────────────────────────────────────────────────────────

  * No climatological aggregation (that is Step 4).
  * No figures (Phase F).
  * No re-computation of stability classifiers (consumed from Step 2).
  * No station validation.
  * No application of the Russian "от" convention; storage is ISO.
"""

from src.favorable.wind import (
    sector_azimuths_iso_deg,
    wind_component_along_sector,
    compute_favorable_wind,
)
from src.favorable.thermal import compute_thermal_variants

__all__ = [
    "sector_azimuths_iso_deg",
    "wind_component_along_sector",
    "compute_favorable_wind",
    "compute_thermal_variants",
]
