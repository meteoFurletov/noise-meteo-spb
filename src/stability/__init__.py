"""
Step 2: Atmospheric stability classification.

Computes two independent estimates of atmospheric stability for every hour
and grid cell:

    (a) Bulk Richardson number from gradients of T and U between 2 m and
        ~110 m (1000 hPa pressure level), discretized into 7 classes A–G
        following Stull (1988).

    (b) Pasquill-Gifford stability class from external forcings (10 m wind
        speed, total cloud cover, solar zenith angle).

Both are retained. Their agreement statistics are a paper result — they
characterize how much classification ambiguity exists in SPb conditions
and which kinds of atmospheric states are most uncertain.

────────────────────────────────────────────────────────────────────────────
Output contract — data/interim/stability.zarr
────────────────────────────────────────────────────────────────────────────

Same dimensions and coords as input ERA5 dataset (without pressure_level).
Adds variables:

    stability_ri        int8, range 0..6 mapping classes A..G respectively.
                        -1 indicates missing / could not compute.
    stability_pasquill  int8, same mapping.
    ri_value            float32, the bulk Richardson number itself,
                        retained for diagnostics and sensitivity studies.
    solar_zenith        float32, degrees, retained for traceability.

────────────────────────────────────────────────────────────────────────────
Methodological choices (see article_draft.md Section 3.1, Decision 8)
────────────────────────────────────────────────────────────────────────────

Bulk Richardson is primary. It uses actual gradients and handles coastal
SPb conditions correctly. The implementation uses:

    Ri_b = (g / T̄) · (T_upper - T_lower) · Δz / (U_upper - U_lower)²

where:
    T̄ is the mean temperature between the two levels
    g = 9.81 m/s²
    Δz is computed from the geopotential height of the upper pressure level
    minus 2 m (the height of t2m / d2m)

Pasquill is the cross-check. It is well-known to over-estimate stability
in maritime conditions and may disagree systematically with Richardson on
hours when wind blows from the Gulf of Finland. That disagreement is
itself a result for the paper.

Class boundaries for Richardson follow Stull (1988) Table 5.1 and are
defined in configs/spb_default.yaml under stability.richardson.classes.

────────────────────────────────────────────────────────────────────────────
Implementation hints (for Claude Code)
────────────────────────────────────────────────────────────────────────────

For Richardson:
- ERA5 1000 hPa geopotential height needs to be computed or assumed
  (~111 m above MSL at sea-level surface pressure, approximate).
- Use np.searchsorted or xr.apply_ufunc to map ri_value to class index.

For Pasquill:
- Solar zenith angle: use metpy.calc.solar_zenith_angle, or compute
  directly from latitude, day-of-year, and hour.
- Standard P-G table from Stull (1988) Table 9.4 — implement as a
  lookup function from (wind_speed_class, insolation_class).
- Insolation level is determined by solar elevation and cloud cover.

Diagnostic figure (docs/figures/stability_agreement.pdf):
- 7×7 confusion matrix, normalized by row, with class labels A–G.
- Optional panel: agreement by season showing whether disagreement is
  seasonally biased (it likely is — winter inversions in particular).
"""

import xarray as xr


def classify_stability(ds: xr.Dataset, config: dict) -> xr.Dataset:
    """Compute Richardson and Pasquill stability classes for the dataset.

    Returns a new dataset with stability_ri, stability_pasquill, ri_value,
    and solar_zenith variables added.
    """
    raise NotImplementedError


def richardson_bulk(ds: xr.Dataset, config: dict) -> xr.DataArray:
    """Compute bulk Richardson number per hour and grid cell."""
    raise NotImplementedError


def pasquill_class(ds: xr.Dataset, config: dict) -> xr.DataArray:
    """Compute Pasquill-Gifford stability class per hour and grid cell."""
    raise NotImplementedError


def agreement_diagnostics(ds: xr.Dataset) -> dict:
    """Compute and return agreement statistics between Richardson and Pasquill.

    Used by the diagnostic figure and reported as a result in the paper.
    """
    raise NotImplementedError
