"""Step 2 v2 — Atmospheric stability classification.

Computes three independent classifiers for every hour and grid cell of the v2
ERA5 cache:

    (a) Bulk Richardson number ``Ri_b`` on the 2 m → 100 m AGL layer, using
        virtual potential temperature with the upper temperature interpolated
        in geopotential height between the bracketing pressure levels
        (1000 / 975 / 950 hPa).

    (b) Inverse Monin–Obukhov length ``1/L`` from ERA5 instantaneous surface
        sensible-heat and moisture fluxes plus the eastward and northward
        instantaneous turbulent stresses.

    (c) Pasquill–Turner stability class from 10 m wind speed, direct-beam
        solar radiation (``fdir``) for daytime, and total cloud cover for
        nighttime.

Each classifier emits both a continuous diagnostic and a binary stable flag;
the runner writes them into a single Zarr cache for Step 3 to consume.

────────────────────────────────────────────────────────────────────────────
Input contract — ``data/interim/era5_spb_v2.zarr``
────────────────────────────────────────────────────────────────────────────

Required variables (validated at runner start):

    Single-level dynamic:
        t2m, d2m, sp, u10, v10, u100, v100, blh,
        ssrd, fdir, tcc,
        ishf (instantaneous_surface_sensible_heat_flux),
        ie   (instantaneous_moisture_flux),
        iews, inss (eastward / northward instantaneous turbulent stress)

    Pressure-level (at 1000, 975, 950 hPa):
        t, z (geopotential), u, v

    Static:
        z_surface (surface geopotential), sdfor

────────────────────────────────────────────────────────────────────────────
Output contract — ``data/interim/stability_v2.zarr``
────────────────────────────────────────────────────────────────────────────

Dimensions: (time, latitude, longitude). Variables:

    Ri_b, inv_L, T_100, u_star                       float32
    pasquill_class                                    int8 (0=undefined; 1..7=A..G)
    stable_Ri, stable_L, stable_PG                    bool
    qc_subterranean_1000hPa,
    qc_low_ustar, qc_t100_fallback                    bool

────────────────────────────────────────────────────────────────────────────
Methodological choices (Phase B of the project plan)
────────────────────────────────────────────────────────────────────────────

- Ri_b threshold for stable classification: 0.1.
- 1/L threshold for stable classification: 0.05 m⁻¹.
- Pasquill class is stable when class ∈ {E, F, G} = {5, 6, 7}.
- ``t100`` is computed in geopotential height using a documented fallback
  hierarchy when the pressure-level grid does not bracket 100 m AGL
  (see ``richardson.py``).
- No smoothing of instantaneous fluxes (raw hourly values).
- ECMWF instantaneous-flux sign convention is upward = atmosphere-warming
  → positive ``Q_H`` ↔ unstable surface forcing; the runner verifies the
  resulting ``inv_L`` field has the expected JJA-day vs. DJF-night sign
  before writing output.

────────────────────────────────────────────────────────────────────────────
What this module does NOT do
────────────────────────────────────────────────────────────────────────────

- It does not compute the favorable-propagation flag (Step 3).
- It does not aggregate to climatology (Step 4).
- It does not produce paper figures (Step 5).
- It does not fetch ERA5 data (Step 1).

────────────────────────────────────────────────────────────────────────────
v1 archive
────────────────────────────────────────────────────────────────────────────

The v1 implementation lives under ``src.stability._v1_archive`` for reference
and so that pre-v2 callers (``src.run``, ``src.viz``, legacy tests) keep
working without modification. New code must not import from there.
"""

# v1 API re-exported for back-compat with existing callers (src/run.py,
# src/viz/__init__.py, tests/test_stability.py, tests/test_sounding_validation.py).
# When those callers are migrated to the v2 runner, this block can go away.
from src.stability._v1_archive import (  # noqa: F401
    CLASS_LABELS,
    CLASS_TO_INT,
    DEFAULT_OUTPUT_PATH,
    agreement_diagnostics,
    classify_stability,
    pasquill_class,
    plot_stability_agreement,
    richardson_bulk,
    solar_zenith_angle,
)
from src.stability._v1_archive._legacy_module import (  # noqa: F401
    _classify_richardson_values,
)
