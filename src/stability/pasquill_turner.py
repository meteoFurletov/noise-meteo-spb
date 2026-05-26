"""Pasquill–Turner stability classification.

Reference: §6 of the Step 2 v2 specification.

Inputs (single-level fields on the v2 grid):
    fdir : J/m² accumulated over the previous hour; ``fdir/3600`` gives
           the mean direct-beam shortwave irradiance in W/m². Used to
           distinguish day vs. night and to bin daytime insolation
           (slight / moderate / strong).
    tcc  : total cloud cover, fraction in [0, 1]; used to bin nighttime
           (cloudy / clear).
    u10, v10 : surface wind components, m/s; magnitude binned per Turner.

Outputs:
    pasquill_class : int8 ∈ {0..7}, with 1..7 mapping to A..G and 0
                     marking undefined (should not happen in practice
                     and is gated by §9.5).
    stable_PG      : bool, True for classes E/F/G.

Classification table (§6.3):

  Daytime (insolation × wind speed):
                 strong  moderate  slight
    < 2 m/s        A        A         B
    2–3 m/s        A        B         C
    3–5 m/s        B        B         C
    5–6 m/s        C        C         D
    ≥ 6 m/s        C        D         D

  Nighttime (cloud × wind speed):
                 cloudy   clear
    < 2 m/s        G        G
    2–3 m/s        E        F
    3–5 m/s        D        E
    5–6 m/s        D        D
    ≥ 6 m/s        D        D

Encoding: A=1, B=2, C=3, D=4, E=5, F=6, G=7.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

PIPELINE_STATE_AWAITING_FDIR = "pasquill_turner_awaiting_fdir"

# Class encoding: index 0 reserved for undefined.
_A, _B, _C, _D, _E, _F, _G = 1, 2, 3, 4, 5, 6, 7

# (insolation index, wind bin) → class
# insolation index: 0=strong, 1=moderate, 2=slight
# wind bin: 0=<2, 1=2-3, 2=3-5, 3=5-6, 4=>=6
_DAY_TABLE = np.array(
    [
        [_A, _A, _B, _C, _C],   # strong
        [_A, _B, _B, _C, _D],   # moderate
        [_B, _C, _C, _D, _D],   # slight
    ],
    dtype=np.int8,
)

# Nighttime: rows = cloud index (0=cloudy, 1=clear), cols = wind bin
_NIGHT_TABLE = np.array(
    [
        [_G, _E, _D, _D, _D],   # cloudy
        [_G, _F, _E, _D, _D],   # clear
    ],
    dtype=np.int8,
)


def compute(ds: xr.Dataset, cfg: dict) -> xr.Dataset:
    """Compute Pasquill–Turner class and the binary stable flag."""
    pg_cfg = cfg["stability"]["pasquill_turner"]
    moderate_threshold = float(pg_cfg["insolation_W_per_m2"]["moderate"])
    strong_threshold = float(pg_cfg["insolation_W_per_m2"]["strong"])
    night_cloud_threshold = float(pg_cfg["nighttime_cloud_threshold"])
    daytime_fdir_threshold = float(pg_cfg["daytime_fdir_threshold_W_per_m2"])
    wind_bin_edges = list(pg_cfg["wind_bins_m_per_s"])

    if "fdir" not in ds.data_vars:
        raise KeyError(
            "Pasquill-Turner classification requires 'fdir' in the v2 cache. "
            "Run scripts/fetch_fdir_per_year.py and scripts/merge_fdir_into_v2_zarr.py."
        )

    # Mean W/m² over the previous hour from accumulated J/m².
    fdir_W = ds["fdir"] / 3600.0
    tcc = ds["tcc"].clip(min=0.0, max=1.0)
    wind_speed = np.hypot(ds["u10"], ds["v10"])

    # Wind-speed bin index ∈ {0..4} for the five Turner classes
    # [<2, 2–3, 3–5, 5–6, ≥6] m/s.
    wind_bin = xr.apply_ufunc(
        np.digitize,
        wind_speed,
        kwargs={"bins": wind_bin_edges, "right": False},
        dask="parallelized",
        output_dtypes=[np.int8],
    ).astype("int8")

    # Daytime test: meaningful direct-beam shortwave irradiance.
    is_day = fdir_W > daytime_fdir_threshold

    # Daytime insolation index: 0=strong, 1=moderate, 2=slight.
    insolation = xr.where(
        fdir_W >= strong_threshold, 0,
        xr.where(fdir_W >= moderate_threshold, 1, 2),
    ).astype("int8")

    # Nighttime cloud index: 0=cloudy, 1=clear.
    cloud_index = xr.where(tcc >= night_cloud_threshold, 0, 1).astype("int8")

    # Lookup using fancy indexing. Stack 2D table indices into a 3D output.
    day_lookup = xr.apply_ufunc(
        lambda ins, wb: _DAY_TABLE[ins, wb],
        insolation,
        wind_bin,
        dask="parallelized",
        output_dtypes=[np.int8],
    )
    night_lookup = xr.apply_ufunc(
        lambda ci, wb: _NIGHT_TABLE[ci, wb],
        cloud_index,
        wind_bin,
        dask="parallelized",
        output_dtypes=[np.int8],
    )

    pg_class = xr.where(is_day, day_lookup, night_lookup).astype("int8")
    pg_class = pg_class.where(
        ds["fdir"].notnull() & ds["tcc"].notnull() & wind_speed.notnull(),
        0,
    ).astype("int8")
    pg_class.name = "pasquill_class"
    pg_class.attrs.update(
        long_name="Pasquill-Turner stability class",
        class_mapping="1=A, 2=B, 3=C, 4=D, 5=E, 6=F, 7=G; 0=undefined",
        method="Turner (1964) wind × insolation/cloud lookup; transitional resolved to more-unstable",
        reference="§6 of Step 2 v2 specification",
    )

    stable_PG = (pg_class >= _E).astype(bool)
    stable_PG.name = "stable_PG"
    stable_PG.attrs.update(
        long_name="pasquill_class ∈ {E, F, G}",
        method="Pasquill class index ≥ 5",
    )

    return xr.Dataset({"pasquill_class": pg_class, "stable_PG": stable_PG})


def empty_outputs(ds: xr.Dataset) -> xr.Dataset:
    """Placeholder zero/False Pasquill fields, used only if ``fdir`` is absent.

    Retained from the pre-fdir phase so the runner stays robust if someone
    re-runs against an older v2 cache. Now that ``fdir`` is in the cache,
    the real :func:`compute` path is always taken.
    """
    template = ds["t2m"]
    pasquill_class = xr.zeros_like(template, dtype="int8")
    pasquill_class.name = "pasquill_class"
    pasquill_class.attrs.update(
        long_name="Pasquill-Turner stability class",
        method="placeholder (awaiting fdir); 0 = undefined",
        class_mapping="1=A, 2=B, 3=C, 4=D, 5=E, 6=F, 7=G; 0=undefined",
        pipeline_state=PIPELINE_STATE_AWAITING_FDIR,
    )
    stable_PG = xr.zeros_like(template, dtype=bool)
    stable_PG.name = "stable_PG"
    stable_PG.attrs.update(
        long_name="pasquill_class ∈ {E,F,G}",
        method="placeholder (awaiting fdir)",
        pipeline_state=PIPELINE_STATE_AWAITING_FDIR,
    )
    return xr.Dataset({"pasquill_class": pasquill_class, "stable_PG": stable_PG})
