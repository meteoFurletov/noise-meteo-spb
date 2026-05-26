"""Thermal-favorable flag with cascade and five sensitivity variants."""

from __future__ import annotations

import numpy as np
import xarray as xr


def compute_thermal_variants(
    stability: xr.Dataset,
    *,
    L_strict_inv_m: float = 0.05,
    L_moderate_inv_m: float = 0.01,
) -> xr.Dataset:
    """Compute the five thermal-favorable variants documented in §5.2.

    Inputs (from data/interim/stability_v2.zarr):
        Ri_b            float32 (time, lat, lon)   continuous bulk Richardson
        inv_L           float32 (time, lat, lon)   continuous inverse Obukhov
        stable_Ri       bool                       Ri_b >= 0.1 with NaN mask
        stable_PG       bool                       Pasquill class in {E, F, G}
        qc_low_ustar    bool                       u* < threshold -> L masked

    Returns a Dataset with:
        favorable_thermal_Ri            primary; Ri with PG cascade on Ri NaN
        favorable_thermal_Ri_strict     Ri only, no cascade (NaN -> False)
        favorable_thermal_L_strict      inv_L >= L_strict_inv_m
        favorable_thermal_L_moderate    inv_L >= L_moderate_inv_m
        favorable_thermal_PG            stable_PG
        cascade_used                    True where Ri NaN was replaced by PG
    """
    Ri_b = stability["Ri_b"]
    inv_L = stability["inv_L"]
    stable_Ri = stability["stable_Ri"].astype(bool)
    stable_PG = stability["stable_PG"].astype(bool)
    qc_low_ustar = stability["qc_low_ustar"].astype(bool)

    Ri_nan = Ri_b.isnull()
    favorable_thermal_Ri = xr.where(Ri_nan, stable_PG, stable_Ri).astype(bool)
    favorable_thermal_Ri = favorable_thermal_Ri.rename("favorable_thermal_Ri")
    favorable_thermal_Ri.attrs.update(
        long_name="thermal favorable flag (Ri primary, PG cascade on NaN)",
        cascade_policy="if Ri_b is NaN, use stable_PG; never cascade to 1/L",
    )

    favorable_thermal_Ri_strict = xr.where(Ri_nan, False, stable_Ri).astype(bool)
    favorable_thermal_Ri_strict = favorable_thermal_Ri_strict.rename(
        "favorable_thermal_Ri_strict"
    )
    favorable_thermal_Ri_strict.attrs.update(
        long_name="thermal favorable flag (Ri only, NaN -> False)",
    )

    # L variants: recompute from continuous inv_L, masking the low-u* regime
    # (consistent with how Phase A produced stable_L) and treating NaN as False.
    inv_L_filled = inv_L.where(~qc_low_ustar)
    L_strict = (inv_L_filled >= np.float32(L_strict_inv_m)).fillna(False).astype(bool)
    L_strict = L_strict.rename("favorable_thermal_L_strict")
    L_strict.attrs.update(
        long_name="thermal favorable flag (1/L strict)",
        threshold_inv_m=float(L_strict_inv_m),
        mask="qc_low_ustar treated as not stable",
    )

    L_moderate = (
        (inv_L_filled >= np.float32(L_moderate_inv_m)).fillna(False).astype(bool)
    )
    L_moderate = L_moderate.rename("favorable_thermal_L_moderate")
    L_moderate.attrs.update(
        long_name="thermal favorable flag (1/L moderate)",
        threshold_inv_m=float(L_moderate_inv_m),
        mask="qc_low_ustar treated as not stable",
    )

    favorable_thermal_PG = stable_PG.rename("favorable_thermal_PG")
    favorable_thermal_PG.attrs.update(
        long_name="thermal favorable flag (Pasquill E + F + G)",
    )

    cascade_used = Ri_nan.rename("cascade_used").astype(bool)
    cascade_used.attrs.update(
        long_name="True where Ri_b NaN triggered fallback to stable_PG",
    )

    return xr.Dataset(
        {
            "favorable_thermal_Ri": favorable_thermal_Ri,
            "favorable_thermal_Ri_strict": favorable_thermal_Ri_strict,
            "favorable_thermal_L_strict": L_strict,
            "favorable_thermal_L_moderate": L_moderate,
            "favorable_thermal_PG": favorable_thermal_PG,
            "cascade_used": cascade_used,
        }
    )
