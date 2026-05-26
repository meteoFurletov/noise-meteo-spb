"""Step 2 v2 orchestration: load v2 cache → classify → verify → write.

Reference: §1, §8, §9 of the Step 2 v2 specification.

The runner is a small script that:

  1. Loads ``data/interim/era5_spb_v2.zarr`` and validates required vars.
  2. Calls each classifier in turn (``richardson``, ``monin_obukhov``,
     ``pasquill_turner``). The Pasquill module is currently a stub
     awaiting ``fdir`` — placeholder zeros are written in its slot and
     the output zarr carries ``pipeline_state =
     'pasquill_turner_awaiting_fdir'``.
  3. Runs verification gates (``qc.py``). Soft gates print; hard gates
     abort the write on failure.
  4. Writes ``data/interim/stability_v2.zarr`` with the documented
     schema, chunking, and global attributes.

Usage::

    python -m src.stability.runner \\
        --config configs/spb_default.yaml \\
        --input  data/interim/era5_spb_v2.zarr \\
        --output data/interim/stability_v2.zarr [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from src.stability import monin_obukhov, pasquill_turner, qc, richardson

LOGGER = logging.getLogger("stability_v2")

REQUIRED_SINGLE_LEVEL = [
    "t2m", "d2m", "sp", "u10", "v10", "u100", "v100", "blh",
    "ssrd", "tcc", "ishf", "ie", "iews", "inss",
]
OPTIONAL_SINGLE_LEVEL = ["fdir"]
REQUIRED_PRESSURE_LEVEL = ["t", "z"]
REQUIRED_STATIC = ["z_surface", "sdfor"]

OUTPUT_VARIABLES = [
    "Ri_b", "inv_L", "pasquill_class",
    "stable_Ri", "stable_L", "stable_PG",
    "qc_subterranean_1000hPa", "qc_low_ustar", "qc_t100_fallback",
    "T_100", "u_star",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 2 v2 stability classifier")
    parser.add_argument("--config", type=Path, default=Path("configs/spb_default.yaml"))
    parser.add_argument("--input", type=Path, default=Path("data/interim/era5_spb_v2.zarr"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/stability_v2.zarr"))
    parser.add_argument("--dry-run", action="store_true", help="parse inputs and validate; do not compute or write")
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.output.exists() and not args.force:
        LOGGER.error("output exists: %s (use --force to overwrite)", args.output)
        return 1

    _stage("load_inputs", "opening %s", args.input)
    ds = xr.open_zarr(args.input, consolidated=False)
    _validate_inputs(ds)

    if args.dry_run:
        LOGGER.info("dry-run OK: inputs validated; shape=%s", _shape_tuple(ds))
        return 0

    _stage("compute_Ri_b", "computing bulk Richardson")
    ri_ds = richardson.compute(ds, cfg)

    _stage("compute_inv_L", "computing inverse Obukhov length")
    mo_ds = monin_obukhov.compute(ds, cfg)

    pg_state = "complete"
    _stage("compute_pasquill", "computing Pasquill-Turner")
    try:
        pg_ds = pasquill_turner.compute(ds, cfg)
    except NotImplementedError as exc:
        LOGGER.warning("pasquill_turner placeholder active: %s", exc)
        pg_ds = pasquill_turner.empty_outputs(ds)
        pg_state = pasquill_turner.PIPELINE_STATE_AWAITING_FDIR

    out = xr.merge([ri_ds, mo_ds, pg_ds])
    out = _coerce_output_schema(out)

    _stage("verify", "running verification gates")
    verification = _run_verification(out, mo_ds, pg_state)
    if not verification["passed"]:
        LOGGER.error("verification failed; output not written")
        return 1

    _stage("write_output", "writing %s", args.output)
    _write_zarr(out, args.output, cfg, pg_state, source=args.input)

    LOGGER.info("=" * 50)
    LOGGER.info("STEP 2 v2: WRITE COMPLETE (pipeline_state=%s)", pg_state)
    LOGGER.info("Ri ↔ L agreement: %.1f %%",
                100 * verification["sections"]["classifier_agreement"]["agreement_Ri_L"])
    LOGGER.info("Output: %s (%.1f MB)", args.output, _du_mb(args.output))
    LOGGER.info("=" * 50)
    return 0


def _stage(name: str, message: str, *args: Any) -> None:
    LOGGER.info("%s: " + message, name, *args)


def _shape_tuple(ds: xr.Dataset) -> tuple[int, ...]:
    return (ds.sizes.get("time", 0), ds.sizes.get("latitude", 0), ds.sizes.get("longitude", 0))


def _validate_inputs(ds: xr.Dataset) -> None:
    missing_single = [name for name in REQUIRED_SINGLE_LEVEL if name not in ds.data_vars]
    missing_pl = [name for name in REQUIRED_PRESSURE_LEVEL if name not in ds.data_vars]
    missing_static = [name for name in REQUIRED_STATIC if name not in ds.data_vars]
    if missing_single or missing_pl or missing_static:
        msg = (
            f"era5_spb_v2.zarr is missing required variables: "
            f"single={missing_single}, pressure={missing_pl}, static={missing_static}"
        )
        raise KeyError(msg)
    if "z_surface" not in ds.data_vars:
        raise KeyError(
            "Surface geopotential z_surface is required for t100 interpolation but "
            "not present in v2 cache; re-fetch the static dataset to include it "
            "before proceeding."
        )
    if "level" not in ds.coords:
        raise KeyError("v2 cache must carry a 'level' coordinate with pressure-level hPa values.")
    levels = list(int(level) for level in ds["level"].values)
    if levels != [1000, 975, 950]:
        raise ValueError(f"unexpected pressure levels: {levels}, expected [1000, 975, 950].")

    missing_optional = [name for name in OPTIONAL_SINGLE_LEVEL if name not in ds.data_vars]
    if missing_optional:
        LOGGER.warning(
            "optional inputs not present in v2 cache: %s — affected classifiers will use placeholders",
            missing_optional,
        )


def _coerce_output_schema(ds: xr.Dataset) -> xr.Dataset:
    """Ensure dtype, ordering, and chunking match the §3 schema."""
    target_dtype = {
        "Ri_b": "float32", "inv_L": "float32",
        "T_100": "float32", "u_star": "float32",
        "pasquill_class": "int8",
        "stable_Ri": "bool", "stable_L": "bool", "stable_PG": "bool",
        "qc_subterranean_1000hPa": "bool",
        "qc_low_ustar": "bool",
        "qc_t100_fallback": "bool",
    }
    for name, dtype in target_dtype.items():
        if name in ds and str(ds[name].dtype) != dtype:
            ds[name] = ds[name].astype(dtype)

    # Reorder dims to (time, latitude, longitude) for consistent chunking.
    for name in ds.data_vars:
        if set(ds[name].dims) == {"time", "latitude", "longitude"}:
            ds[name] = ds[name].transpose("time", "latitude", "longitude")
    return ds[OUTPUT_VARIABLES]


def _run_verification(out: xr.Dataset, mo_ds: xr.Dataset, pg_state: str) -> dict:
    sections: dict[str, dict] = {}
    sections["schema"] = qc.check_schema(out)
    sections["ri_b_distribution"] = qc.check_ri_b_distribution(out)
    sections["inv_L_distribution"] = qc.check_inv_L_distribution(out)
    sections["inv_L_sign_convention"] = monin_obukhov.verify_sign_convention(out["inv_L"])
    sections["pasquill_distribution"] = qc.check_pasquill_distribution(out)
    sections["classifier_agreement"] = qc.check_classifier_agreement(out)
    sections["subterranean_rate"] = qc.check_subterranean_rate(out)
    sections["spatial_summary"] = qc.spatial_summary(out)

    LOGGER.info("─" * 50)
    LOGGER.info("verification results (pipeline_state=%s)", pg_state)
    for name, result in sections.items():
        passed = result.get("passed", None)
        flag = "PASS" if passed else ("FAIL" if passed is False else "INFO")
        if "skipped_reason" in result or "PG_pairs_skipped_reason" in result:
            flag = "SKIP"
        LOGGER.info("  [%s] %s", flag, name)
        for k, v in result.items():
            if k == "passed":
                continue
            if isinstance(v, (np.ndarray,)):
                LOGGER.info("    %s = (array, shape %s)", k, v.shape)
                continue
            if isinstance(v, float):
                LOGGER.info("    %s = %.6f", k, v)
            else:
                LOGGER.info("    %s = %s", k, v)
    LOGGER.info("─" * 50)

    # Spatial summary (§9.8): print the small grid for eyeball inspection.
    grid = sections["spatial_summary"]["stable_Ri_per_cell"]
    LOGGER.info("stable_Ri per cell (mean):")
    for row in grid:
        LOGGER.info("    " + " ".join(f"{v:5.2f}" for v in row))

    gated = [
        sections["schema"]["passed"],
        sections["ri_b_distribution"]["passed"],
        sections["inv_L_distribution"]["passed"],
        sections["inv_L_sign_convention"]["passed"],
        sections["pasquill_distribution"]["passed"],
        sections["classifier_agreement"]["passed"],
        sections["subterranean_rate"]["passed"],
    ]
    return {"sections": sections, "passed": bool(all(gated))}


def _write_zarr(
    out: xr.Dataset,
    output_path: Path,
    cfg: dict,
    pg_state: str,
    source: Path,
) -> None:
    n_time = out.sizes["time"]
    chunks_time = min(8760, n_time)
    encoding: dict[str, dict] = {}
    for name in out.data_vars:
        if set(out[name].dims) == {"time", "latitude", "longitude"}:
            encoding[name] = {
                "chunks": (chunks_time, out.sizes["latitude"], out.sizes["longitude"]),
            }

    out.attrs.update(
        pipeline_version="v2",
        pipeline_state=pg_state,
        created_at=pd.Timestamp.utcnow().isoformat(),
        source_dataset=str(source),
        git_commit=_git_commit(),
        configuration_excerpt=yaml.safe_dump(cfg.get("stability", {}), sort_keys=False),
    )

    tmp = output_path.with_name(output_path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_zarr(tmp, mode="w", encoding=encoding, zarr_format=2)
    if output_path.exists():
        shutil.rmtree(output_path)
    tmp.rename(output_path)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "uncommitted"


def _du_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


if __name__ == "__main__":
    sys.exit(main())
