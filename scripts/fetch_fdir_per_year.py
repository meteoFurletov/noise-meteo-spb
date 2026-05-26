"""Fetch ERA5 fdir (total-sky direct solar radiation at surface) per year.

One CDS request per year, single variable, full hourly, full v2 bbox.
Idempotent: skips years whose output file already exists and is non-empty.

Run: python scripts/fetch_fdir_per_year.py
Output: data/raw/era5_v2/era5_spb_v2_fdir_<year>.nc
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cdsapi
import yaml

LOGGER = logging.getLogger("fetch_fdir")

CONFIG_PATH = Path("configs/spb_default.yaml")
DATASET = "reanalysis-era5-single-levels"
VARIABLE = "total_sky_direct_solar_radiation_at_surface"
MAX_ATTEMPTS_PER_YEAR = 3
RETRY_BACKOFF_SECONDS = 60.0


def _is_transient_cds_error(exc: BaseException) -> bool:
    """True for CDS errors worth retrying within the same script run.

    The motivating cases on this project so far are SSL EOFs during the
    job-status polling (``SSL: UNEXPECTED_EOF_WHILE_READING``) and generic
    HTTPS connection-pool resets. Both manifest as ``str(exc)`` mentioning
    'SSL', 'connection', 'EOF', or 'timeout'. Permanent errors (bad
    request, missing variable) are not retried.
    """
    msg = str(exc).lower()
    transient = ("ssl", "eof", "connection", "timeout", "gateway",
                 "too many requests", "temporarily")
    return any(token in msg for token in transient)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = yaml.safe_load(CONFIG_PATH.read_text())["era5_v2"]
    bbox = list(cfg["area"]["bbox"])
    raw_dir = Path(cfg["cache"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    start_year = int(str(cfg["time"]["start"])[:4])
    end_year = int(str(cfg["time"]["end"])[:4])
    years = list(range(start_year, end_year + 1))

    client = cdsapi.Client()
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    hours = [f"{h:02d}:00" for h in range(24)]

    submitted = 0
    skipped = 0
    failed: list[int] = []

    for year in years:
        target = raw_dir / f"era5_spb_v2_fdir_{year}.nc"
        if target.exists() and target.stat().st_size > 0:
            LOGGER.info("cached: %s", target)
            skipped += 1
            continue

        tmp_path = target.with_suffix(".nc.tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        request = {
            "product_type": "reanalysis",
            "variable": [VARIABLE],
            "year": [str(year)],
            "month": months,
            "day": days,
            "time": hours,
            "area": bbox,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        year_succeeded = False
        for attempt in range(1, MAX_ATTEMPTS_PER_YEAR + 1):
            LOGGER.info(
                "submitting fdir %s -> %s (attempt %d/%d)",
                year, target, attempt, MAX_ATTEMPTS_PER_YEAR,
            )
            t0 = time.monotonic()
            try:
                client.retrieve(DATASET, request, str(tmp_path))
                tmp_path.rename(target)
                submitted += 1
                LOGGER.info("completed fdir %s in %.1f min", year, (time.monotonic() - t0) / 60.0)
                year_succeeded = True
                break
            except Exception as exc:
                if tmp_path.exists():
                    tmp_path.unlink()
                if attempt < MAX_ATTEMPTS_PER_YEAR and _is_transient_cds_error(exc):
                    LOGGER.warning(
                        "transient CDS error on fdir %s attempt %d/%d; "
                        "sleeping %.0fs before retry: %s",
                        year, attempt, MAX_ATTEMPTS_PER_YEAR, RETRY_BACKOFF_SECONDS, exc,
                    )
                    time.sleep(RETRY_BACKOFF_SECONDS)
                    continue
                LOGGER.error("FAILED fdir %s after %d attempt(s): %s", year, attempt, exc)
                failed.append(year)
                break
        if not year_succeeded and year not in failed:
            failed.append(year)

    LOGGER.info(
        "done: submitted=%d skipped=%d failed=%d (failed_years=%s)",
        submitted,
        skipped,
        len(failed),
        failed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
