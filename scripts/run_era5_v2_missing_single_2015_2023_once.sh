#!/bin/zsh
set -u

LABEL="local.noise-meteo-spb.era5-v2-missing-single"
PROJECT_DIR="/Users/meteof/Projects/noise-meteo-spb"

cd "$PROJECT_DIR" || exit 1
.venv/bin/python -m src.run era5_v2_backfill \
  --stage missing_single_levels \
  --start-year 2015 \
  --end-year 2023
exit_code=$?

/bin/launchctl remove "$LABEL" >/dev/null 2>&1 || true
exit "$exit_code"
