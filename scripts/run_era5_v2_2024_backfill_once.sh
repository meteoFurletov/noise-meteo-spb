#!/bin/zsh
set -u

LABEL="local.noise-meteo-spb.era5-v2-backfill-2024"
PROJECT_DIR="/Users/meteof/Projects/noise-meteo-spb"

cd "$PROJECT_DIR" || exit 1
.venv/bin/python -m src.run era5_v2_backfill --year 2024
exit_code=$?

/bin/launchctl remove "$LABEL" >/dev/null 2>&1 || true
exit "$exit_code"
