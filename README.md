# noise-meteo-spb

**Climatology of meteorological propagation conditions for traffic noise calculation in Saint Petersburg.**

## What this is

A computational pipeline that produces a small climatological lookup table describing how often atmospheric conditions favor noise propagation in each direction, at each location, for each season and time of day across the Saint Petersburg metropolitan area. The lookup table is computed once from ten years of ERA5 reanalysis data and serves as the meteorological input to traffic noise calculations made under Russian normative methods (ГОСТ 31295.2-2005, СП 276.1325800.2016).

This project is the computational core of an academic paper being prepared for submission. The science is described in detail in `docs/article_draft.md`; this README explains how the code fits together.

## What the deliverable looks like

A NetCDF file at `data/processed/p_favorable_spb.nc` with dimensions:

- `cell` — ERA5 grid cells covering the Saint Petersburg domain (~50 cells)
- `sector` — 18 azimuth sectors of 20° each
- `season` — 4 (DJF, MAM, JJA, SON)
- `period` — 3 Lden time-of-day periods (day, evening, night)

And two variables:

- `p_favorable[cell, sector, season, period]` — fraction of hours with favorable propagation
- `n_samples[cell, sector, season, period]` — number of hourly observations behind each estimate

That single file, around 100 KB, is the principal scientific output. Everything else in this repo exists to produce it correctly and to visualize what it shows.

## Why this matters scientifically

Russian normative practice currently applies a single conservative meteorological correction (`C_0` in the language of ГОСТ 31295.2) when computing long-term traffic noise levels. This is a scalar — it doesn't vary with direction, season, or location. Western noise mapping practice (CNOSSOS-EU, Nord2000) replaces this scalar with a directional and seasonal climatology derived from local meteorological data. No equivalent computation has been published for any Russian region.

This repository fills that gap for Saint Petersburg.

## Pipeline at a glance

```
ERA5 hourly (cloud) ──> 1. fetch & open
                           │
                           ▼
                   xarray.Dataset (88k hours × 50 cells × 8 vars)
                           │
                           ▼
                       2. classify stability
                       (Richardson + Pasquill, both)
                           │
                           ▼
                       3. compute favorable per sector
                       (ISO 9613-2 logic: wind component OR thermal)
                           │
                           ▼
                       4. aggregate to climatology
                       (group by cell × sector × season × period)
                           │
                           ▼
                   p_favorable_spb.nc  ← THE DELIVERABLE
                           │
                           ▼
                       5. visualize
                       (polar plots, maps, seasonal cycles)
```

Steps 1–5 correspond to top-level CLI entry points in `src/run.py` and to the chapter sections in the paper.

## Quick start

```bash
# install
poetry install   # or: pip install -e .

# run the pipeline end to end (takes ~30 min on first run, including ERA5 fetch)
python -m src.run all --config configs/spb_default.yaml

# or run individual steps
python -m src.run fetch     # step 1: pull ERA5 to local Zarr cache
python -m src.run stability # step 2: compute Ri and Pasquill
python -m src.run favorable # step 3: per-sector favorable flag
python -m src.run aggregate # step 4: build the lookup table
python -m src.run viz       # step 5: produce paper figures
```

Each step writes to `data/interim/` or `data/processed/`. Reruns are cached.

## Repository structure

```
.
├── CLAUDE.md                # navigation map for AI-assisted development
├── README.md                # this file
├── pyproject.toml           # dependencies and project metadata
├── configs/
│   └── spb_default.yaml     # domain bounds, time window, thresholds
├── src/
│   ├── run.py               # CLI entry points for each pipeline step
│   ├── data/                # step 1: ERA5 fetch and dataset assembly
│   ├── stability/           # step 2: Richardson and Pasquill classifiers
│   ├── favorable/           # step 3: per-sector favorable-condition logic
│   ├── aggregate/           # step 4: climatological aggregation
│   └── viz/                 # step 5: figure generation
├── data/
│   ├── raw/                 # ERA5 downloads (gitignored)
│   ├── interim/             # per-step intermediate outputs (gitignored)
│   ├── processed/           # final deliverables (committed if small)
│   └── external/            # static reference data (e.g., station coords)
├── notebooks/               # exploratory analysis and figure prototyping
├── tests/                   # unit tests for stability and favorable logic
└── docs/
    ├── article_draft.md     # the scientific paper this code supports
    └── figures/             # generated figures for the paper
```

## Reading order for a new contributor

1. **This README** — you're here.
2. **`CLAUDE.md`** — pointers to where the actual implementation knowledge lives.
3. **`docs/article_draft.md`** — the scientific framing, the methodological choices, and why each step exists. The Chapter 3 «Стратегия» section explains the eight key decisions encoded in this code.
4. **`configs/spb_default.yaml`** — every parameter you might want to vary lives here.
5. **`src/run.py`** — the five-step CLI is the simplest expression of the pipeline.

## Status

Work in progress. The harness is in place; implementations are being filled in step by step.

## Citation

If you use this code or the resulting climatology, please cite the accompanying paper (in preparation) and the underlying ERA5 dataset (Hersbach et al., 2020; Copernicus Climate Change Service, 2023).

## License

[TBD — likely MIT for code, CC-BY for derived data products.]
