# CLAUDE.md

This file is the entry point for AI-assisted development on this repository. Read this first; it tells you where the knowledge lives.

## What this project is

A Python pipeline that processes hourly ERA5 reanalysis data (2014–2024) over the Saint Petersburg domain into a small climatological lookup table of "favorable propagation conditions" probabilities. The lookup table is used as input to traffic noise calculations made under Russian normative methods. The pipeline is complete: the lookup tables, validation dataset and paper figures are committed under `data/processed/` and `docs/figures/phase_f_v2/`, and the per-step reports under `docs/*_report.md` hold the numbers quoted in the thesis.

The full scientific motivation is in `docs/article_draft.md`. **Read at least the introduction and Chapter 3 «Стратегия» before making methodological decisions.** This is not a generic acoustics or meteorology project; it has specific commitments to specific Russian normative documents and specific design choices.

## Working principles for this repo

**Empirical before defensive.** Many design choices in this pipeline (wind threshold, stability classification, sector binning) involve methodological tradeoffs. The right approach is to implement, run, look at results, then defend in writing. Do not write extensive justification before the empirical evidence exists.

**The lookup table is the product.** Everything else — fetching, classification, per-hour favorable flags — is intermediate and can be discarded after the lookup table exists. Don't over-engineer the intermediate stages. Do over-engineer the lookup table format and metadata.

**Match the paper.** Code modules correspond directly to sections of the paper. A change in code structure should be reflected in the paper draft, and vice versa. If the code does something the paper doesn't describe, that's a documentation bug.

**Russian normative basis is non-negotiable.** ISO 9613-2 / ГОСТ 31295.2 logic is the foundation. Deviations from the standard are scientific contributions and must be explicit; defaults must match the standard. When in doubt, check `docs/article_draft.md` Section 1.3.3.

## Where things live

- `docs/article_draft.md` — the paper. Contains the scientific reasoning, the methodological decisions, and the literature context. **Section 3.1 enumerates the eight key methodological decisions** that this code implements.
- `thesis_doc/` — the thesis sources (chapters, figure registry, bibliography), compiled to `.docx` by `src/thesisgen/` via `scripts/thesis_build.py`.
- `docs/step*_report.md`, `docs/voeikovo_validation.md` — per-step reports; the source of every number quoted in the thesis and README.
- `configs/spb_default.yaml` — every numerical parameter (domain bounds, time window, variable lists, stability thresholds, wind threshold, sector count, case-study geometry). Change parameters here, not in code.
- `src/run.py` — Typer CLI. Hosts the v2 ERA5 fetch commands (`era5_v2*`), `soundings`, `viz`, and the archived v1 steps. Steps 2–4 of the current (v2) pipeline are run as `python -m src.<step>.runner`.
- `src/data/` — Step 1. `era5.py` is the CDS-based v2 fetch and Zarr assembly; `soundings.py` loads Voeikovo radiosondes for validation; `__init__.py` keeps the v1 ARCO path.
- `src/stability/` — Step 2. Three classifiers (`richardson.py`, `monin_obukhov.py`, `pasquill_turner.py`), QC gates (`qc.py`), and `runner.py`. The v1 two-classifier code is under `_v1_archive/`.
- `src/favorable/` — Step 3. `wind.py` and `thermal.py` implement the ISO 9613-2 criteria; `runner.py` combines them and writes the per-sector flag.
- `src/aggregate/` — Step 4. `aggregator.py` and `time_bins.py` produce the lookup table; `cnossos_combiner.py` produces the CNOSSOS-EU companion; `runner.py` has `--mode primary|cnossos`.
- `src/validation/` — ERA5 Richardson vs radiosonde Richardson at Voeikovo (`voeikovo_v2.py`).
- `src/case_study/` — KAD km 105 receiver grid and the minimal MGSU/NMPB propagation model behind the buffer-zone figures.
- `src/viz/` — Step 5. Paper figures. Each function in this module produces one specific figure referenced in the paper; `generate_all_figures` and `generate_paper_png_figures` list them.
- `src/exploration/` — station audits and sensitivity comparisons that back the `docs/*_comparison*.md` notes.

## Steps and their contracts

Each step has a docstring at the top of its `__init__.py` describing inputs, outputs, and key methodological choices. Read the docstring before modifying the step.

| Step | Entry point | Reads | Writes |
|---|---|---|---|
| 1. fetch | `noise-meteo era5_v2` | CDS API | `data/interim/era5_spb_v2.zarr` |
| 2. stability | `python -m src.stability.runner` | step 1 output | `data/interim/stability_v2.zarr` |
| 3. favorable | `python -m src.favorable.runner` | steps 1, 2 | `data/interim/favorable_v2.zarr` |
| 4. aggregate | `python -m src.aggregate.runner [--mode cnossos]` | step 3 | `data/processed/p_favorable_spb.nc`, `p_favorable_cnossos.nc` |
| validation | `noise-meteo soundings`, `python -m src.validation.voeikovo_v2` | step 2, Wyoming archive | `data/processed/voeikovo_validation.nc`, `docs/voeikovo_validation.md` |
| 5. viz | `noise-meteo viz` | step 4 | `docs/figures/*.{pdf,png}` |

The v1 pipeline (`noise-meteo fetch / stability / favorable / aggregate`, ARCO-Zarr source, 1000 hPa upper level, two classifiers) is archived and kept runnable for comparison only.

## Decisions already made (do not relitigate without evidence)

These are nailed down in the paper and should not change without empirical or supervisor input:

- **Domain**: Saint Petersburg metropolitan area, roughly 59.5°–60.5°N, 29.5°–31.0°E.
- **Time window**: 2014–2024 inclusive, hourly (96 432 time steps).
- **Reanalysis**: ERA5 (not ERA5-Land for this iteration; ERA5-Land is a future-work upgrade).
- **Sector count**: 18 sectors of 20°.
- **Seasons**: DJF, MAM, JJA, SON.
- **Lden periods**: day 07:00–19:00, evening 19:00–23:00, night 23:00–07:00 local time (UTC+3).
- **Stability methods**: bulk Richardson (2 m → 100 m, threshold 0.1) primary, with cascade to Pasquill–Turner when Ri is undefined in calm shear; Monin–Obukhov 1/L and Pasquill–Turner retained as sensitivity variants; Pasquill–Turner E–G is the thermal criterion of the CNOSSOS companion table.
- **Favorable criterion**: ISO 9613-2 logic — wind component along sector ≥ 2 m/s OR stable stratification.
- **Case study**: KAD ring road km 105, three modes (static MGSU, worst-case C_0 = 0, climatological), 45/55/60 dBA contours.

## Decisions under sensitivity analysis (parameters)

These are configured in `configs/spb_default.yaml` and should be sweepable:

- Wind threshold for "favorable" wind component (ISO 9613-2 says 1–5 m/s; CNOSSOS uses ~2 m/s; check 1, 2, 3).
- Richardson stable threshold and the minimum shear floor below which Ri is undefined.
- Thermal criterion variant (Ri with cascade, Ri strict, 1/L strict, 1/L moderate, Pasquill–Turner); all five are stored in the lookup table.

## Things to be careful about

**Time zone.** ERA5 is in UTC. Saint Petersburg is UTC+3 year-round (no daylight saving). Day/night classification must use *local* time, not UTC. The day/evening/night boundaries in `configs/spb_default.yaml` are local-time hours; conversion happens once when computing the period coordinate.

**Sector convention.** Azimuth angles are measured from north (0°), clockwise, to where the wind is blowing *toward*. So a sector centered at 90° is "wind blowing east." This is the meteorological convention and matches ERA5; do not invert it.

**Wind component along sector.** A sector is a direction *from the source toward the receiver*, i.e., the direction noise is propagating. Favorable wind means wind has a positive component in that direction. Don't confuse this with the meteorological "wind direction" which is the direction the wind is *coming from* — it's the opposite sign by convention.

**Empty bins.** Some (cell, sector, season, period) combinations will have very few samples (rare wind directions in specific seasons). Always carry `n_samples` alongside `p_favorable` and never display a probability without the sample count.

**Coastal effects.** Saint Petersburg is on the Gulf of Finland. The southern shore has different meteorological behavior from the inland eastern parts. Expect spatial structure in the output. Don't average it away.

## How to extend

Adding a new derived quantity (e.g., a different stability classifier, a different favorable criterion) means: a new submodule under `src/`, a new step in `src/run.py`, a new section in the paper. All three change together.

Adding a new visualization means: a new function in `src/viz/`, a call in `generate_all_figures` and `generate_paper_png_figures`, a corresponding mention in the paper's figure list, no other changes.

Tests (`uv run pytest`) run offline on synthetic data in a few seconds and lint (`uv run ruff check .`) must stay clean; CI runs both.

Changing a parameter means: edit `configs/spb_default.yaml`, run the pipeline again, update the paper if the change is methodological.

## What this code does NOT do

For clarity about scope:

- It does not compute certified noise levels. `src/case_study/` carries a deliberately minimal propagation model (MGSU 2015 emission table, NMPB-style distance correction) that exists only to show how buffer zones respond to the climatology.
- It does not produce production buffer-zone maps. The KAD km 105 figures are a demonstration on one segment.
- It does not do urban-canopy-resolving meteorology. ERA5 at ~31 km is mesoscale; coastal vs. inland differences will be resolved, but street-level effects will not.
- It is not a CNOSSOS-EU implementation. It produces an input that *could* feed CNOSSOS-EU, but the propagation calculation itself follows the simpler ГОСТ 31295.2 framework.

## Pointers when stuck

- **Methodological question** → `docs/article_draft.md` Section 1.3, 1.5, or 3.1.
- **Parameter question** → `configs/spb_default.yaml`, then the relevant module's docstring.
- **What does this step do** → the `__init__.py` docstring of the step's module.
- **Why this design choice** → `docs/article_draft.md` Section 3.1 (the eight decisions).
- **External library question** → `pyproject.toml` for versions, then upstream docs (xarray, cartopy, numpy).
