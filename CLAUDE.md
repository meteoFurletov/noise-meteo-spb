# CLAUDE.md

This file is the entry point for AI-assisted development on this repository. Read this first; it tells you where the knowledge lives.

## What this project is

A Python pipeline that processes ten years of ERA5 reanalysis data over the Saint Petersburg domain into a small climatological lookup table of "favorable propagation conditions" probabilities. The lookup table is used as input to traffic noise calculations made under Russian normative methods.

The full scientific motivation is in `docs/article_draft.md`. **Read at least the introduction and Chapter 3 «Стратегия» before making methodological decisions.** This is not a generic acoustics or meteorology project; it has specific commitments to specific Russian normative documents and specific design choices.

## Working principles for this repo

**Empirical before defensive.** Many design choices in this pipeline (wind threshold, stability classification, sector binning) involve methodological tradeoffs. The right approach is to implement, run, look at results, then defend in writing. Do not write extensive justification before the empirical evidence exists.

**The lookup table is the product.** Everything else — fetching, classification, per-hour favorable flags — is intermediate and can be discarded after the lookup table exists. Don't over-engineer the intermediate stages. Do over-engineer the lookup table format and metadata.

**Match the paper.** Code modules correspond directly to sections of the paper. A change in code structure should be reflected in the paper draft, and vice versa. If the code does something the paper doesn't describe, that's a documentation bug.

**Russian normative basis is non-negotiable.** ISO 9613-2 / ГОСТ 31295.2 logic is the foundation. Deviations from the standard are scientific contributions and must be explicit; defaults must match the standard. When in doubt, check `docs/article_draft.md` Section 1.3.3.

## Where things live

- `docs/article_draft.md` — the paper. Contains the scientific reasoning, the methodological decisions, and the literature context. **Section 3.1 enumerates the eight key methodological decisions** that this code implements.
- `configs/spb_default.yaml` — every numerical parameter (domain bounds, time window, wind threshold, stability cutoffs, sector count). Change parameters here, not in code.
- `src/run.py` — the five-step CLI. This is the canonical pipeline definition.
- `src/data/` — Step 1. ERA5 fetch and assembly. Owns the contract that downstream code receives a clean `xarray.Dataset` with documented dimensions and variables.
- `src/stability/` — Step 2. Two independent stability classifiers (Richardson, Pasquill) plus an agreement diagnostic.
- `src/favorable/` — Step 3. Per-sector favorable-condition computation following ISO 9613-2 logic.
- `src/aggregate/` — Step 4. Climatological aggregation to the lookup table.
- `src/viz/` — Step 5. Paper figures. Each function in this module produces one specific figure referenced in the paper.

## Steps and their contracts

Each step has a docstring at the top of its `__init__.py` describing inputs, outputs, and key methodological choices. Read the docstring before modifying the step.

| Step | Module | Reads | Writes |
|---|---|---|---|
| 1. fetch | `src/data` | ERA5 cloud / CDS | `data/interim/era5_spb.zarr` |
| 2. stability | `src/stability` | step 1 output | `data/interim/stability.zarr` |
| 3. favorable | `src/favorable` | steps 1, 2 | `data/interim/favorable.zarr` |
| 4. aggregate | `src/aggregate` | step 3 | `data/processed/p_favorable_spb.nc` |
| 5. viz | `src/viz` | step 4 | `docs/figures/*.{pdf,png}` |

## Decisions already made (do not relitigate without evidence)

These are nailed down in the paper and should not change without empirical or supervisor input:

- **Domain**: Saint Petersburg metropolitan area, roughly 59.5°–60.5°N, 29.5°–31.0°E.
- **Time window**: 2014–2024 inclusive, ten years of hourly data.
- **Reanalysis**: ERA5 (not ERA5-Land for this iteration; ERA5-Land is a future-work upgrade).
- **Sector count**: 18 sectors of 20°.
- **Seasons**: DJF, MAM, JJA, SON.
- **Lden periods**: day 07:00–19:00, evening 19:00–23:00, night 23:00–07:00 local time (UTC+3).
- **Stability methods**: bulk Richardson primary, Pasquill cross-check.
- **Favorable criterion**: ISO 9613-2 logic — wind component along sector ≥ threshold OR stable stratification.

## Decisions under sensitivity analysis (parameters)

These are configured in `configs/spb_default.yaml` and should be sweepable:

- Wind threshold for "favorable" wind component (ISO 9613-2 says 1–5 m/s; CNOSSOS uses ~2 m/s; check 1, 2, 3).
- Richardson cutoffs for stability classes.
- Reference height for upper wind in Richardson (100 m vs. lowest model level above 100 m).

## Things to be careful about

**Time zone.** ERA5 is in UTC. Saint Petersburg is UTC+3 year-round (no daylight saving). Day/night classification must use *local* time, not UTC. The day/evening/night boundaries in `configs/spb_default.yaml` are local-time hours; conversion happens once when computing the period coordinate.

**Sector convention.** Azimuth angles are measured from north (0°), clockwise, to where the wind is blowing *toward*. So a sector centered at 90° is "wind blowing east." This is the meteorological convention and matches ERA5; do not invert it.

**Wind component along sector.** A sector is a direction *from the source toward the receiver*, i.e., the direction noise is propagating. Favorable wind means wind has a positive component in that direction. Don't confuse this with the meteorological "wind direction" which is the direction the wind is *coming from* — it's the opposite sign by convention.

**Empty bins.** Some (cell, sector, season, period) combinations will have very few samples (rare wind directions in specific seasons). Always carry `n_samples` alongside `p_favorable` and never display a probability without the sample count.

**Coastal effects.** Saint Petersburg is on the Gulf of Finland. The southern shore has different meteorological behavior from the inland eastern parts. Expect spatial structure in the output. Don't average it away.

## How to extend

Adding a new derived quantity (e.g., a different stability classifier, a different favorable criterion) means: a new submodule under `src/`, a new step in `src/run.py`, a new section in the paper. All three change together.

Adding a new visualization means: a new function in `src/viz/`, a corresponding mention in the paper's figure list, no other changes.

Changing a parameter means: edit `configs/spb_default.yaml`, run the pipeline again, update the paper if the change is methodological.

## What this code does NOT do

For clarity about scope:

- It does not compute noise levels. Noise calculations consume the lookup table; they are implemented separately (see the `notebooks/` interactive prototype, and ultimately a sibling `noise-spb` repo).
- It does not produce buffer-zone maps. Those are downstream of noise calculation.
- It does not do urban-canopy-resolving meteorology. ERA5 at ~31 km is mesoscale; coastal vs. inland differences will be resolved, but street-level effects will not.
- It is not a CNOSSOS-EU implementation. It produces an input that *could* feed CNOSSOS-EU, but the propagation calculation itself follows the simpler ГОСТ 31295.2 framework.

## Pointers when stuck

- **Methodological question** → `docs/article_draft.md` Section 1.3, 1.5, or 3.1.
- **Parameter question** → `configs/spb_default.yaml`, then the relevant module's docstring.
- **What does this step do** → the `__init__.py` docstring of the step's module.
- **Why this design choice** → `docs/article_draft.md` Section 3.1 (the eight decisions).
- **External library question** → `pyproject.toml` for versions, then upstream docs (xarray, cartopy, numpy).
