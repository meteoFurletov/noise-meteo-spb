"""
CLI entry points for the five-step pipeline.

This module is the canonical definition of the pipeline. Each command
corresponds to one step described in README.md and to one chapter section
in docs/article_draft.md.

Usage:
    python -m src.run all                  # run the whole pipeline
    python -m src.run fetch                # step 1: fetch ERA5
    python -m src.run stability            # step 2: classify stability
    python -m src.run favorable            # step 3: per-sector favorable flag
    python -m src.run aggregate            # step 4: build the lookup table
    python -m src.run viz                  # step 5: produce paper figures

All commands accept --config to point at an alternative YAML configuration.
The default is configs/spb_default.yaml.
"""

from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    help="Pipeline for SPb meteorological propagation climatology.",
)

DEFAULT_CONFIG = Path("configs/spb_default.yaml")


@app.command()
def fetch(config: Path = DEFAULT_CONFIG) -> None:
    """Step 1. Pull ERA5 hourly data for the SPb domain to local Zarr cache.

    Reads:
        - configs/<config>.yaml (domain bounds, time window, variable list)
        - ERA5 cloud (ARCO Zarr) or CDS API depending on era5.source

    Writes:
        - data/interim/era5_spb.zarr (lazy-openable xarray dataset)

    Notes:
        See src/data/__init__.py for the dataset contract — what dimensions,
        coordinates, and variables are guaranteed for downstream steps.
    """
    raise NotImplementedError("Step 1: see src/data/")


@app.command()
def stability(config: Path = DEFAULT_CONFIG) -> None:
    """Step 2. Compute Richardson and Pasquill stability classes for every hour.

    Reads:
        - data/interim/era5_spb.zarr

    Writes:
        - data/interim/stability.zarr (adds stability_ri, stability_pasquill)
        - docs/figures/stability_agreement.pdf (diagnostic confusion matrix)

    Notes:
        Two independent classifiers, both retained. Agreement statistics are
        a paper result, not just a sanity check.
        See src/stability/__init__.py for class definitions and methods.
    """
    raise NotImplementedError("Step 2: see src/stability/")


@app.command()
def favorable(config: Path = DEFAULT_CONFIG) -> None:
    """Step 3. Compute per-sector favorable-condition flag for every hour.

    Reads:
        - data/interim/era5_spb.zarr (wind components)
        - data/interim/stability.zarr (stability class)

    Writes:
        - data/interim/favorable.zarr (favorable[time, cell, sector] bool)

    Notes:
        Implements ISO 9613-2 / ГОСТ 31295.2 logic strictly. Default wind
        threshold is 2.0 m/s; configurable for sensitivity analysis.
        See src/favorable/__init__.py for the criterion definition.
    """
    raise NotImplementedError("Step 3: see src/favorable/")


@app.command()
def aggregate(config: Path = DEFAULT_CONFIG) -> None:
    """Step 4. Aggregate per-hour flags into the climatological lookup table.

    Reads:
        - data/interim/favorable.zarr

    Writes:
        - data/processed/p_favorable_spb.nc (THE deliverable)

    Notes:
        Output dimensions: (cell, sector, season, period). Two variables:
        p_favorable (fraction) and n_samples (count). Sample count is
        carried alongside p_favorable so downstream consumers can flag
        statistically thin estimates.
        See src/aggregate/__init__.py for the binning specification.
    """
    raise NotImplementedError("Step 4: see src/aggregate/")


@app.command()
def viz(config: Path = DEFAULT_CONFIG) -> None:
    """Step 5. Produce paper figures from the lookup table.

    Reads:
        - data/processed/p_favorable_spb.nc

    Writes:
        - docs/figures/polar_p_favorable_<cell>.pdf (headline polar plot)
        - docs/figures/map_p_favorable_<sector>_<season>.pdf (spatial maps)
        - docs/figures/seasonal_cycle_p_favorable.pdf (annual cycle)
        - docs/figures/sector_season_heatmap.pdf (overview matrix)

    Notes:
        Each figure corresponds to a specific reference in the paper draft.
        See src/viz/__init__.py for the figure list and what each shows.
    """
    raise NotImplementedError("Step 5: see src/viz/")


@app.command()
def all(config: Path = DEFAULT_CONFIG) -> None:
    """Run all five steps in sequence."""
    fetch(config)
    stability(config)
    favorable(config)
    aggregate(config)
    viz(config)


if __name__ == "__main__":
    app()
