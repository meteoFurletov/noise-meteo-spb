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
    python -m src.run soundings            # validation: Voeikovo radiosondes
    python -m src.run aggregate            # step 4: build the lookup table
    python -m src.run viz                  # step 5: produce paper figures

All commands accept --config to point at an alternative YAML configuration.
The default is configs/spb_default.yaml.
"""

from pathlib import Path

import typer
import xarray as xr

from src.config import load_config
from src.aggregate import aggregate_to_climatology, write_netcdf
from src.data import fetch_era5, open_cached
from src.data.soundings import (
    assemble_soundings,
    compute_sounding_richardson,
    write_soundings_dataset,
)
from src.favorable import compute_favorable
from src.stability import agreement_diagnostics, classify_stability, plot_stability_agreement

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
    fetch_era5(load_config(config))


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
    loaded_config = load_config(config)
    era5 = open_cached()
    classified = classify_stability(era5, loaded_config)
    plot_stability_agreement(classified, Path("docs/figures/stability_agreement.pdf"))

    diagnostics = agreement_diagnostics(classified)
    typer.echo(
        "Stability classification complete: "
        f"{diagnostics['agreement_fraction']:.1%} agreement over "
        f"{diagnostics['n_valid']:,} valid samples."
    )


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
    loaded_config = load_config(config)
    era5 = open_cached()
    stability_ds = xr.open_zarr("data/interim/stability.zarr", consolidated=False)
    out = compute_favorable(era5, stability_ds, loaded_config)

    typer.echo(
        "Favorable-condition classification complete: "
        f"{', '.join(out.data_vars)} written to data/interim/favorable.zarr."
    )


@app.command()
def soundings(config: Path = DEFAULT_CONFIG) -> None:
    """Validation. Fetch Voeikovo soundings and compute sounding bulk Ri.

    Reads:
        - configs/<config>.yaml (soundings station/time settings)
        - University of Wyoming Atmospheric Soundings archive

    Writes:
        - data/raw/soundings/YYYY/MM/YYYYMMDD_HH.csv
        - data/external/station_id_cutover.json
        - data/interim/soundings_spb.zarr

    Notes:
        This is a validation side path for Step 3. It intentionally does not
        advance the main pipeline to aggregation.
    """
    loaded_config = load_config(config)
    sounding_config = loaded_config["soundings"]
    time_config = sounding_config["time"]

    ds = assemble_soundings(
        time_config["start"],
        time_config["end"],
        time_config.get("sample_strategy", "seasonal"),
    )
    out = compute_sounding_richardson(ds)
    write_soundings_dataset(out)

    valid = int(out["valid"].sum().item())
    total = int(out.sizes["time"])
    typer.echo(
        "Sounding validation cache complete: "
        f"{valid:,}/{total:,} launches valid; ri_sounding written to "
        "data/interim/soundings_spb.zarr."
    )


@app.command()
def aggregate(config: Path = DEFAULT_CONFIG) -> None:
    """Step 4. Aggregate per-hour flags into the climatological lookup table.

    Reads:
        - data/interim/favorable.zarr

    Writes:
        - data/processed/p_favorable_spb.nc (THE deliverable)

    Notes:
        Output dimensions: (cell, sector, season, period) for directional
        probabilities and (cell, season, period) for thermal probability and
        sample count. Sample count is carried alongside p_favorable so
        downstream consumers can flag statistically thin estimates.
        See src/aggregate/__init__.py for the binning specification.
    """
    loaded_config = load_config(config)
    favorable_ds = xr.open_zarr("data/interim/favorable.zarr", consolidated=False)
    out = aggregate_to_climatology(favorable_ds, loaded_config)

    output_config = loaded_config.get("output", {})
    output_path = Path(output_config.get("processed_path", "data/processed/p_favorable_spb.nc"))
    write_netcdf(out, output_path, output_config.get("metadata", {}))

    typer.echo(
        "Climatological lookup table complete: "
        f"{', '.join(out.data_vars)} written to {output_path}."
    )


@app.command()
def viz(config: Path = DEFAULT_CONFIG) -> None:
    """Step 5. Produce paper figures from the lookup table.

    Reads:
        - data/processed/p_favorable_spb.nc

    Writes:
        - docs/figures/fig1_two_component.pdf
        - docs/figures/fig2_seasonal_thermal.pdf
        - docs/figures/fig3_hero_kad.pdf
        - docs/figures/fig4_validation.pdf
        - docs/figures/fig5_pipeline.pdf

    Notes:
        Each figure corresponds to a specific reference in the paper draft.
        See src/viz/__init__.py for the figure list and what each shows.
    """
    from src.viz import generate_all_figures

    loaded_config = load_config(config)
    output_config = loaded_config.get("output", {})
    climatology_path = Path(
        output_config.get("processed_path", "data/processed/p_favorable_spb.nc")
    )
    climatology = xr.open_dataset(climatology_path)
    output_dir = Path("docs/figures")
    generate_all_figures(climatology, output_dir, loaded_config)

    typer.echo(f"Publication figures written to {output_dir}.")


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
