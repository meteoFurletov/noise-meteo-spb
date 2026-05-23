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
import logging
import time

import typer
import xarray as xr

from src.config import load_config
from src.aggregate import aggregate_to_climatology, write_netcdf
from src.data import fetch_era5, open_cached
from src.data.era5 import (
    _load_v2_config,
    assemble_zarr,
    assemble_zarr_from_raw_cache,
    describe_backfill_dry_run,
    describe_dry_run,
    describe_timeseries_dry_run,
    enqueue_backfill_requests,
    fetch_all,
    fetch_pressure_then_missing_single_levels,
    fetch_single_levels_timeseries_all,
    verify_cache,
)
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


@app.command("era5_v2")
def era5_v2(
    config: Path = DEFAULT_CONFIG,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print CDS requests without contacting CDS.",
    ),
    year: int | None = typer.Option(
        None,
        "--year",
        help="Fetch a single year for debugging instead of the full configured range.",
    ),
) -> None:
    """Step 1 v2. Fetch the expanded ERA5 CDS cache.

    This is a separate command, rather than a --version flag on fetch, so the
    legacy ARCO-backed v1 cache remains available for old notebooks while the
    CDS-backed v2 cache can be generated explicitly.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _load_v2_config(config)
    selected_years = [year] if year is not None else None

    if dry_run:
        for line in describe_dry_run(cfg, selected_years):
            typer.echo(line)
        return

    started = time.monotonic()
    downloaded_paths = fetch_all(cfg, selected_years)
    fetch_elapsed = time.monotonic() - started

    zarr_path = assemble_zarr(downloaded_paths, cfg)
    processing_elapsed = time.monotonic() - started - fetch_elapsed
    checks = verify_cache(zarr_path, cfg)
    total_elapsed = time.monotonic() - started
    zarr_size_mb = _path_size_bytes(zarr_path) / 1024**2

    typer.echo(
        "ERA5 v2 status: "
        f"elapsed {total_elapsed / 60:.1f} min "
        f"(CDS fetch/queue {fetch_elapsed / 60:.1f} min, processing "
        f"{processing_elapsed / 60:.1f} min). "
        f"Final zarr size is {zarr_size_mb:.1f} MB "
        f"({'near' if 140 <= zarr_size_mb <= 230 else 'outside'} the expected ~180 MB range)."
    )
    typer.echo(
        "Verification: "
        f"z(1000 hPa) < z_surface fraction = "
        f"{checks['z1000_below_z_surface_fraction']:.4%}; "
        f"sdfor < 50 m everywhere = {checks['sdfor_lt_50m_everywhere']}."
    )
    if checks["sdfor_offending_cells"]:
        typer.echo(f"sdfor offending cells: {checks['sdfor_offending_cells']}")
    if not checks["time_covers_config"]:
        typer.echo(
            "Note: cache does not cover the full configured 2014-2024 window; "
            "this is expected for --year debugging runs."
        )


@app.command("era5_v2_assemble")
def era5_v2_assemble(config: Path = DEFAULT_CONFIG) -> None:
    """Assemble and verify the completed hybrid ERA5 v2 raw cache."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _load_v2_config(config)

    started = time.monotonic()
    zarr_path = assemble_zarr_from_raw_cache(cfg)
    processing_elapsed = time.monotonic() - started
    checks = verify_cache(zarr_path, cfg)
    zarr_size_mb = _path_size_bytes(zarr_path) / 1024**2

    typer.echo(
        "ERA5 v2 assemble complete: "
        f"processing {processing_elapsed / 60:.1f} min, "
        f"zarr={zarr_path}, size={zarr_size_mb:.1f} MB."
    )
    typer.echo(
        "Verification: "
        f"time {checks['time_first']} to {checks['time_last']}, "
        f"gaps={checks['time_has_gaps']}, "
        f"z(1000 hPa) < z_surface fraction="
        f"{checks['z1000_below_z_surface_fraction']:.4%}, "
        f"sdfor < 50 m everywhere={checks['sdfor_lt_50m_everywhere']}."
    )
    if checks["sdfor_offending_cells"]:
        typer.echo(f"sdfor offending cells: {checks['sdfor_offending_cells']}")


@app.command("era5_v2_timeseries")
def era5_v2_timeseries(
    config: Path = DEFAULT_CONFIG,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print point time-series requests without contacting CDS.",
    ),
) -> None:
    """Fetch fast single-level point time-series variables for all SPb grid cells."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _load_v2_config(config)
    if dry_run:
        for line in describe_timeseries_dry_run(cfg):
            typer.echo(line)
        return

    started = time.monotonic()
    result = fetch_single_levels_timeseries_all(cfg)
    elapsed = time.monotonic() - started
    typer.echo(
        "ERA5 v2 time-series complete: "
        f"{len(result['paths'])}/{result['n_points']} grid points, "
        f"{result['n_variables']} variables, "
        f"{result['n_cached']} cached, "
        f"elapsed {elapsed / 60:.1f} min, raw_dir={result['raw_dir']}."
    )


@app.command("era5_v2_backfill")
def era5_v2_backfill(
    config: Path = DEFAULT_CONFIG,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print pressure/missing-single requests without contacting CDS.",
    ),
    year: int | None = typer.Option(
        None,
        "--year",
        help="Fetch a single year instead of the full configured range.",
    ),
    start_year: int | None = typer.Option(
        None,
        "--start-year",
        help="First year to fetch when --year is not set.",
    ),
    end_year: int | None = typer.Option(
        None,
        "--end-year",
        help="Last year to fetch when --year is not set.",
    ),
    stage: str = typer.Option(
        "all",
        "--stage",
        help="Which backfill stage to run: pressure, missing_single_levels, or all.",
    ),
) -> None:
    """Backfill slow CDS fields: pressure first, then missing single-level vars."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _load_v2_config(config)
    selected_years = _selected_years(year, start_year, end_year)

    if dry_run:
        for line in describe_backfill_dry_run(cfg, selected_years, stage):
            typer.echo(line)
        return

    started = time.monotonic()
    result = fetch_pressure_then_missing_single_levels(cfg, selected_years, stage)
    elapsed = time.monotonic() - started
    typer.echo(
        "ERA5 v2 backfill complete: "
        f"{len(result['pressure_levels'])} pressure file(s), "
        f"{len(result['missing_single_levels'])} missing-single file(s), "
        f"elapsed {elapsed / 60:.1f} min."
    )


@app.command("era5_v2_enqueue")
def era5_v2_enqueue(
    config: Path = DEFAULT_CONFIG,
    year: int | None = typer.Option(
        None,
        "--year",
        help="Submit a single year instead of the full configured range.",
    ),
    start_year: int | None = typer.Option(
        None,
        "--start-year",
        help="First year to submit when --year is not set.",
    ),
    end_year: int | None = typer.Option(
        None,
        "--end-year",
        help="Last year to submit when --year is not set.",
    ),
    stage: str = typer.Option(
        "all",
        "--stage",
        help="Which backfill stage to enqueue: pressure, missing_single_levels, or all.",
    ),
    log_path: Path = typer.Option(
        Path("logs/era5_v2_enqueue_requests.jsonl"),
        "--log-path",
        help="JSONL file where submitted CDS request IDs are recorded.",
    ),
) -> None:
    """Submit packed ERA5 v2 backfill requests to CDS without waiting."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _load_v2_config(config)
    selected_years = _selected_years(year, start_year, end_year)
    result = enqueue_backfill_requests(cfg, selected_years, stage, log_path)
    typer.echo(
        "ERA5 v2 enqueue complete: "
        f"{result['submitted']} submitted, "
        f"{result['skipped']} skipped, "
        f"{result['failed']} failed, "
        f"log={result['log_path']}."
    )


def _selected_years(
    year: int | None,
    start_year: int | None,
    end_year: int | None,
) -> list[int] | None:
    if year is not None:
        if start_year is not None or end_year is not None:
            raise typer.BadParameter("--year cannot be combined with --start-year/--end-year")
        return [year]
    if start_year is None and end_year is None:
        return None
    if start_year is None or end_year is None:
        raise typer.BadParameter("--start-year and --end-year must be provided together")
    if end_year < start_year:
        raise typer.BadParameter("--end-year must be greater than or equal to --start-year")
    return list(range(start_year, end_year + 1))


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


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


if __name__ == "__main__":
    app()
