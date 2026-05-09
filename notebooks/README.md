# Notebooks

Exploratory notebooks live here. Anything that becomes part of the paper
should migrate from a notebook to a function in `src/viz/` and a CLI step.

Notebooks are not run as part of the pipeline. They are scratch space for:

- Looking at intermediate results during development
- Prototyping new figures before promoting them to `src/viz/`
- Validating against ground stations (Pulkovo, Voeykovo, Kronstadt)
- Sensitivity analyses (sweeping wind threshold, comparing stability methods)

Suggested first notebooks once the pipeline runs:

- `01_first_look.ipynb` — open the lookup table, plot every cell's polar
  rose on a grid, find the most asymmetric cells.
- `02_sensitivity_threshold.ipynb` — recompute step 4 with three wind
  thresholds, plot how p_favorable shifts.
- `03_validation_stations.ipynb` — compare ERA5 winds at the cells nearest
  to Pulkovo, Voeykovo, and Kronstadt with Roshydromet observations.
- `04_seasonal_inversion_frequency.ipynb` — characterize the SPb inversion
  climatology that drives the thermal-favorable component.

These are notebooks, not modules — keep them messy and exploratory. Migrate
results that survive to `src/`.
