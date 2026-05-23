# PNG Figure Status

Date: 2026-05-14 14:23 MSK

Step 2 completed. The original PDFs in `docs/figures/` were not deleted. The paper PNG set was regenerated from source code into `docs/figures/png/`.

## Generated Files

| PNG | Dimensions | File size | Status |
|---|---:|---:|---|
| `fig1_two_component.png` | 2554 x 1085 px | 237.8 KiB | OK. Russian compass labels render correctly; explicit azimuth convention label added. |
| `fig2_seasonal_thermal.png` | 1774 x 1279 px | 131.3 KiB | OK. Colorbar label made specific to `p_favorable_thermal`; long label split over two lines to avoid clipping. |
| `fig3_hero_kad.png` | 2606 x 2335 px | 1577.4 KiB | OK. Russian labels and `КАД 105 км` callout render cleanly. Basemap attribution remains in English as provider/license text, not a technical annotation. |
| `fig4_validation.png` | 3281 x 1127 px | 494.3 KiB | OK. Three-panel validation figure regenerated with Russian matrix labels, bounded seasonal scatter, and exact bar legend entries. |
| `fig5_stability_agreement.png` | 1762 x 1594 px | 325.0 KiB | OK. Promoted stability-agreement matrix regenerated as Fig. 5 with Russian title, axes, colorbar, and N annotation. |

All files are below the ~2 MB target.

## Rendering Checks

- Visual inspection confirmed Cyrillic text renders with DejaVu Serif; no boxes, question marks, or missing glyphs observed.
- Fig. 2 initially clipped the long horizontal colorbar label; fixed in source by using a horizontal colorbar with a two-line Russian label.
- Fig. 4 initially had a crowded bar legend; fixed in source by moving the legend below the bar panel with smaller text and sufficient clearance.
- Fig. 5 initially had the N annotation overlapping the title; fixed in source by adding title padding and placing the annotation below the title.

## Source Modules Touched

- `src/viz/__init__.py`
  - Standardized matplotlib rcParams for paper PNG output.
  - Added `generate_paper_png_figures(...)` for the climatology-derived PNG paper figures.
  - Russian-labeled and adjusted Figs. 1, 2, 3, and 4.
  - Changed Fig. 4 panel (b) from hexbin density to bounded seasonal scatter with legend and annotation.
- `src/stability/__init__.py`
  - Russian-labeled the Richardson vs Pasquill agreement matrix.
  - Added the Fig. 5 N annotation and PNG compression handling.

No notebooks were edited.

## Verification Commands

- `uv run ruff check src/viz/__init__.py src/stability/__init__.py`
- Regeneration from source:
  `uv run python -c "... generate_paper_png_figures(...); plot_stability_agreement(...)"` 
