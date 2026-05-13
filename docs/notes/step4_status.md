Step 4 completed on 2026-05-13: the archive-ready lookup table was written to
`data/processed/p_favorable_spb.nc` with size 98.2 KiB.
The dataset shape is `cell=35`, `sector=18`, `season=4`, `period=3`, with
`p_favorable[cell, sector, season, period]`, `p_favorable_wind[cell, sector, season, period]`,
`p_favorable_thermal[cell, season, period]`, and `n_samples[cell, season, period]`.
All sanity checks pass: all four variables are present, there are no NaNs, probabilities are
within [0, 1], `p_favorable` is everywhere greater than or equal to both component probabilities,
and period sample counts follow the expected 12/4/8-hour split.
Mean sample counts per cell-season-period are day 12054, evening 4018, and night 8036, with exact
0.500/0.167/0.333 period fractions within each season, so the midnight-wrapping night bin is correct.
For cell 16, the sector-normalized weighted mean `p_favorable` is 0.553; the domain-wide weighted
mean is 0.590, which matches the earlier ~0.59 headline more closely.
Note that the literal round-trip expression without dividing by the 18 sectors returns 9.948 because
`n_samples` intentionally has no sector dimension in the final contract.
The headline thermal contrast is strong: mean `p_favorable_thermal` across cells is 0.530 for
DJF-night versus 0.145 for JJA-day, a 3.65x winter-night to summer-day contrast.
