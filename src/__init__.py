"""
noise-meteo-spb

Climatology of meteorological propagation conditions for traffic noise
calculation in Saint Petersburg.

The pipeline produces a single NetCDF file at data/processed/p_favorable_spb.nc
containing the probability of "favorable propagation conditions" (per ISO 9613-2
/ ГОСТ 31295.2) for each ERA5 grid cell in the SPb domain, broken down by
azimuth sector, season, and time-of-day period.

This file is consumed by downstream traffic noise calculation code to replace
the single conservative C_0 coefficient currently used in Russian normative
practice with a directionally and temporally resolved climatology.

See README.md for project framing and CLAUDE.md for navigation.
See docs/article_draft.md Sections 1.3, 1.5, and 3.1 for methodological context.
"""

__version__ = "0.1.0"
