"""
Unit tests for stability classification.

These tests pin down:
- Richardson formula uses correct sign convention (positive Ri = stable)
- Class boundaries match config exactly
- Pasquill table lookup matches Stull (1988) Table 9.4 reference values
- Solar zenith calculation matches metpy / known reference values

These tests catch silent sign flips and off-by-one bin errors that would
quietly corrupt the climatology.
"""

import pytest


def test_positive_richardson_is_stable():
    """Positive bulk Ri must map to stability classes E/F/G."""
    pytest.skip("To implement once src/stability/ is filled in.")


def test_negative_richardson_is_unstable():
    """Negative bulk Ri must map to A/B/C."""
    pytest.skip("To implement.")


def test_richardson_class_boundaries_match_config():
    """Class assignment must respect the boundaries in spb_default.yaml."""
    pytest.skip("To implement.")


def test_pasquill_high_wind_clear_day_is_class_C_or_D():
    """Reference case from Stull (1988) Table 9.4: 6 m/s wind, strong
    insolation, clear sky — should give class C (slightly unstable)."""
    pytest.skip("To implement.")


def test_pasquill_calm_clear_night_is_class_F_or_G():
    """Reference case: calm wind, clear night — should give class F or G."""
    pytest.skip("To implement.")


def test_solar_zenith_at_pulkovo_summer_solstice_noon():
    """Solar zenith at Pulkovo (~60°N) at summer solstice noon should be
    about 90° - (90° - 60° + 23.4°) ≈ 36.6°."""
    pytest.skip("To implement.")
