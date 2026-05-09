"""
Unit tests for the favorable-propagation criterion.

These tests pin down the methodologically critical behavior:
- Sector convention (north = 0°, clockwise, propagation TOWARD)
- Wind component sign (positive = favorable for that sector)
- Thermal flag is omnidirectional
- Combined criterion is OR (not AND)

If any of these tests fail, the science is broken. They should be the
first thing implemented and the last thing changed.
"""

import pytest


def test_sector_zero_is_north():
    """Sector 0 should be centered on azimuth 0° (north)."""
    pytest.skip("To implement once src/favorable/ is filled in.")


def test_west_wind_favors_eastward_propagation():
    """Wind from the west (positive u) should mark sector 90° (east) as favorable."""
    pytest.skip("To implement.")


def test_below_threshold_wind_not_favorable_unless_stable():
    """Light wind (below threshold) should be wind-unfavorable, but stable
    stratification still flags all sectors as thermally favorable."""
    pytest.skip("To implement.")


def test_stable_stratification_favors_all_sectors():
    """When atmosphere is stable, all 18 sectors should be favorable
    regardless of wind direction."""
    pytest.skip("To implement.")


def test_combined_criterion_is_or_not_and():
    """An hour with strong wind in one sector should mark that sector as
    favorable even if stratification is unstable (lapse)."""
    pytest.skip("To implement.")


def test_wind_threshold_is_configurable():
    """Changing the threshold parameter should change which sectors are flagged."""
    pytest.skip("To implement.")
