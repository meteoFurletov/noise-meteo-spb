"""Back-compat shim for the v1 ``src.stability.validation`` module.

The v1 sounding-vs-ERA5 validation lives under ``_v1_archive``; existing
callers (``src/viz/__init__.py``, ``tests/test_sounding_validation.py``)
import from this module path, so re-export everything they need.
"""

from src.stability._v1_archive.validation import *  # noqa: F401,F403
from src.stability._v1_archive.validation import (  # noqa: F401
    agreement_statistics,
    find_nearest_era5_cell,
)
