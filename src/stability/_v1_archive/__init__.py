"""Archive of the v1 stability classification module.

Preserved for reference and so that existing callers (src/viz, tests, src/run)
keep importing the v1 API while the v2 pipeline takes over the top-level
``src.stability`` namespace. New code should import from the v2 modules
directly (``src.stability.richardson``, etc.), not from here.
"""

from src.stability._v1_archive._legacy_module import (  # noqa: F401
    CLASS_LABELS,
    CLASS_TO_INT,
    DEFAULT_OUTPUT_PATH,
    agreement_diagnostics,
    classify_stability,
    pasquill_class,
    plot_stability_agreement,
    richardson_bulk,
    solar_zenith_angle,
)
