"""
Configuration loading.

Reads YAML config files (default: configs/spb_default.yaml) and exposes
them as attribute-accessible objects. All numerical parameters used by
the pipeline pass through this module — do not hardcode values elsewhere.

Implementation note for Claude Code:
    A simple dotted-dict wrapper over the parsed YAML is sufficient here.
    Pydantic models would be nice but not required; the config schema is
    documented in configs/spb_default.yaml itself via inline comments.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML config file. Returns plain nested dicts."""
    with open(path) as f:
        return yaml.safe_load(f)
