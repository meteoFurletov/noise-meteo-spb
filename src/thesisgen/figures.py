"""Figure registry loaded from ``figures.yaml``.

A figure key is referenced from markdown body text via ``[[fig:key]]``; the
renderer expands it to an embedded PNG with the registered caption. Markdown
``![caption](path)`` images bypass the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}


class FigureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Figure:
    key: str
    path: Path
    caption: str


class FigureRegistry:
    def __init__(self, entries: dict[str, Figure]) -> None:
        self._entries = entries

    def get(self, key: str) -> Figure:
        if key not in self._entries:
            raise FigureError(f"Unknown figure key: [[fig:{key}]]. Add it to figures.yaml.")
        return self._entries[key]

    def __contains__(self, key: str) -> bool:
        return key in self._entries


def load_registry(yaml_path: Path, root: Path) -> FigureRegistry:
    """Load figures.yaml and validate every path exists and is a supported raster."""
    if not yaml_path.exists():
        return FigureRegistry({})
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    entries: dict[str, Figure] = {}
    problems: list[str] = []
    for key, payload in raw.items():
        if not isinstance(payload, dict) or "path" not in payload:
            problems.append(f"  {key}: entry must be a mapping with at least 'path'")
            continue
        path = (root / payload["path"]).resolve()
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            suggestion = path.with_suffix(".png")
            problems.append(
                f"  {key}: '{payload['path']}' has unsupported suffix '{suffix}'. "
                f"Save a PNG copy at {suggestion} instead."
            )
            continue
        if not path.exists():
            problems.append(f"  {key}: file not found at {path}")
            continue
        entries[key] = Figure(key=key, path=path, caption=payload.get("caption", ""))
    if problems:
        raise FigureError("Problems in figures.yaml:\n" + "\n".join(problems))
    return FigureRegistry(entries)


def validate_image_path(path: Path) -> None:
    """Raise FigureError if the path's suffix isn't a supported raster format."""
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise FigureError(
            f"Unsupported image format '{suffix}' for {path}. "
            f"python-docx can only embed PNG/JPG. "
            f"Save a PNG copy at {path.with_suffix('.png')} and update the reference."
        )
