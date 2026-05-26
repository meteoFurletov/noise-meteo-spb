"""GOST-compliant DOCX builder for the thesis.

Compiles the markdown sources under ``thesis_doc/`` into a single ``.docx``
formatted per GOST 7.32-2017 (host standard) and GOST R 7.0.100-2018
(bibliography). See ``docs/`` and ``CLAUDE.md`` for the project context.

Entry point: ``src.thesisgen.build.build_thesis(root: Path) -> Path``.
"""

from src.thesisgen.build import build_thesis

__all__ = ["build_thesis"]
