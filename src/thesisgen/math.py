"""LaTeX → OMML via pandoc (single batched subprocess call).

Pandoc is the reference implementation for LaTeX→OOXML math; its output is
the same OMML structure Word itself produces, which means Word desktop will
open the result without complaint.

The naïve approach (one subprocess per formula) would take ~60 seconds for
the thesis's 600+ formulas. Instead we collect all unique formulas upfront,
batch-convert them in a single pandoc invocation using a sentinel paragraph
between each, and cache the results for the renderer to look up.

Public API:

- ``precompute(sources: Iterable[str])`` — builds the cache.
- ``get_oMath(src)`` — returns a *detached copy* of the ``<m:oMath>`` element,
  ready to inject into a paragraph as inline math.
- ``get_oMathPara(src)`` — same wrapped in ``<m:oMathPara>`` for display math.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lxml import etree

OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_SENTINEL = "PANDOCFORMULASENTINEL"
_SENTINEL_RE = re.compile(rf"^{_SENTINEL}(\d+)$")

# Module-level cache populated by precompute().
_CACHE: dict[str, etree._Element | None] = {}


@dataclass
class MathFailure:
    file: str
    line: int
    source: str
    error: str


def precompute(sources: Iterable[str]) -> None:
    """Run pandoc once to convert every unique formula; populate the cache.

    Idempotent — already-cached sources are skipped on re-entry.
    """
    todo = sorted({s.strip() for s in sources if s.strip() and s.strip() not in _CACHE})
    if not todo:
        return

    md_parts: list[str] = []
    for i, src in enumerate(todo):
        md_parts.append(f"{_SENTINEL}{i}")
        md_parts.append(f"$${src}$$")
    md = "\n\n".join(md_parts) + "\n"

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            ["pandoc", "-f", "markdown+tex_math_dollars", "-t", "docx", "-o", tmp_path],
            input=md,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if proc.returncode != 0:
            for src in todo:
                _CACHE[src] = None
            return
        with zipfile.ZipFile(tmp_path) as z:
            doc_xml = z.read("word/document.xml")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    tree = etree.fromstring(doc_xml)
    current_idx: int | None = None
    for p in tree.iter(f"{{{W_NS}}}p"):
        text = "".join(p.itertext()).strip()
        m = _SENTINEL_RE.match(text)
        if m:
            current_idx = int(m.group(1))
            continue
        if current_idx is None:
            continue
        omath_para = p.find(f"{{{OMML_NS}}}oMathPara")
        omath = omath_para.find(f"{{{OMML_NS}}}oMath") if omath_para is not None else p.find(f"{{{OMML_NS}}}oMath")
        if omath is not None:
            _CACHE[todo[current_idx]] = omath
        current_idx = None

    # Mark unresolved as None so the renderer logs a failure for them.
    for src in todo:
        _CACHE.setdefault(src, None)


def get_oMath(src: str) -> etree._Element | None:
    """Return a detached deep copy of the cached ``<m:oMath>`` for ``src``, or None."""
    el = _CACHE.get(src.strip())
    return deepcopy(el) if el is not None else None


def get_oMathPara(src: str) -> etree._Element | None:
    """Wrap the cached ``<m:oMath>`` in an ``<m:oMathPara>`` for display equations."""
    omath = get_oMath(src)
    if omath is None:
        return None
    nsmap = {"m": OMML_NS}
    para = etree.SubElement(etree.Element("dummy", nsmap=nsmap), f"{{{OMML_NS}}}oMathPara")
    para.append(omath)
    return para


# Backwards-compatible names the renderer already imports.
def latex_to_omml(src: str) -> etree._Element | None:
    return get_oMath(src)


def latex_to_omath_para(src: str) -> etree._Element | None:
    return get_oMathPara(src)
