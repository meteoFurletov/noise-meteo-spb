"""Russian typographic normalization for thesis body text.

Applied as a pre-pass to raw markdown *before* AST parsing, restricted to
text content (callers must not pass URLs, code, or fenced blocks).

Rules implemented (per the reference pack, §22):
- ASCII double quotes → «ёлочки» at the outer level
- ``--`` → em-dash (—) wrapped with NBSP-before / space-after
- digit-hyphen-digit → en-dash (–) with no spaces (ranges)
- NBSP insertion in canonical Russian patterns: initials, units, abbreviations,
  one-letter prepositions
"""

from __future__ import annotations

import re

NBSP = " "
EM_DASH = "—"
EN_DASH = "–"


def _quotes(text: str) -> str:
    """Replace pairs of ASCII " with «...». Naive left-to-right pairing."""
    out: list[str] = []
    open_quote = True
    for ch in text:
        if ch == '"':
            out.append("«" if open_quote else "»")
            open_quote = not open_quote
        else:
            out.append(ch)
    return "".join(out)


def _dashes(text: str) -> str:
    # Only treat a hyphen between numbers as a range en-dash when the whole
    # token is bounded by whitespace/punctuation on the *outside* — otherwise
    # we corrupt identifiers like "ГОСТ 31295.2-2005" or "ISO 9613-2:1996".
    text = re.sub(r"(?<!\S)(\d+)\s*-\s*(\d+)(?!\S)", lambda m: f"{m.group(1)}{EN_DASH}{m.group(2)}", text)
    text = re.sub(r"(?<!-)--(?!-)", EM_DASH, text)
    text = re.sub(rf"\s+{re.escape(EM_DASH)}\s+", f"{NBSP}{EM_DASH} ", text)
    return text


_INITIALS_RE = re.compile(r"\b([А-ЯЁA-Z])\.\s*([А-ЯЁA-Z])\.\s+([А-ЯЁA-Z][а-яёa-z]+)")
_NUM_UNIT_RE = re.compile(
    r"(\d)\s+(кг|г|мг|т|км|м|см|мм|мкм|л|мл|с|мин|ч|сут|год|г\.|°C|%|шт|МВт|кВт|Вт|Гц|кГц|МГц|ГПа|МПа|кПа|Па)\b"
)
_PAGE_NUM_RE = re.compile(r"\b([сc])\.\s+(\d)")
_NUMSIGN_RE = re.compile(r"№\s+(\d)")
_SHORT_ABBR_RE = re.compile(r"\b(т)\.\s+(е|п|д)\.")
_I_DR_RE = re.compile(r"\b(и)\s+(др|т\.\s*д)\.")
_PREPOSITION_RE = re.compile(r"(?<=\s)([вВкКсСуУоОиИ])\s+(?=[А-ЯЁA-Zа-яёa-z])")


def _nbsp(text: str) -> str:
    text = _INITIALS_RE.sub(lambda m: f"{m.group(1)}.{NBSP}{m.group(2)}.{NBSP}{m.group(3)}", text)
    text = _NUM_UNIT_RE.sub(lambda m: f"{m.group(1)}{NBSP}{m.group(2)}", text)
    text = _PAGE_NUM_RE.sub(lambda m: f"{m.group(1)}.{NBSP}{m.group(2)}", text)
    text = _NUMSIGN_RE.sub(lambda m: f"№{NBSP}{m.group(1)}", text)
    text = _SHORT_ABBR_RE.sub(lambda m: f"{m.group(1)}.{NBSP}{m.group(2)}.", text)
    text = _I_DR_RE.sub(lambda m: f"{m.group(1)}{NBSP}{m.group(2)}.", text)
    text = _PREPOSITION_RE.sub(lambda m: f"{m.group(1)}{NBSP}", text)
    return text


def normalize(text: str) -> str:
    """Apply all typography rules in order: quotes → dashes → NBSP."""
    text = _quotes(text)
    text = _dashes(text)
    text = _nbsp(text)
    return text
