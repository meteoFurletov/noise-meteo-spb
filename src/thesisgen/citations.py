"""Citation rewriting plus prose-citation scanning.

Two responsibilities:

1. **Formal citations** — ``[@key]`` / ``[@key, с. 42]`` tokens are mapped to
   numbers (``[N]`` / ``[N, с. 42]``) sequentially in order of appearance
   (default) or alphabetically. Unresolved keys fail the build with a
   file+line checklist.

2. **Prose author-year scan** — finds Russian/English ``[Author, YYYY]``
   style mentions that haven't yet been migrated to ``[@key]`` form so the
   user has a checklist of remaining work. This does NOT fail the build.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

_CITE_RE = re.compile(r"\[@([A-Za-z0-9_\-]+)(?:,\s*([^\];]+))?(?:;\s*@([A-Za-z0-9_\-]+)(?:,\s*([^\];]+))?)*\]")
_CITE_ENTRY_RE = re.compile(r"@([A-Za-z0-9_\-]+)(?:,\s*([^\];]+))?")

# Matches a single prose-style citation segment such as "Salomons, 2001",
# "Kephalopoulos et al., 2012", "ГОСТ 31295.2-2005", "ISO 9613-2:1996",
# "Директива 2002/49/EC". A leading capital letter or digit, then any
# non-bracket payload that ends with a 4-digit year (optionally suffixed).
_PROSE_SEGMENT_RE = re.compile(
    r"(?:[А-ЯA-Z][^\];]*?,\s*\d{4}[a-z]?"  # Author, 2001
    r"|[А-ЯA-Z][A-Za-zА-Яа-я]+\s+\d{4}/\d+/[A-Z]+"  # Директива 2002/49/EC
    r"|(?:ГОСТ|ISO|СП|СНиП|EN|DIN)\s+[\d\.\-:/]+(?:-\d{4})?)"  # ГОСТ 31295.2-2005
)
_BRACKETED_RE = re.compile(r"\[([^\[\]@]+?)\]")

Order = Literal["appearance", "alphabetical"]


@dataclass
class ProseHit:
    file: str
    line: int
    snippet: str


@dataclass
class UnresolvedHit:
    file: str
    line: int
    key: str


class UnresolvedCitationError(RuntimeError):
    def __init__(self, hits: list[UnresolvedHit]) -> None:
        lines = ["Unresolved citation keys (not in sources.yaml):"]
        by_file: dict[str, list[UnresolvedHit]] = {}
        for h in hits:
            by_file.setdefault(h.file, []).append(h)
        for f, group in sorted(by_file.items()):
            lines.append(f"  {f}")
            for h in group:
                lines.append(f"    L{h.line}: [@{h.key}]")
        super().__init__("\n".join(lines))
        self.hits = hits


def build_citation_map(
    sections: Iterable[tuple[str, str]],
    sources: dict[str, dict],
    order: Order = "appearance",
) -> dict[str, int]:
    """Scan ``(filename, body)`` pairs and return ``{key: N}``.

    Raises ``UnresolvedCitationError`` with file+line hits if any cited key
    is missing from ``sources``.
    """
    seen: list[str] = []
    unresolved: list[UnresolvedHit] = []
    for filename, body in sections:
        for line_no, line in enumerate(body.splitlines(), start=1):
            for match in _CITE_RE.finditer(line):
                for entry in _CITE_ENTRY_RE.finditer(match.group(0)):
                    key = entry.group(1)
                    if key not in seen:
                        seen.append(key)
                    if key not in sources:
                        unresolved.append(UnresolvedHit(file=filename, line=line_no, key=key))
    if unresolved:
        raise UnresolvedCitationError(unresolved)

    if order == "alphabetical":
        ordered = sorted(seen, key=lambda k: _sort_key(k, sources[k]))
    else:
        ordered = seen
    return {key: i + 1 for i, key in enumerate(ordered)}


def _sort_key(key: str, entry: dict) -> tuple:
    title = (entry.get("author") or entry.get("title") or key).lower()
    cyr = 0 if title and "а" <= title[0] <= "я" else 1
    return (cyr, title)


def rewrite(text: str, citemap: dict[str, int]) -> str:
    """Replace every ``[@key…]`` token in ``text`` with its assigned number.

    Supports multi-citations ``[@a; @b]`` → ``[1, 2]`` and locators
    ``[@a, с. 5]`` → ``[1, с. 5]``. When any entry carries a locator, the
    join switches to ``; `` to preserve grouping; otherwise ``, `` is used.
    """

    def _sub(m: re.Match[str]) -> str:
        rendered: list[str] = []
        any_locator = False
        for entry in _CITE_ENTRY_RE.finditer(m.group(0)):
            key = entry.group(1)
            suffix = entry.group(2)
            n = citemap[key]
            if suffix:
                any_locator = True
                rendered.append(f"{n}, {suffix.strip()}")
            else:
                rendered.append(str(n))
        sep = "; " if any_locator else ", "
        return f"[{sep.join(rendered)}]"

    return _CITE_RE.sub(_sub, text)


def find_prose_citations(filename: str, body: str) -> list[ProseHit]:
    """Return every prose-style bracketed citation in ``body``.

    A "prose citation" is a ``[...]`` group whose contents look like an
    author/year mention or a regulatory code (ГОСТ, ISO, СП, СНиП, EN, DIN).
    Tokens of the form ``[@key]`` (resolved formal cites) and pure numeric
    ranges (``[12]``, ``[1, с. 42]`` — already rewritten formal cites) are
    not reported.
    """
    hits: list[ProseHit] = []
    for line_no, line in enumerate(body.splitlines(), start=1):
        for m in _BRACKETED_RE.finditer(line):
            payload = m.group(1).strip()
            if not _PROSE_SEGMENT_RE.search(payload):
                continue
            if payload[0].isdigit():
                # Already a numbered ref like "12" or "12, с. 42".
                continue
            hits.append(ProseHit(file=filename, line=line_no, snippet=m.group(0)))
    return hits
