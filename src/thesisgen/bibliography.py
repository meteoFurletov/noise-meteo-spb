"""Render ``sources.yaml`` entries into GOST R 7.0.100-2018 bibliography lines.

One small formatter per entry type. Each composes the GOST areas separated by
``. — `` with prescribed-punctuation spaces (per §17 of the reference pack).

The entry must include ``type``. Unknown types fall back to a permissive
formatter that joins ``author / title / place / publisher / year``.
"""

from __future__ import annotations

from typing import Callable

EM_DASH = "—"
NBSP = " "
SEP = f". {EM_DASH} "


def _join(*parts: str | None) -> str:
    return SEP.join(p for p in parts if p)


def _strip_trailing_dot(s: str) -> str:
    return s[:-1] if s.endswith(".") else s


def _content_type(entry: dict) -> str | None:
    ctype = entry.get("content_type")
    access = entry.get("access")
    if ctype and access:
        return f"{ctype} : {access}"
    return None


def _book(e: dict) -> str:
    author = e.get("author")
    title = e["title"]
    subtitle = f" : {e['subtitle']}" if e.get("subtitle") else ""
    responsibility = e.get("responsibility") or (author if author else None)
    head = f"{_strip_trailing_dot(author)}. {title}{subtitle}" if author else f"{title}{subtitle}"
    if responsibility:
        head = f"{head} / {responsibility}"
    edition = e.get("edition")
    imprint = f"{e['place']} : {e['publisher']}, {e['year']}"
    pages = f"{e['pages']} с." if e.get("pages") else None
    isbn = f"ISBN {e['isbn']}" if e.get("isbn") else None
    return _join(head, edition, imprint, pages, isbn, _content_type(e)) + "."


def _article(e: dict) -> str:
    author = e.get("author")
    title = e["title"]
    responsibility = e.get("responsibility") or author
    head = f"{_strip_trailing_dot(author)}. {title}" if author else title
    if responsibility:
        head = f"{head} / {responsibility}"
    journal = e["journal"]
    year = e["year"]
    issue = f"{NBSP}№{NBSP}{e['issue']}" if e.get("issue") else ""
    volume = f"Т.{NBSP}{e['volume']}" if e.get("volume") else None
    pages = f"С.{NBSP}{e['pages']}" if e.get("pages") else None
    journal_part = f"{journal}. {EM_DASH} {year}.{(' ' + EM_DASH + ' ' + volume) if volume else ''}{(' ' + EM_DASH + issue) if issue else ''}"
    head = f"{head} // {journal_part}"
    return _join(head, pages, _content_type(e)) + "."


def _standard(e: dict) -> str:
    code = e["code"]
    title = e["title"]
    head = f"{code}. {title}"
    imprint_parts = [p for p in (e.get("place"), e.get("publisher")) if p]
    imprint = ", ".join([" : ".join(imprint_parts), str(e["year"])]) if imprint_parts else str(e["year"])
    pages = f"{e['pages']} с." if e.get("pages") else None
    return _join(head, imprint, pages, _content_type(e)) + "."


def _dissertation(e: dict) -> str:
    author = e["author"]
    title = e["title"]
    degree = e.get("degree", "канд. техн. наук")
    specialty = e.get("specialty", "")
    spec_part = f" : {specialty}" if specialty else ""
    head = f"{author}. {title} : дис. ... {degree}{spec_part} / {e.get('responsibility', author)}"
    if e.get("organization"):
        head = f"{head} ; {e['organization']}"
    imprint = f"{e['place']}, {e['year']}"
    pages = f"{e['pages']} с." if e.get("pages") else None
    return _join(head, imprint, pages, _content_type(e)) + "."


def _website(e: dict) -> str:
    author = e.get("author")
    title = e["title"]
    site = e.get("site")
    head = f"{_strip_trailing_dot(author)}. {title}" if author else title
    if site:
        head = f"{head} // {site} : [сайт]"
    year = e.get("year")
    if year:
        head = f"{head}. {EM_DASH} {year}"
    url = e["url"]
    accessed = e.get("accessed", "")
    url_part = f"URL: {url}" + (f" (дата обращения: {accessed})" if accessed else "")
    return _join(head, url_part, _content_type(e) or "Текст : электронный") + "."


def _law(e: dict) -> str:
    jurisdiction = e.get("jurisdiction", "Российская Федерация")
    title = e["title"]
    code = e.get("code", "")
    head = f"{jurisdiction}. Законы. {title}"
    if code:
        head = f"{head} : {code}"
    imprint = f"{e.get('place', 'Москва')} : {e.get('publisher', '')}, {e['year']}".replace(" : ,", ",")
    pages = f"{e['pages']} с." if e.get("pages") else None
    return _join(head, imprint, pages, _content_type(e)) + "."


def _conference(e: dict) -> str:
    author = e.get("author")
    title = e["title"]
    head = f"{_strip_trailing_dot(author)}. {title}" if author else title
    if author:
        head = f"{head} / {author}"
    proceedings = e["proceedings"]
    imprint = f"{proceedings}. {EM_DASH} {e['place']} : {e['publisher']}, {e['year']}"
    pages = f"С.{NBSP}{e['pages']}" if e.get("pages") else None
    head = f"{head} // {imprint}"
    return _join(head, pages, _content_type(e)) + "."


_FORMATTERS: dict[str, Callable[[dict], str]] = {
    "book": _book,
    "article": _article,
    "standard": _standard,
    "dissertation": _dissertation,
    "website": _website,
    "law": _law,
    "conference": _conference,
}


def render_entry(entry: dict) -> str:
    """Render one bibliography entry. Caller prepends ``N. `` numbering."""
    fmt = _FORMATTERS.get(entry.get("type", ""))
    if fmt is None:
        raise ValueError(
            f"Unknown bibliography entry type: {entry.get('type')!r}. "
            f"Expected one of {sorted(_FORMATTERS)}."
        )
    return fmt(entry)
