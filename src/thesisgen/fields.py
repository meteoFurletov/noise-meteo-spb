"""Word field helpers: auto-updating TOC and page-number fields.

python-docx has no native support for either; we build the underlying
``w:fldChar`` / ``w:instrText`` OOXML directly. After opening the file, the
user must press ``Ctrl+A`` then ``F9`` once to populate the fields.
"""

from __future__ import annotations

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph


def _field(run, instr: str, placeholder: str | None = None) -> None:
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    if placeholder is not None:
        t = OxmlElement("w:t")
        t.text = placeholder
        separate.append(t)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr_el, separate, end):
        run._r.append(el)


def insert_toc(paragraph: Paragraph) -> None:
    """Insert a ``TOC`` field covering Heading 1-3 plus GostStructuralHeading.

    The ``\\t "GostStructuralHeading,1"`` switch makes Word include our
    custom structural style (ВВЕДЕНИЕ, ЗАКЛЮЧЕНИЕ, СПИСОК ЛИТЕРАТУРЫ,
    ПРИЛОЖЕНИЕ) at TOC level 1 — outline-level inheritance via style alone
    isn't reliable across Word versions.
    """
    run = paragraph.add_run()
    _field(
        run,
        r'TOC \o "1-3" \h \z \u \t "GostStructuralHeading,1"',
        placeholder="Обновите поле (Ctrl+A, F9)",
    )


def insert_page_number(paragraph: Paragraph) -> None:
    """Insert a ``PAGE`` field into the paragraph."""
    run = paragraph.add_run()
    _field(run, "PAGE", placeholder="#")


def enable_field_auto_update(doc) -> None:
    """Set ``<w:updateFields w:val="true"/>`` in settings.xml.

    With this flag, Word prompts to update fields on open and populates the
    TOC + page numbers without the user needing to press Ctrl+A, F9.
    """
    settings = doc.settings.element
    existing = settings.find(qn("w:updateFields"))
    if existing is None:
        el = OxmlElement("w:updateFields")
        el.set(qn("w:val"), "true")
        settings.append(el)
    else:
        existing.set(qn("w:val"), "true")
