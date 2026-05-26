"""GOST 7.32-2017 docx styles: margins, fonts, headings, captions.

The ``set_cyrillic_font`` helper is the non-obvious bit: python-docx writes
only ``w:ascii`` and ``w:hAnsi`` for ``font.name``, so Mac Word silently
substitutes Calibri for Cyrillic glyphs. We must also set ``w:eastAsia`` and
``w:cs`` to pin Times New Roman across all script ranges.
"""

from __future__ import annotations

from docx.document import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt
from docx.styles.style import BaseStyle

FONT_NAME = "Times New Roman"
BODY_SIZE_PT = 14
LINE_SPACING = WD_LINE_SPACING.ONE_POINT_FIVE
FIRST_LINE_INDENT_CM = 1.25

MARGIN_LEFT_MM = 30
MARGIN_RIGHT_MM = 15
MARGIN_TOP_MM = 20
MARGIN_BOTTOM_MM = 20


def set_cyrillic_font(style: BaseStyle, name: str = FONT_NAME) -> None:
    """Pin font across ascii / hAnsi / eastAsia / cs ranges.

    Why: python-docx ``font.name`` only writes ascii+hAnsi; on Mac Word
    Cyrillic falls through to the eastAsia/cs slot, which defaults to Calibri.
    """
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(attr), name)


def _ensure_paragraph_style(doc: Document, name: str) -> BaseStyle:
    if name in doc.styles:
        return doc.styles[name]
    return doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)


def _set_outline_level(style: BaseStyle, level: int) -> None:
    """Write ``<w:pPr><w:outlineLvl w:val="N"/></w:pPr>`` into the style.

    Word's TOC field collects paragraphs by outline level (0-based) in
    addition to built-in heading styles, so custom styles need this set
    explicitly to appear in the TOC.
    """
    style_el = style.element
    ppr = style_el.find(qn("w:pPr"))
    if ppr is None:
        ppr = OxmlElement("w:pPr")
        style_el.append(ppr)
    existing = ppr.find(qn("w:outlineLvl"))
    if existing is not None:
        ppr.remove(existing)
    lvl = OxmlElement("w:outlineLvl")
    lvl.set(qn("w:val"), str(level))
    ppr.append(lvl)


def apply_gost_styles(doc: Document) -> None:
    """Configure margins, the Normal style, headings, and the GOST custom styles."""
    section = doc.sections[0]
    section.left_margin = Mm(MARGIN_LEFT_MM)
    section.right_margin = Mm(MARGIN_RIGHT_MM)
    section.top_margin = Mm(MARGIN_TOP_MM)
    section.bottom_margin = Mm(MARGIN_BOTTOM_MM)

    normal = doc.styles["Normal"]
    set_cyrillic_font(normal)
    normal.font.size = Pt(BODY_SIZE_PT)
    pf = normal.paragraph_format
    pf.line_spacing_rule = LINE_SPACING
    pf.first_line_indent = Cm(FIRST_LINE_INDENT_CM)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)

    for lvl, size in ((1, 16), (2, 14), (3, 14)):
        h = doc.styles[f"Heading {lvl}"]
        set_cyrillic_font(h)
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = None
        hpf = h.paragraph_format
        hpf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        hpf.first_line_indent = Cm(FIRST_LINE_INDENT_CM)
        hpf.line_spacing_rule = LINE_SPACING
        hpf.space_before = Pt(12)
        hpf.space_after = Pt(12)
        hpf.keep_with_next = True
        if lvl == 1:
            hpf.page_break_before = True

    structural = _ensure_paragraph_style(doc, "GostStructuralHeading")
    set_cyrillic_font(structural)
    structural.font.size = Pt(BODY_SIZE_PT)
    structural.font.bold = True
    structural.font.all_caps = True
    spf = structural.paragraph_format
    spf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spf.first_line_indent = Cm(0)
    spf.line_spacing_rule = LINE_SPACING
    spf.page_break_before = True
    spf.space_after = Pt(18)
    spf.keep_with_next = True
    # Outline level 0 = Heading 1 in TOC. Without this, the TOC field
    # (\o "1-3") ignores this style and ВВЕДЕНИЕ / ЗАКЛЮЧЕНИЕ /
    # СПИСОК ЛИТЕРАТУРЫ / ПРИЛОЖЕНИЕ never appear in the table of contents.
    _set_outline_level(structural, 0)

    fig = _ensure_paragraph_style(doc, "GostFigureCaption")
    set_cyrillic_font(fig)
    fig.font.size = Pt(BODY_SIZE_PT)
    fpf = fig.paragraph_format
    fpf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fpf.first_line_indent = Cm(0)
    fpf.line_spacing_rule = LINE_SPACING
    fpf.space_before = Pt(6)
    fpf.space_after = Pt(12)

    tbl = _ensure_paragraph_style(doc, "GostTableCaption")
    set_cyrillic_font(tbl)
    tbl.font.size = Pt(BODY_SIZE_PT)
    tpf = tbl.paragraph_format
    tpf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    tpf.first_line_indent = Cm(0)
    tpf.line_spacing_rule = LINE_SPACING
    tpf.space_before = Pt(12)
    tpf.space_after = Pt(6)
    tpf.keep_with_next = True

    bib = _ensure_paragraph_style(doc, "GostBibEntry")
    set_cyrillic_font(bib)
    bib.font.size = Pt(BODY_SIZE_PT)
    bpf = bib.paragraph_format
    bpf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    bpf.first_line_indent = Cm(0)
    bpf.left_indent = Cm(FIRST_LINE_INDENT_CM)
    bpf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    bpf.space_after = Pt(4)

    enum = _ensure_paragraph_style(doc, "GostEnumeration")
    set_cyrillic_font(enum)
    enum.font.size = Pt(BODY_SIZE_PT)
    epf = enum.paragraph_format
    epf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    epf.first_line_indent = Cm(FIRST_LINE_INDENT_CM)
    epf.line_spacing_rule = LINE_SPACING
