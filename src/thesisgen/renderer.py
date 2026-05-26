"""Walk a markdown-it AST and emit docx primitives.

Supports the subset of markdown the thesis uses: headings, paragraphs, bullet
and ordered lists, fenced code, images, tables, inline emphasis. Inline
``$...$`` math is kept as literal text (deferred for v1).

Cross-section concerns handled here:

- **Heading normalization** — section files use mixed conventions (``#``,
  ``##``, sometimes prefixed with ``§`` or ``Глава N``). We drop chapter-title
  headings, strip the ``§`` prefix and the `` — `` separator after
  the section number, and re-level so each section's top heading becomes
  ``Heading 2``.

- **Figure registry token** — the inline token ``[[fig:key]]`` resolves
  against ``figures.yaml`` and inserts an embedded image with caption.

- **Image-format guard** — PDF (or any non-raster) image path raises
  ``FigureError`` with a suggested PNG path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from docx.document import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from markdown_it import MarkdownIt
from markdown_it.token import Token

from src.thesisgen.figures import FigureRegistry, validate_image_path
from src.thesisgen.math import MathFailure, latex_to_omath_para, latex_to_omml
from src.thesisgen.typography import normalize

_FIG_TOKEN_RE = re.compile(r"\[\[fig:([A-Za-z0-9_\-]+)\]\]")
_CHAPTER_HEADING_RE = re.compile(r"^\s*Глава\s+\d+", re.IGNORECASE)
_SECTION_PREFIX_RE = re.compile(r"^\s*§\s*")
_NUMBER_DASH_RE = re.compile(r"^(\d+(?:\.\d+)*)\s*[—–-]\s*")
# $...$ — inline; $$...$$ — display (matched separately, longer first)
_MATH_DISPLAY_RE = re.compile(r"\$\$([\s\S]+?)\$\$")
_MATH_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)")


# CommonMark-escapable ASCII punctuation. Any backslash-escape `\X` for X in
# this set is unescaped by markdown-it. Math content has to survive that pass,
# so we proactively escape every occurrence so the source pandoc finally sees
# is byte-identical to what was in the markdown.
_ESC_CHARS = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _shield_math_content(content: str) -> str:
    """Backslash-escape CommonMark specials inside math content.

    Rules per character:
    - ``\\X`` where X ∈ _ESC_CHARS → emit ``\\\\X``. markdown-it then unescapes
      ``\\\\`` → ``\\`` and leaves X literal, so pandoc receives ``\\X``.
      Handles LaTeX ``\\\\``, ``\\,``, ``\\[``, ``\\{``, ``\\_``, etc.
    - bare X (X ∈ _ESC_CHARS, no leading backslash) → emit ``\\X``. markdown-it
      unescapes back to X, but the escape blocks emphasis / link / etc. parsing
      while the source goes through markdown-it. Handles raw ``_``, ``*``,
      ``[``, ``]`` inside math.
    - anything else → verbatim.

    Without this, e.g. ``$$ ... \\\\[4pt] ... $$`` becomes ``\\[4pt]`` then
    ``[4pt]`` (markdown eats two layers of escape), and pandoc gets broken
    LaTeX that it silently renders as garbled text.
    """
    out: list[str] = []
    i = 0
    n = len(content)
    while i < n:
        c = content[i]
        if c == "\\" and i + 1 < n and content[i + 1] in _ESC_CHARS:
            out.append("\\\\")
            out.append(content[i + 1])
            i += 2
        elif c in _ESC_CHARS:
            out.append("\\")
            out.append(c)
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _preprocess_math(text: str) -> str:
    """Make math spans survive markdown-it: shield specials inside the content,
    and surround ``$$..$$`` blocks with blank lines so they're parsed as
    standalone display-math paragraphs even when the source has them inline
    with prose. Without the blank lines, ``$$..$$`` inside a paragraph fails
    ``_is_display_math_only`` and is then excluded from inline-math matching
    via the ``(?<!\\$)\\$(?!\\$)`` lookarounds — so the formula is emitted as
    literal text.
    """
    def _disp(m: re.Match[str]) -> str:
        return f"\n\n$${_shield_math_content(m.group(1))}$$\n\n"
    text = _MATH_DISPLAY_RE.sub(_disp, text)

    def _inl(m: re.Match[str]) -> str:
        return f"${_shield_math_content(m.group(1))}$"
    text = _MATH_INLINE_RE.sub(_inl, text)
    return text


@dataclass
class RenderContext:
    """Per-chapter rendering state."""

    figure_n: int = 0
    table_n: int = 0
    prefix: str = ""
    assets_root: Path = field(default_factory=Path)
    figures: FigureRegistry = field(default_factory=lambda: FigureRegistry({}))
    math_failures: list[MathFailure] = field(default_factory=list)
    current_file: str = ""

    def fig_label(self) -> str:
        self.figure_n += 1
        return f"{self.prefix}{self.figure_n}" if self.prefix else str(self.figure_n)

    def tbl_label(self) -> str:
        self.table_n += 1
        return f"{self.prefix}{self.table_n}" if self.prefix else str(self.table_n)

    def record_math_failure(self, source: str, line: int = 0) -> None:
        self.math_failures.append(
            MathFailure(file=self.current_file, line=line, source=source, error="conversion failed")
        )


def parse(md_text: str) -> list[Token]:
    md_text = _preprocess_math(md_text)
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    return md.parse(md_text)


def collect_math_fragments(tokens: list[Token]) -> list[str]:
    """Walk a parsed AST and return every math source ($...$ and $$...$$).

    Crucially, this sees text *after* markdown-it's CommonMark backslash
    unescaping (so ``7\\,560`` in markdown source appears here as ``7,560``).
    The renderer scans the same post-parse text, so this matches what the
    cache lookup will actually receive.
    """
    found: list[str] = []
    for i, t in enumerate(tokens):
        if t.type == "inline" and t.children:
            text = "".join(c.content for c in t.children if c.type == "text")
            full_text = "".join(c.content for c in t.children if c.type in ("text", "code_inline"))
            # Display math: paragraph that is just $$...$$
            stripped = text.strip()
            if stripped.startswith("$$") and stripped.endswith("$$"):
                inner = stripped[2:-2]
                if "$$" not in inner:
                    found.append(inner)
                    continue
            # Inline math anywhere in the text
            for c in t.children:
                if c.type != "text":
                    continue
                for m in _MATH_INLINE_RE.finditer(c.content):
                    found.append(m.group(1))
    return found


def normalize_heading_text(text: str) -> str:
    text = _SECTION_PREFIX_RE.sub("", text)
    text = _NUMBER_DASH_RE.sub(lambda m: f"{m.group(1)} ", text)
    return text.strip()


def is_chapter_title(text: str) -> bool:
    """Headings that name an entire chapter (`Глава 4. ...`) must be skipped:
    the chapter title comes from the manifest, not the section file."""
    return bool(_CHAPTER_HEADING_RE.match(text))


def _heading_level_offset(tokens: list[Token]) -> int:
    """Compute how much to add to every heading level so the minimum maps to 2.

    Sections that lead with ``#`` get +1; sections that lead with ``##`` get 0.
    """
    min_level = None
    for i, t in enumerate(tokens):
        if t.type == "heading_open":
            text = _inline_text(tokens[i + 1])
            if is_chapter_title(text):
                continue
            level = int(t.tag[1])
            if min_level is None or level < min_level:
                min_level = level
    if min_level is None:
        return 0
    return max(0, 2 - min_level)


def render(doc: Document, tokens: list[Token], ctx: RenderContext) -> None:
    """Walk tokens and emit into ``doc``."""
    offset = _heading_level_offset(tokens)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.type == "heading_open":
            inline = tokens[i + 1]
            text = _inline_text(inline)
            if is_chapter_title(text):
                i += 3
                continue
            src_level = int(t.tag[1])
            level = min(3, src_level + offset)
            clean = normalize_heading_text(text)
            style = doc.styles[f"Heading {level}"]
            p = doc.add_paragraph(normalize(clean), style=style)
            p.paragraph_format.first_line_indent = doc.styles["Heading 1"].paragraph_format.first_line_indent
            i += 3
        elif t.type == "paragraph_open":
            inline = tokens[i + 1]
            if _is_image_only(inline):
                _emit_image(doc, inline, ctx)
            elif _is_figure_token_only(inline):
                _emit_figure_token(doc, inline, ctx)
            elif _is_display_math_only(inline):
                _emit_display_math(doc, inline, ctx)
            else:
                p = doc.add_paragraph()
                _emit_inline(doc, p, inline, ctx)
            i += 3
        elif t.type == "bullet_list_open":
            i = _emit_list(doc, tokens, i, ctx, ordered=False)
        elif t.type == "ordered_list_open":
            i = _emit_list(doc, tokens, i, ctx, ordered=True)
        elif t.type == "table_open":
            i = _emit_table(doc, tokens, i, ctx)
        elif t.type == "fence":
            p = doc.add_paragraph(t.content.rstrip("\n"))
            p.paragraph_format.first_line_indent = None
            for run in p.runs:
                run.font.name = "Courier New"
            i += 1
        else:
            i += 1


def _inline_text(token: Token) -> str:
    if token.children is None:
        return token.content
    return "".join(c.content for c in token.children if c.type in ("text", "code_inline"))


def _is_image_only(inline: Token) -> bool:
    if not inline.children:
        return False
    images = [c for c in inline.children if c.type == "image"]
    text_nodes = [c for c in inline.children if c.type == "text" and c.content.strip()]
    return len(images) == 1 and not text_nodes


def _is_figure_token_only(inline: Token) -> bool:
    if not inline.children:
        return False
    text = "".join(c.content for c in inline.children if c.type == "text").strip()
    return bool(text) and bool(_FIG_TOKEN_RE.fullmatch(text))


def _is_display_math_only(inline: Token) -> bool:
    """A paragraph that contains only $$...$$ (optionally with trailing punctuation)."""
    if not inline.children:
        return False
    text = "".join(c.content for c in inline.children if c.type == "text").strip()
    if not text.startswith("$$") or not text.endswith("$$"):
        return False
    inner = text[2:-2]
    return "$$" not in inner


def _emit_display_math(doc: Document, inline: Token, ctx: RenderContext) -> None:
    text = "".join(c.content for c in inline.children if c.type == "text").strip()
    inner = text[2:-2]
    omath_para = latex_to_omath_para(inner)
    p = doc.add_paragraph()
    p.alignment = 1  # CENTER
    p.paragraph_format.first_line_indent = None
    if omath_para is None:
        ctx.record_math_failure(inner)
        p.add_run(text)
    else:
        p._p.append(omath_para)


def _emit_inline(doc: Document, p, inline: Token, ctx: RenderContext) -> None:
    if not inline.children:
        _emit_text_with_tokens(doc, p, inline.content, ctx)
        return
    bold = italic = False
    for child in inline.children:
        if child.type == "strong_open":
            bold = True
        elif child.type == "strong_close":
            bold = False
        elif child.type == "em_open":
            italic = True
        elif child.type == "em_close":
            italic = False
        elif child.type == "text":
            _emit_text_with_tokens(doc, p, child.content, ctx, bold=bold, italic=italic)
        elif child.type == "code_inline":
            run = p.add_run(child.content)
            run.font.name = "Courier New"
        elif child.type == "softbreak":
            p.add_run(" ")
        elif child.type == "hardbreak":
            p.add_run().add_break()


def _emit_text_with_tokens(doc: Document, p, text: str, ctx: RenderContext, *, bold: bool = False, italic: bool = False) -> None:
    """Add ``text`` as one or more runs, expanding inline ``$math$`` and ``[[fig:key]]`` tokens.

    Math tokens are inlined as native OMML; figure tokens render as a centered
    picture on a new line.
    """
    # Build a combined match list: (start, end, kind, payload)
    spans: list[tuple[int, int, str, str]] = []
    for m in _MATH_INLINE_RE.finditer(text):
        spans.append((m.start(), m.end(), "math", m.group(1)))
    for m in _FIG_TOKEN_RE.finditer(text):
        spans.append((m.start(), m.end(), "fig", m.group(1)))
    spans.sort()

    if not spans:
        run = p.add_run(normalize(text))
        run.bold = bold
        run.italic = italic
        return

    pos = 0
    for start, end, kind, payload in spans:
        if start < pos:
            continue  # overlap, skip
        before = text[pos:start]
        if before:
            run = p.add_run(normalize(before))
            run.bold = bold
            run.italic = italic
        if kind == "math":
            omml = latex_to_omml(payload)
            if omml is None:
                ctx.record_math_failure(payload)
                run = p.add_run(f"${payload}$")
                run.bold = bold
                run.italic = italic
            else:
                p._p.append(omml)
        elif kind == "fig":
            _emit_figure_by_key(doc, payload, ctx)
        pos = end
    after = text[pos:]
    if after:
        run = p.add_run(normalize(after))
        run.bold = bold
        run.italic = italic


def _emit_figure_token(doc: Document, inline: Token, ctx: RenderContext) -> None:
    text = "".join(c.content for c in inline.children if c.type == "text").strip()
    m = _FIG_TOKEN_RE.fullmatch(text)
    assert m is not None
    _emit_figure_by_key(doc, m.group(1), ctx)


def _emit_figure_by_key(doc: Document, key: str, ctx: RenderContext) -> None:
    fig = ctx.figures.get(key)
    _emit_picture(doc, fig.path, fig.caption, ctx)


def _emit_image(doc: Document, inline: Token, ctx: RenderContext) -> None:
    img = next(c for c in inline.children if c.type == "image")
    src = img.attrs.get("src", "")
    caption = normalize("".join(c.content for c in (img.children or []) if c.type == "text"))
    path = Path(src) if Path(src).is_absolute() else (ctx.assets_root / src).resolve()
    validate_image_path(path)
    _emit_picture(doc, path, caption, ctx)


def _emit_picture(doc: Document, path: Path, caption: str, ctx: RenderContext) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = None
    try:
        p.add_run().add_picture(str(path), width=Inches(6))
    except Exception as exc:
        p.add_run(f"[не удалось вставить рисунок {path}: {exc}]")
    label = ctx.fig_label()
    cap = doc.add_paragraph(f"Рисунок {label} — {caption}", style=doc.styles["GostFigureCaption"])
    cap.paragraph_format.first_line_indent = None


def _emit_list(doc: Document, tokens: list[Token], start: int, ctx: RenderContext, *, ordered: bool) -> int:
    closer = "ordered_list_close" if ordered else "bullet_list_close"
    i = start + 1
    item_n = 0
    while i < len(tokens) and tokens[i].type != closer:
        t = tokens[i]
        if t.type == "list_item_open":
            item_n += 1
            j = i + 1
            inlines: list[Token] = []
            while j < len(tokens) and tokens[j].type != "list_item_close":
                if tokens[j].type == "inline":
                    inlines.append(tokens[j])
                j += 1
            marker = f"{item_n})" if ordered else "—"
            p = doc.add_paragraph(style=doc.styles["GostEnumeration"])
            p.add_run(f"{marker} ")
            for k, inline in enumerate(inlines):
                if k > 0:
                    p.add_run(" ")
                _emit_inline(doc, p, inline, ctx)
            i = j + 1
        else:
            i += 1
    return i + 1


def _emit_table(doc: Document, tokens: list[Token], start: int, ctx: RenderContext) -> int:
    # Collect each cell's inline token so we can render math/figures inside it.
    rows: list[list[Token]] = []
    i = start + 1
    current_row: list[Token] = []
    while i < len(tokens) and tokens[i].type != "table_close":
        t = tokens[i]
        if t.type == "tr_open":
            current_row = []
        elif t.type == "tr_close":
            rows.append(current_row)
        elif t.type in ("th_open", "td_open"):
            inline = tokens[i + 1]
            current_row.append(inline)
            i += 3
            continue
        i += 1

    if not rows:
        return i + 1

    label = ctx.tbl_label()
    cap = doc.add_paragraph(f"Таблица {label}", style=doc.styles["GostTableCaption"])
    cap.paragraph_format.first_line_indent = None

    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    for r_idx, row in enumerate(rows):
        for c_idx, cell_inline in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = None
            _emit_inline(doc, p, cell_inline, ctx)
    return i + 1
