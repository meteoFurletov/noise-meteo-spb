"""Unit tests for the pure-function pieces of src.thesisgen."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.thesisgen import bibliography, citations, figures, renderer, typography

NBSP = "\xa0"


class TestTypography:
    def test_outer_quotes_become_yolochki(self):
        out = typography.normalize('Он сказал "привет" мне.')
        assert "«привет»" in out

    def test_double_hyphen_becomes_em_dash(self):
        out = typography.normalize("Объект -- это система.")
        assert "—" in out
        assert "--" not in out

    def test_numeric_range_uses_en_dash(self):
        out = typography.normalize("в 2014-2024 годах")
        assert "2014–2024" in out

    def test_standard_code_hyphen_is_preserved(self):
        out = typography.normalize("ГОСТ 31295.2-2005")
        assert "31295.2-2005" in out

    def test_initials_get_nbsp(self):
        out = typography.normalize("А. С. Пушкин был поэтом.")
        assert f"А.{NBSP}С.{NBSP}Пушкин" in out

    def test_number_unit_gets_nbsp(self):
        out = typography.normalize("масса 5 кг")
        assert f"5{NBSP}кг" in out

    def test_page_reference_gets_nbsp(self):
        out = typography.normalize("см. с. 42 источника")
        assert f"с.{NBSP}42" in out


class TestCitations:
    def test_appearance_order_assigns_sequential_numbers(self):
        sources = {"a": {}, "b": {}, "c": {}}
        bodies = [("f1.md", "text [@b] more [@a]"), ("f2.md", "again [@a] and [@c]")]
        m = citations.build_citation_map(bodies, sources, "appearance")
        assert m == {"b": 1, "a": 2, "c": 3}

    def test_rewrite_simple_and_page_form(self):
        m = {"foo": 5}
        out = citations.rewrite("see [@foo] and [@foo, с. 12]", m)
        assert "[5]" in out
        assert "[5, с. 12]" in out

    def test_unresolved_key_raises_with_line_numbers(self):
        with pytest.raises(citations.UnresolvedCitationError) as exc:
            citations.build_citation_map([("bad.md", "line one\nline two [@missing]")], {}, "appearance")
        assert exc.value.hits[0].line == 2
        assert exc.value.hits[0].file == "bad.md"
        assert exc.value.hits[0].key == "missing"

    def test_prose_citation_finder_detects_author_year(self):
        hits = citations.find_prose_citations(
            "x.md", "Salomons [Salomons, 2001] и [Kephalopoulos et al., 2012]"
        )
        snippets = [h.snippet for h in hits]
        assert "[Salomons, 2001]" in snippets
        assert "[Kephalopoulos et al., 2012]" in snippets

    def test_prose_citation_finder_detects_standards(self):
        hits = citations.find_prose_citations("x.md", "См. [ГОСТ 31295.2-2005] и [ISO 9613-2:1996].")
        snippets = [h.snippet for h in hits]
        assert "[ГОСТ 31295.2-2005]" in snippets
        assert "[ISO 9613-2:1996]" in snippets

    def test_prose_citation_finder_ignores_numbered_and_at_cites(self):
        hits = citations.find_prose_citations(
            "x.md", "Refs [@iso-9613-2-1996], [12], [1, с. 42]"
        )
        assert hits == []


class TestBibliography:
    def test_book_renders_with_isbn_and_content_type(self):
        entry = {
            "type": "book",
            "author": "Иванов И. И.",
            "title": "Основы акустики",
            "responsibility": "И. И. Иванов",
            "place": "Москва",
            "publisher": "Наука",
            "year": 2020,
            "pages": 320,
            "isbn": "978-5-02-000000-0",
            "content_type": "Текст",
            "access": "непосредственный",
        }
        out = bibliography.render_entry(entry)
        assert "Иванов И. И." in out
        assert "Москва : Наука, 2020" in out
        assert "320 с." in out
        assert "ISBN 978-5-02-000000-0" in out
        assert "Текст : непосредственный" in out
        assert out.endswith(".")

    def test_standard_renders_code_and_imprint(self):
        entry = {
            "type": "standard",
            "code": "ГОСТ 31295.2-2005",
            "title": "Шум. Затухание звука",
            "place": "Москва",
            "publisher": "Стандартинформ",
            "year": 2006,
            "pages": 40,
            "content_type": "Текст",
            "access": "непосредственный",
        }
        out = bibliography.render_entry(entry)
        assert out.startswith("ГОСТ 31295.2-2005.")
        assert "Москва : Стандартинформ, 2006" in out

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            bibliography.render_entry({"type": "blog-post"})


class TestHeadingNormalizer:
    def test_strips_section_symbol(self):
        assert renderer.normalize_heading_text("§3.1 — Стратегия") == "3.1 Стратегия"

    def test_strips_dash_after_number(self):
        assert renderer.normalize_heading_text("3.1 — Стратегия") == "3.1 Стратегия"

    def test_passes_through_simple_heading(self):
        assert renderer.normalize_heading_text("1.1 Транспортный шум") == "1.1 Транспортный шум"

    def test_drops_chapter_title(self):
        assert renderer.is_chapter_title("Глава 4. Программная реализация")
        assert renderer.is_chapter_title("ГЛАВА 4. Что-то")
        assert not renderer.is_chapter_title("4.1 Архитектура")


class TestFigureRegistry:
    def test_load_registry_validates_paths(self, tmp_path):
        png = tmp_path / "figures" / "a.png"
        png.parent.mkdir()
        png.write_bytes(b"\x89PNG\r\n\x1a\n")
        registry_yaml = tmp_path / "figures.yaml"
        registry_yaml.write_text(
            yaml.safe_dump({"fig_a": {"path": "figures/a.png", "caption": "Test"}})
        )
        reg = figures.load_registry(registry_yaml, tmp_path)
        assert "fig_a" in reg
        assert reg.get("fig_a").caption == "Test"

    def test_rejects_pdf_path(self, tmp_path):
        pdf = tmp_path / "figures" / "a.pdf"
        pdf.parent.mkdir()
        pdf.write_bytes(b"%PDF-")
        registry_yaml = tmp_path / "figures.yaml"
        registry_yaml.write_text(
            yaml.safe_dump({"fig_a": {"path": "figures/a.pdf", "caption": "Test"}})
        )
        with pytest.raises(figures.FigureError) as exc:
            figures.load_registry(registry_yaml, tmp_path)
        assert ".png" in str(exc.value)

    def test_validate_image_path_rejects_pdf(self):
        with pytest.raises(figures.FigureError):
            figures.validate_image_path(Path("docs/figures/x.pdf"))

    def test_validate_image_path_accepts_png(self):
        figures.validate_image_path(Path("docs/figures/x.png"))  # no raise
