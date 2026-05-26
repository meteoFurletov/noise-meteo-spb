"""Entry point: build the thesis .docx from a thesis_doc/ tree.

Usage:
    python scripts/thesis_build.py thesis_doc/
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from src.thesisgen import build_thesis


def _print_prose_checklist(hits) -> None:
    if not hits:
        print("✓ Прозовых ссылок [Author, YYYY] не найдено — всё в [@key].")
        return
    grouped: dict[str, list] = defaultdict(list)
    for h in hits:
        grouped[h.file].append(h)
    print(f"\n⚠ Прозовые ссылки, не сконвертированные в [@key] ({len(hits)} шт.):")
    for filename in sorted(grouped):
        print(f"  {filename}")
        for h in grouped[filename]:
            print(f"    L{h.line}: {h.snippet}")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/thesis_build.py <thesis_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2
    report = build_thesis(root)
    print(f"Wrote {report.out_path}")
    print(
        f"Рисунков: {report.n_figures}, таблиц: {report.n_tables}, "
        f"источников: {report.n_sources}, формул: {report.n_math} "
        f"(сконвертировано {report.n_math - len(report.math_failures)}, ошибок {len(report.math_failures)})."
    )
    _print_prose_checklist(report.prose_hits)
    if report.math_failures:
        print(f"\n⚠ Формулы, которые не удалось сконвертировать ({len(report.math_failures)} шт.):")
        by_file: dict[str, list] = {}
        for mf in report.math_failures:
            by_file.setdefault(mf.file, []).append(mf)
        for filename in sorted(by_file):
            print(f"  {filename}")
            for mf in by_file[filename][:10]:
                print(f"    ${mf.source[:80]}$")
            if len(by_file[filename]) > 10:
                print(f"    ... и ещё {len(by_file[filename]) - 10} формул")
    print("\nOpen in Word, press Ctrl+A then F9 once to populate TOC and page numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
