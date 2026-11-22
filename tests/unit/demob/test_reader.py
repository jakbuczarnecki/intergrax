# © Artur Czarnecki. All rights reserved.

from pathlib import Path

from scripts.demob.reader import read_lines


def test_read_lines_normalizes_and_skips_blank(tmp_path: Path) -> None:
    src = tmp_path / "input.txt"
    src.write_text("  Hello   World\n\n  \t  \nFOO\tbar\n", encoding="utf-8")
    assert read_lines(src) == ["hello world", "foo bar"]


def test_read_lines_empty_file(tmp_path: Path) -> None:
    src = tmp_path / "empty.txt"
    src.write_text("", encoding="utf-8")
    assert read_lines(src) == []


def test_read_lines_accepts_str_path(tmp_path: Path) -> None:
    src = tmp_path / "one.txt"
    src.write_text("  Line One  \n", encoding="utf-8")
    assert read_lines(str(src)) == ["line one"]
