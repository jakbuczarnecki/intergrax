from __future__ import annotations

from pathlib import Path

import pytest
from scripts.demo.reader import read_lines

pytestmark = pytest.mark.unit


def test_read_lines_returns_normalized_nonempty(tmp_path: Path) -> None:
    src = tmp_path / "sample.txt"
    src.write_text("  Hello  \n\n  WORLD  \n   \nFOO\tbar\n", encoding="utf-8")
    assert read_lines(src) == ["hello", "world", "foo bar"]


def test_read_lines_empty_file(tmp_path: Path) -> None:
    src = tmp_path / "empty.txt"
    src.write_text("", encoding="utf-8")
    assert read_lines(src) == []


def test_read_lines_whitespace_only_lines_skipped(tmp_path: Path) -> None:
    src = tmp_path / "blank.txt"
    src.write_text("  \n\t\n", encoding="utf-8")
    assert read_lines(src) == []
