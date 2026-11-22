from __future__ import annotations

from pathlib import Path

import pytest
from scripts.demo.main import copy_normalized

pytestmark = pytest.mark.unit


def test_copy_normalized_writes_normalized_lines_and_returns_count(tmp_path: Path) -> None:
    src = tmp_path / "input.txt"
    dst = tmp_path / "output.txt"
    src.write_text("  Hello  \n\n  WORLD  \nFOO\tbar\n", encoding="utf-8")

    count = copy_normalized(src, dst)

    assert count == 3
    assert dst.read_text(encoding="utf-8") == "hello\nworld\nfoo bar\n"


def test_copy_normalized_empty_source_returns_zero(tmp_path: Path) -> None:
    src = tmp_path / "empty.txt"
    dst = tmp_path / "out.txt"
    src.write_text("", encoding="utf-8")

    count = copy_normalized(src, dst)

    assert count == 0
    assert dst.read_text(encoding="utf-8") == ""
