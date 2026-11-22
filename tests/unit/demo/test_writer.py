from __future__ import annotations

from pathlib import Path

import pytest
from scripts.demo.writer import write_lines

pytestmark = pytest.mark.unit


def test_write_lines_writes_utf8_with_trailing_newlines(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    write_lines(target, ["alpha", "beta", "żółć"])

    assert target.read_text(encoding="utf-8") == "alpha\nbeta\nżółć\n"
    assert target.read_bytes() == "alpha\nbeta\nżółć\n".encode()


def test_write_lines_empty_sequence_creates_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    write_lines(target, [])

    assert target.read_text(encoding="utf-8") == ""
    assert target.is_file()
