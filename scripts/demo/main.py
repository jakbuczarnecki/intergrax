"""Copy normalized text lines between files for the factory demo pipeline."""

from __future__ import annotations

from pathlib import Path

from scripts.demo.reader import read_lines
from scripts.demo.writer import write_lines


def copy_normalized(src: str | Path, dst: str | Path) -> int:
    """Read normalized lines from src, write them to dst; return the line count."""
    lines = read_lines(src)
    write_lines(dst, lines)
    return len(lines)
