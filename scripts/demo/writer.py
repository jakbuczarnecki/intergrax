"""Write UTF-8 line-oriented text for the factory demo pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def write_lines(path: str | Path, lines: Sequence[str]) -> None:
    """Write each string as one line with a trailing newline, UTF-8."""
    target = Path(path)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(f"{line}\n")
