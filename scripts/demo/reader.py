"""Read and normalize lines from a text file."""

from __future__ import annotations

from pathlib import Path

from scripts.demo.helper import normalize


def read_lines(path: str | Path) -> list[str]:
    """Read UTF-8 text; return normalize()d lines, omitting those that are empty after normalize."""
    text = Path(path).read_text(encoding="utf-8")
    result: list[str] = []
    for line in text.splitlines():
        normalized = normalize(line)
        if normalized:
            result.append(normalized)
    return result
