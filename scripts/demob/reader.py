# © Artur Czarnecki. All rights reserved.

"""Line-oriented file reading for the factory E2E demo."""

from pathlib import Path

from scripts.demob.helper import normalize


def read_lines(path: str | Path) -> list[str]:
    """Return normalized, non-empty lines from a UTF-8 text file."""
    text = Path(path).read_text(encoding="utf-8")
    result: list[str] = []
    for line in text.splitlines():
        normalized = normalize(line)
        if normalized:
            result.append(normalized)
    return result
