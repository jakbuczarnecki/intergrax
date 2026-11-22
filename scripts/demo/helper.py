"""Text normalization for the factory demo pipeline."""

from __future__ import annotations


def normalize(text: str) -> str:
    """Strip ends, collapse runs of whitespace to one space, lowercase."""
    return " ".join(text.split()).lower()
