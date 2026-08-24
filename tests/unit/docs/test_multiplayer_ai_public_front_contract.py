# © Artur Czarnecki. All rights reserved.

"""PROMO-P0-3B3: Multiplayer AI public feature hub front."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "README.md"
DOC_PATH = (
    REPO_ROOT / "docs" / "project" / "capabilities" / "architecture" / "MULTIPLAYER_AI.md"
)

_PUBLIC_FRONT_BOUNDARY = re.compile(
    r"^## Engineering canon\s*$",
    re.MULTILINE,
)

_FORBIDDEN_PUBLIC_HEADLINES = (
    "Cursor read scope",
    "Do not read this entire file in one session",
    "READY_FOR_REVIEW",
    "**Current active task:**",
    "**Next task after",
    "**Feature plan (1:1):**",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _public_front(text: str) -> str:
    match = _PUBLIC_FRONT_BOUNDARY.search(text)
    if match is None:
        raise AssertionError("Missing ## Engineering canon section")
    return text[: match.start()]


def test_readme_links_canonical_architecture_doc() -> None:
    readme_text = _read(README_PATH)
    assert DOC_PATH.is_file()
    assert (
        "docs/project/capabilities/architecture/MULTIPLAYER_AI.md" in readme_text
    )
    assert "Multiplayer AI" in readme_text


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped
    raise AssertionError("Missing H1 heading")


def test_public_front_structure() -> None:
    text = _read(DOC_PATH)
    front = _public_front(text)

    assert (
        _first_heading(text)
        == "# Multiplayer AI — Multi-layer Feature Architecture"
    )
    assert "## Why it matters" in front
    assert "## Current reality / maturity boundary" in front
    assert "## At a glance" in front
    assert "## Core mental model" in front
    assert "Principal" in front
    assert "Decision ≠ HITL" in front or "Decision / Approval" in front


def test_public_front_excludes_maintainer_headlines() -> None:
    front = _public_front(_read(DOC_PATH))
    for phrase in _FORBIDDEN_PUBLIC_HEADLINES:
        assert phrase not in front, f"Public front contains maintainer headline: {phrase!r}"


def test_maintainer_history_moved_below_public_front() -> None:
    text = _read(DOC_PATH)
    front = _public_front(text)
    assert "## Engineering canon" not in front
    assert "## Engineering canon" in text
    assert "**Feature plan (1:1):**" in text
    assert "MP-1" in text
    assert "**Current active task:**" in text
