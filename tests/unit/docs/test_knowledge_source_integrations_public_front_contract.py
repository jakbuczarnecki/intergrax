# © Artur Czarnecki. All rights reserved.

"""PROMO-P0-3B1: Knowledge Source Integrations public domain hub front."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "README.md"
DOC_PATH = REPO_ROOT / "docs" / "project" / "architecture" / "KNOWLEDGE_SOURCE_INTEGRATIONS.md"

_PUBLIC_FRONT_BOUNDARY = re.compile(
    r"^## 2\. Binding architectural decisions\s*$",
    re.MULTILINE,
)

_FORBIDDEN_PUBLIC_HEADLINES = (
    "Cursor read scope",
    "Branch: development",
    "**Branch:**",
    "**Status:** `CORRECTED / READY_FOR_REVIEW`",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _public_front(text: str) -> str:
    match = _PUBLIC_FRONT_BOUNDARY.search(text)
    if match is None:
        raise AssertionError("Missing ## 2. Binding architectural decisions section")
    return text[: match.start()]


def test_readme_links_canonical_architecture_doc() -> None:
    readme_text = _read(README_PATH)
    assert DOC_PATH.is_file()
    assert "docs/project/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md" in readme_text
    assert "Knowledge Source Integrations" in readme_text


def test_public_front_structure() -> None:
    text = _read(DOC_PATH)
    front = _public_front(text)

    assert text.splitlines()[0].strip() == "# Knowledge Source Integrations"
    assert "## Why it matters" in front
    assert "## Current reality / maturity boundary" in front
    assert "## At a glance" in front
    assert "## Core mental model" in front
    assert "MODE 1 — INDEXED RAG" in front
    assert "MODE 2 — DURABLE MATERIALIZATION" in front
    assert "MODE 3 — LIVE ACCESS" in front


def test_public_front_excludes_maintainer_headlines() -> None:
    front = _public_front(_read(DOC_PATH))
    for phrase in _FORBIDDEN_PUBLIC_HEADLINES:
        assert phrase not in front, f"Public front contains maintainer headline: {phrase!r}"


def test_maintainer_history_moved_below_public_front() -> None:
    text = _read(DOC_PATH)
    front = _public_front(text)
    assert "## Implementation / architecture history" not in front
    assert "## Implementation / architecture history" in text
    assert "VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1" in text
    assert "Correction summary" in text
