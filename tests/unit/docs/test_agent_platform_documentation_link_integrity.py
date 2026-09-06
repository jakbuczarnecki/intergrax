# © Artur Czarnecki. All rights reserved.

"""Bounded link integrity gate for Agent Platform enterprise documentation."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.docs.public_link_integrity import (
    BrokenLocalLink,
    extract_local_refs,
    is_local_ref,
    resolve_local_target,
)

pytestmark = pytest.mark.gate

REPO_ROOT = Path(__file__).resolve().parents[3]

CORE_AGENT_PLATFORM_DOCS: tuple[Path, ...] = (
    REPO_ROOT / "docs/project/architecture/AGENT_DISTRIBUTION.md",
    REPO_ROOT / "docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md",
    REPO_ROOT / "docs/project/maintainers/audits/AGENT_PLATFORM_FINAL_CLOSURE.md",
)

AGENT_CREATION_GUIDE = (
    REPO_ROOT / "docs/project/technical/guides/AGENT_CREATION_GUIDE.md"
)

ADR_AGENT_008 = (
    REPO_ROOT / "docs/project/technical/adr/entries/2026-09-06/ADR-AGENT-008.md"
)

_MALFORMED_PSEUDO_FENCE = "`\text"


def _broken_links_for(doc_path: Path) -> list[BrokenLocalLink]:
    broken: list[BrokenLocalLink] = []
    source = doc_path.relative_to(REPO_ROOT).as_posix()
    text = doc_path.read_text(encoding="utf-8")
    for ref in extract_local_refs(text):
        if not is_local_ref(ref):
            continue
        target = resolve_local_target(doc_path.parent, ref)
        if target is None:
            continue
        if target.suffix.lower() == ".md" and target.is_file():
            continue
        if target.exists():
            continue
        broken.append(BrokenLocalLink(source=source, target=ref))
    return broken


def test_core_agent_platform_documentation_local_links_resolve() -> None:
    missing = [path for path in CORE_AGENT_PLATFORM_DOCS if not path.is_file()]
    assert not missing, f"missing documentation roots: {missing}"
    broken: list[BrokenLocalLink] = []
    for doc_path in CORE_AGENT_PLATFORM_DOCS:
        broken.extend(_broken_links_for(doc_path))
    assert not broken, broken


def test_agent_creation_guide_navigation_hygiene() -> None:
    assert AGENT_CREATION_GUIDE.is_file()
    text = AGENT_CREATION_GUIDE.read_text(encoding="utf-8")
    assert "(.#" not in text, "TOC must use (#anchor) not (.#anchor)"
    assert "intergrax_runtime_architecture.md" not in text
    assert "](../../architecture/" in text or "](../../architecture/" in text.replace(
        "(../../architecture/", "(../../architecture/"
    )


def test_adr_agent_008_markdown_rendering_hygiene() -> None:
    assert ADR_AGENT_008.is_file()
    text = ADR_AGENT_008.read_text(encoding="utf-8")
    for pattern in ("[\n", "[\t", _MALFORMED_PSEUDO_FENCE, "[egistry_", "[eference_"):
        assert pattern not in text, f"malformed markdown pattern: {pattern!r}"
    assert "\traffic_" not in text
    assert "\test_" not in text
