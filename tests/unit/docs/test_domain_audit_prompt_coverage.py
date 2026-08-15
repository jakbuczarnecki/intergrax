# © Artur Czarnecki. All rights reserved.

"""Canonical domain ids must match generated domain audit prompts."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB = REPO_ROOT / "docs" / "project" / "architecture" / "intergrax_runtime_architecture.md"
AUDIT = REPO_ROOT / "docs" / "project" / "maintainers" / "audit"
GENERATOR = REPO_ROOT / "scripts" / "audit" / "generate_domain_audit_prompts.py"
COMMON = REPO_ROOT / "scripts" / "audit" / "architecture_audit_common.py"

_HUB_DOMAIN_ROW = re.compile(
    r"^\| \d+ \| `([A-Z][A-Z0-9_]+)` \| \[`\1\.md`\]",
    re.MULTILINE,
)
_GENERATOR_DOMAIN_ID = re.compile(r'^\s+"id": "([A-Z][A-Z0-9_]+)",', re.MULTILINE)


def _canonical_domain_ids() -> list[str]:
    return _HUB_DOMAIN_ROW.findall(HUB.read_text(encoding="utf-8"))


def _generator_domain_ids() -> list[str]:
    return _GENERATOR_DOMAIN_ID.findall(GENERATOR.read_text(encoding="utf-8"))


def _generated_audit_prompt_ids(canonical: set[str]) -> set[str]:
    return {path.stem for path in AUDIT.glob("*.md") if path.stem in canonical}


def test_canonical_domain_ids_match_generated_audit_prompts() -> None:
    canonical = _canonical_domain_ids()
    canonical_set = set(canonical)
    assert canonical, "hub domain index is empty"
    assert len(canonical) == len(canonical_set), f"duplicate hub domain ids: {canonical}"

    generated = _generated_audit_prompt_ids(canonical_set)
    missing = sorted(canonical_set - generated)
    extra = sorted(generated - canonical_set)
    assert not missing, f"canonical domains missing generated audit prompts: {missing}"
    assert not extra, f"generated audit prompts without canonical hub ids: {extra}"
    assert generated == canonical_set


def test_generator_registry_matches_canonical_domain_ids() -> None:
    canonical = set(_canonical_domain_ids())
    generator_ids = set(_generator_domain_ids())
    assert generator_ids == canonical, (
        f"generator DOMAINS != hub domain ids; "
        f"only_hub={sorted(canonical - generator_ids)}; "
        f"only_generator={sorted(generator_ids - canonical)}"
    )


_DOMAIN_ORDER_IDS = re.compile(r'^\s+"([A-Z][A-Z0-9_]+)",?\s*$', re.MULTILINE)


def _audit_domain_order() -> list[str]:
    text = COMMON.read_text(encoding="utf-8")
    start = text.find("DOMAIN_ORDER:")
    end = text.find(")", start)
    return _DOMAIN_ORDER_IDS.findall(text[start:end])


def test_audit_domain_order_matches_canonical_hub_order() -> None:
    canonical = _canonical_domain_ids()
    assert _audit_domain_order() == canonical
