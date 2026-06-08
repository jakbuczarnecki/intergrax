# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog, list_catalog_skill_ids

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_PROVIDERS = REPO_ROOT / "intergrax" / "skills" / "providers"

_REQUIRED_SECTIONS = (
    "## Purpose",
    "## How it works",
    "## How to use",
    "## What you get",
    "## Tools unlocked",
)


@pytest.fixture(autouse=True)
def _bootstrap_catalog() -> None:
    reset_default_skills_for_tests()
    clear_skill_catalog()
    register_default_skills()
    yield
    reset_default_skills_for_tests()


def _usage_path_for_skill(skill_id: str) -> Path:
    bundle, _ = skill_id.split(".", 1)
    return SKILLS_PROVIDERS / bundle / skill_id / "USAGE.md"


def test_shipped_catalog_has_forty_nine_skills() -> None:
    assert len(list_catalog_skill_ids()) == 49


def test_each_shipped_skill_has_usage_md() -> None:
    missing: list[str] = []
    for skill_id in list_catalog_skill_ids():
        if not _usage_path_for_skill(skill_id).is_file():
            missing.append(skill_id)
    assert not missing, f"Missing per-skill USAGE.md for: {', '.join(missing)}"


def test_each_skill_usage_md_has_required_sections() -> None:
    violations: list[str] = []
    for skill_id in list_catalog_skill_ids():
        text = _usage_path_for_skill(skill_id).read_text(encoding="utf-8")
        for section in _REQUIRED_SECTIONS:
            if section not in text:
                violations.append(f"{skill_id}: missing {section}")
    assert not violations, "\n".join(violations)


def test_each_bundle_has_index_usage_md() -> None:
    bundles = {skill_id.split(".", 1)[0] for skill_id in list_catalog_skill_ids()}
    missing = [
        bundle_id
        for bundle_id in sorted(bundles)
        if not (SKILLS_PROVIDERS / bundle_id / "USAGE.md").is_file()
    ]
    assert not missing, f"Missing bundle index USAGE.md for: {', '.join(missing)}"
