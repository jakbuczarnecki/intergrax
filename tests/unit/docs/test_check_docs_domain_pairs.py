# © Artur Czarnecki. All rights reserved.

"""Invariant tests for docs domain/feature pair integrity checker."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCH = REPO_ROOT / "docs" / "project" / "architecture"
PLAN = REPO_ROOT / "docs" / "project" / "maintainers" / "plans"
FEATURE_ARCH = REPO_ROOT / "docs" / "project" / "capabilities" / "architecture"
FEATURE_PLAN = REPO_ROOT / "docs" / "project" / "capabilities" / "plan"
CAPABILITIES_README = REPO_ROOT / "docs" / "project" / "capabilities" / "README.md"
CHECKER = REPO_ROOT / "scripts" / "docs" / "check_docs_domain_pairs.py"
_DOCS_SCRIPTS = REPO_ROOT / "scripts" / "docs"
if str(_DOCS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_DOCS_SCRIPTS))
from docs_domain_common import canonical_domain_ids  # noqa: E402
import check_docs_domain_pairs as domain_pairs  # noqa: E402

_FEATURE_INDEX_ROW = re.compile(r"^\| `([A-Z][A-Z0-9_]+)` \|", re.MULTILINE)


def _canonical_feature_ids() -> set[str]:
    text = CAPABILITIES_README.read_text(encoding="utf-8")
    start = text.find("## Current multi-layer features")
    end = text.find("**Satellites", start)
    return set(_FEATURE_INDEX_ROW.findall(text[start:end]))


def test_canonical_domain_ids_have_architecture_and_plan_files() -> None:
    canonical = set(canonical_domain_ids())
    arch_ids = {path.stem for path in ARCH.glob("*.md")}
    plan_ids = {path.stem for path in PLAN.glob("*.md")}
    assert canonical <= arch_ids
    assert canonical <= plan_ids
    assert len(canonical) == len(canonical_domain_ids())


def test_canonical_feature_ids_have_architecture_and_plan_files() -> None:
    feature_ids = _canonical_feature_ids()
    arch_ids = {path.stem for path in FEATURE_ARCH.glob("*.md")}
    plan_ids = {path.stem for path in FEATURE_PLAN.glob("*.md")}
    assert feature_ids <= arch_ids
    assert feature_ids <= plan_ids


def test_check_docs_domain_pairs_passes_on_current_repo() -> None:
    completed = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    canonical_count = len(canonical_domain_ids())
    assert f"check_docs_domain_pairs: OK ({canonical_count} domain pairs" in completed.stdout


def test_checker_fails_when_canonical_architecture_side_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(domain_pairs, "canonical_domain_ids", lambda: ("MISSING_ARCH_DOMAIN",))
    monkeypatch.setattr(domain_pairs, "DOMAIN_ORDER", ("MISSING_ARCH_DOMAIN",))
    count = domain_pairs._check_domain_pairs(errors)
    assert count == 1
    assert any("missing architecture/MISSING_ARCH_DOMAIN.md" in item for item in errors)


def test_checker_fails_when_canonical_plan_side_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "PLAN_MISSING_ONLY"
    arch_file = ARCH / f"{name}.md"
    arch_file.write_text(
        "maintainers/plans/PLAN_MISSING_ONLY.md\narchitecture/PLAN_MISSING_ONLY.md",
        encoding="utf-8",
    )
    try:
        errors: list[str] = []
        monkeypatch.setattr(domain_pairs, "canonical_domain_ids", lambda: (name,))
        monkeypatch.setattr(domain_pairs, "DOMAIN_ORDER", (name,))
        count = domain_pairs._check_domain_pairs(errors)
        assert count == 1
        assert any("missing maintainers/plans/PLAN_MISSING_ONLY.md" in item for item in errors)
    finally:
        arch_file.unlink(missing_ok=True)


def test_checker_fails_on_feature_pair_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(domain_pairs, "_canonical_feature_ids", lambda: ["FEATURE_A", "FEATURE_B"])
    domain_pairs._check_feature_pairs(errors)
    assert any("canonical feature missing capabilities/architecture/FEATURE_A.md" in item for item in errors)
