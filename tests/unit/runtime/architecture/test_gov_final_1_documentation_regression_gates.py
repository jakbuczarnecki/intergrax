# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-1 — Governance documentation regression gates (semantic invariants)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]

ARCH_GOVERNED = _REPO_ROOT / "docs" / "project" / "architecture" / "GOVERNED_EXECUTION.md"
PLAN_GOVERNED = _REPO_ROOT / "docs" / "project" / "maintainers" / "plans" / "GOVERNED_EXECUTION.md"
GAP_LEDGER = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "GOVERNANCE_ARCHITECTURE_REBASE_GAP_LEDGER.md"
)

_COMPETING_GOV_SSOT = (
    "canonical architecture SSOT for the entire Governance Plane",
    "sole Governance Plane architecture SSOT",
)

_GR6_PLANNED_ROW = re.compile(
    r"GR-6\s*\|[^\n]*\|\s*Planned\b",
    re.IGNORECASE,
)
_GR7_PLANNED_ROW = re.compile(
    r"GR-7\s*\|[^\n]*\|\s*Planned\b",
    re.IGNORECASE,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "").replace("__", "")).strip()


def test_gov_final_1_architecture_ssot_declaration() -> None:
    arch = _read(ARCH_GOVERNED)
    assert "canonical architecture SSOT for the entire Governance Plane" in arch
    assert "GOV-FINAL-1 reconciliation" in arch
    assert "Governance implementation truth" in arch


def test_gov_final_1_maintainer_plan_not_competing_ssot() -> None:
    plan = _read(PLAN_GOVERNED)
    for claim in _COMPETING_GOV_SSOT:
        assert claim not in plan, "maintainer plan must not declare competing Governance architecture SSOT"
    assert "Canonical architecture:" in plan
    assert "GOVERNED_EXECUTION.md" in plan


def test_gov_final_1_gr6_gr7_not_regressed_to_planned_in_plan() -> None:
    plan = _read(PLAN_GOVERNED)
    assert not _GR6_PLANNED_ROW.search(plan), "GR-6 must not regress to Planned in maintainer roadmap table"
    assert not _GR7_PLANNED_ROW.search(plan), "GR-7 must not regress to Planned in maintainer roadmap table"
    assert "IMPLEMENTED" in plan and "qualification **OPEN**" in plan


def test_gov_final_1_gap_ledger_gr6_gr7_implementation_truth() -> None:
    ledger = _read(GAP_LEDGER)
    gr6_lines = [line for line in ledger.splitlines() if "| GR-6 |" in line and "Decision" in line]
    gr7_lines = [line for line in ledger.splitlines() if "| GR-7 |" in line and "Reliability" in line]
    assert gr6_lines, "gap ledger must contain GR-6 roadmap row"
    assert gr7_lines, "gap ledger must contain GR-7 roadmap row"
    assert "IMPLEMENTED" in gr6_lines[-1]
    assert "IMPLEMENTED" in gr7_lines[-1]
    assert _GR6_PLANNED_ROW.search(ledger) is None
    assert _GR7_PLANNED_ROW.search(ledger) is None


def test_gov_final_1_authority_boundary_phrases_preserved() -> None:
    arch = _read(ARCH_GOVERNED)
    norm = _normalize(arch)
    assert "Reliability ≠ Governance authority" in arch or "Reliability != Governance authority" in arch
    assert "Human APPROVED ≠ automatic Governance ALLOW" in arch or "Human APPROVED != automatic Governance ALLOW" in norm
    assert "ExecutionContinuationPort" in arch
    assert "Nexus is internal" in norm or "Nexus is internal Execution" in arch


def test_gov_final_1_control_plane_mutation_not_enterprise_closed() -> None:
    arch = _read(ARCH_GOVERNED)
    assert "CONTROL_PLANE_MUTATION" in arch
    g3b_rows = [line for line in arch.splitlines() if "CONTROL_PLANE_MUTATION" in line and "|" in line]
    assert g3b_rows, "G3B table must include CONTROL_PLANE_MUTATION row"
    status_cell = g3b_rows[0].split("|")[2].strip() if len(g3b_rows[0].split("|")) > 2 else ""
    assert "GAP" in status_cell.upper(), f"CONTROL_PLANE_MUTATION status must remain GAP, got {status_cell!r}"
    assert "CLOSED" not in status_cell.upper()


def test_gov_final_1_no_full_governance_enterprise_certification_claim() -> None:
    arch = _read(ARCH_GOVERNED)
    forbidden = (
        "FULL GOVERNANCE ENTERPRISE CERTIFIED",
        "Governance Layer enterprise certified",
        "full Governance Plane enterprise certified",
    )
    for phrase in forbidden:
        assert phrase not in arch
