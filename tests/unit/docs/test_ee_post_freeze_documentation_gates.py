# © Artur Czarnecki. All rights reserved.

"""EE-POST-FREEZE-FINAL — gap audit and documentation reconciliation gates."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests.unit.runtime.architecture._ee_final_arch_facts import p0_bypass_count
from tests.unit.runtime.architecture._ee_final_enterprise_facts import (
    FINAL_ARCHITECTURE_DOC,
    P0_INVENTORY,
)
from tests.unit.runtime.architecture.test_platform_execution_unification_p0_bypass_inventory import (
    _EXPECTED_ENTRYPOINT_COUNT,
    _inventory_doc_text,
    _parse_central_inventory_rows,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]

POST_FREEZE_GAP_AUDIT = (
    _REPO
    / "docs/project/maintainers/qualification/EXECUTION_ENGINE_POST_FREEZE_EXHAUSTIVE_GAP_AUDIT.md"
)
DOC_RECONCILIATION = (
    _REPO
    / "docs/project/maintainers/qualification/EXECUTION_ENGINE_AND_DECISION_DOCUMENTATION_RECONCILIATION.md"
)
INVENTORY_DOC = (
    _REPO
    / "docs/project/maintainers/architecture/EXECUTION_ENGINE_DOCUMENTATION_INVENTORY.md"
)
MAINTAINER_HUB = _REPO / "docs/project/maintainers/architecture/EXECUTION_ENGINE.md"
DECISION_SYSTEM = _REPO / "docs/project/architecture/DECISION_SYSTEM.md"
DECISION_PLAN = _REPO / "docs/project/maintainers/plans/DECISION_SYSTEM.md"

_CANONICAL_DOC_LINKS: tuple[tuple[Path, str], ...] = (
    (MAINTAINER_HUB, "EXECUTION_ENGINE_FINAL_ENTERPRISE_ARCHITECTURE.md"),
    (MAINTAINER_HUB, "DECISION_SYSTEM.md"),
    (MAINTAINER_HUB, "EXECUTION_ENGINE_POST_FREEZE_EXHAUSTIVE_GAP_AUDIT.md"),
    (FINAL_ARCHITECTURE_DOC, "EXECUTION_ENGINE.md"),
    (FINAL_ARCHITECTURE_DOC, "DECISION_SYSTEM.md"),
    (DECISION_SYSTEM, "UNIFIED_EXECUTION_ARCHITECTURE.md"),
    (POST_FREEZE_GAP_AUDIT, "EXECUTION_ENGINE.md"),
    (DOC_RECONCILIATION, "EXECUTION_ENGINE_DOCUMENTATION_INVENTORY.md"),
)

_STALE_DECISION_RUNTIME_OWNER = re.compile(
    r"(?i)(?:canonical|authoritative|lifecycle)\s+owner[^\n]{0,80}\bDecisionRuntime\b"
)

_MERMAID_SECTIONS_FINAL = (
    "## 26. Master end-to-end flow",
    "## 27. Decision integration",
    "## 28. Identity model (diagram)",
    "## 29. Nexus orchestration",
    "## 30. Tool / side-effect path",
    "## 31. Evidence, observability, diagnostics",
    "## 32. Graceful shutdown",
    "## 33. HITL continuation",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_ee_post_freeze_gap_audit_document_present_and_pass() -> None:
    assert POST_FREEZE_GAP_AUDIT.is_file()
    text = _read(POST_FREEZE_GAP_AUDIT)
    assert "FINAL GAP AUDIT VERDICT: PASS" in text
    assert "**PRODUCTION CODE CHANGED** | **NO**" in text
    assert "**SUPPORTED EXECUTION BYPASS** | **0**" in text


def test_ee_post_freeze_documentation_reconciliation_present_and_pass() -> None:
    assert DOC_RECONCILIATION.is_file()
    text = _read(DOC_RECONCILIATION)
    assert "DOCUMENTATION RECONCILIATION: PASS" in text


def test_ee_post_freeze_p0_bypass_zero_ssot() -> None:
    assert p0_bypass_count() == 0
    rows = _parse_central_inventory_rows(_inventory_doc_text())
    assert len(rows) == _EXPECTED_ENTRYPOINT_COUNT
    p0 = _read(P0_INVENTORY)
    supported = re.search(
        r"^\| Supported execution bypasses \(production\) \| (\d+) \|",
        p0,
        flags=re.MULTILINE,
    )
    assert supported is not None
    assert int(supported.group(1)) == 0


def test_execution_engine_hub_start_here_and_post_freeze_links() -> None:
    text = _read(MAINTAINER_HUB)
    assert "### Start here (recommended reading order)" in text
    assert "EXECUTION_ENGINE_POST_FREEZE_EXHAUSTIVE_GAP_AUDIT.md" in text
    assert (
        "Post-freeze exhaustive gap audit" in text
        or "post-freeze gap audit" in text.lower()
    )


def test_execution_documentation_inventory_post_freeze_index() -> None:
    text = _read(INVENTORY_DOC)
    assert "EXECUTION_ENGINE_POST_FREEZE_EXHAUSTIVE_GAP_AUDIT.md" in text
    assert "Last architecture reconciliation" in text


def test_execution_decision_documentation_consistency() -> None:
    decision = _read(DECISION_SYSTEM)
    assert "semantic decision lifecycle hosted by canonical Execution" in decision
    assert "not a second runtime" in decision
    assert "```mermaid" in decision
    assert "ExecutionRuntime" in decision
    final = _read(FINAL_ARCHITECTURE_DOC)
    assert "Decision decides **what**" in final or "Decision decides **WHAT**" in final


def test_execution_documentation_canonical_links_resolve() -> None:
    missing: list[str] = []
    for doc_path, target_fragment in _CANONICAL_DOC_LINKS:
        doc_dir = doc_path.parent
        text = _read(doc_path)
        for match in re.finditer(r"\]\(([^)]+)\)", text):
            href = match.group(1).split("#", 1)[0].strip()
            if not href or href.startswith("http"):
                continue
            if target_fragment not in href:
                continue
            resolved = (doc_dir / href).resolve()
            if not resolved.is_file():
                missing.append(f"{doc_path.name} -> {href}")
    assert missing == []


def test_decision_system_no_runtime_ownership_docs() -> None:
    violations: list[str] = []
    for path in (
        DECISION_SYSTEM,
        DECISION_PLAN,
        FINAL_ARCHITECTURE_DOC,
        MAINTAINER_HUB,
    ):
        text = _read(path)
        for match in _STALE_DECISION_RUNTIME_OWNER.finditer(text):
            if (
                "no DecisionRuntime"
                in text[max(0, match.start() - 40) : match.end() + 40]
            ):
                continue
            violations.append(f"{path.name}: {match.group(0)!r}")
    assert violations == []


def test_decision_plan_marked_implemented_historical() -> None:
    plan = _read(DECISION_PLAN)
    assert "IMPLEMENTED" in plan and "HISTORICAL PLAN" in plan


def test_execution_final_enterprise_diagram_pack_complete() -> None:
    text = _read(FINAL_ARCHITECTURE_DOC)
    for heading in _MERMAID_SECTIONS_FINAL:
        assert heading in text, heading
    assert "```mermaid" in text


def test_ee_post_freeze_gap_audit_head_matches_git() -> None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    head = proc.stdout.strip()
    audit = _read(POST_FREEZE_GAP_AUDIT)
    match = re.search(r"\*\*AUDITED_HEAD\*\* \| `([0-9a-f]{40})`", audit)
    assert match is not None
    audited = match.group(1)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", audited, head],
        cwd=_REPO,
        check=False,
    )
    assert ancestor.returncode == 0 or audited == head
