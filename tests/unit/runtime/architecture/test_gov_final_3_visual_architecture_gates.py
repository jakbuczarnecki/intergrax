# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-3 — Governance Plane visual architecture documentation gates (semantic)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
ARCH_GOVERNED = _REPO_ROOT / "docs" / "project" / "architecture" / "GOVERNED_EXECUTION.md"
_FORBIDDEN_COMPETING_SSOT = _REPO_ROOT / "docs" / "project" / "architecture" / "GOVERNANCE_VISUAL_ARCHITECTURE.md"

_MERMAID_FENCE = re.compile(r"```mermaid\s*\n", re.MULTILINE)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _hitl_visual_section(arch: str) -> str:
    pattern = re.compile(r"^### 6\. HITL / continuation ownership \(Diagram #5\)\s*$", re.MULTILINE)
    match = pattern.search(arch)
    if not match:
        return ""
    start = match.start()
    next_heading = re.search(r"\n### 7\. ", arch[match.end() :])
    end = match.end() + next_heading.start() if next_heading else len(arch)
    return arch[start:end]


def _section_after(arch: str, heading: str) -> str:
    pattern = re.compile(rf"^## {re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(arch)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"\n## ", arch[start:])
    end = start + next_heading.start() if next_heading else len(arch)
    return arch[start:end]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "").replace("__", "")).strip()


def test_gov_final_3_visual_layer_section_present() -> None:
    arch = _read(ARCH_GOVERNED)
    assert "Visual Architecture Layer (GOV-FINAL-3)" in arch
    visual = _section_after(arch, "Visual Architecture Layer (GOV-FINAL-3)")
    assert visual, "visual layer section must be non-empty"


def test_gov_final_3_required_visual_topics() -> None:
    arch = _read(ARCH_GOVERNED)
    visual = _section_after(arch, "Visual Architecture Layer (GOV-FINAL-3)")
    required_phrases = (
        "Governance ownership map",
        "Root admission vs inner governance",
        "Meaningful side effect flow",
        "Decision → Governance → Execution",
        "HITL / continuation ownership",
        "Governance → Reliability handoff",
        "Fail-closed negative paths",
        "Strategy coverage",
        "Evidence / Diagnostics",
        "Control-plane mutation",
        "Pluginability",
        "Architecture boundary table",
    )
    for phrase in required_phrases:
        assert phrase in visual, f"missing visual topic: {phrase}"


def test_gov_final_3_mermaid_diagram_presence() -> None:
    arch = _read(ARCH_GOVERNED)
    visual = _section_after(arch, "Visual Architecture Layer (GOV-FINAL-3)")
    blocks = _MERMAID_FENCE.findall(visual)
    assert len(blocks) >= 10, f"expected at least 10 Mermaid diagrams in visual layer, got {len(blocks)}"
    for diagram_kind in (
        "flowchart",
        "sequenceDiagram",
    ):
        assert diagram_kind in visual, f"visual layer must include {diagram_kind}"


def test_gov_final_3_mermaid_blocks_balanced() -> None:
    arch = _read(ARCH_GOVERNED)
    visual = _section_after(arch, "Visual Architecture Layer (GOV-FINAL-3)")
    opens = len(_MERMAID_FENCE.findall(visual))
    closes = visual.count("```\n") + (1 if visual.rstrip().endswith("```") else 0)
    assert opens >= 10
    assert visual.count("```mermaid") == opens
    fence_triples = visual.count("```")
    assert fence_triples >= opens * 2, "unbalanced markdown fences in visual layer"


def test_gov_final_3_authority_semantics_markers() -> None:
    arch = _read(ARCH_GOVERNED)
    norm = _normalize(arch)
    assert "Decision System owns decision truth" in norm or "Decision System — owns decision truth" in arch
    assert "Governance Plane — owns permission" in arch or "Governance owns permission" in norm
    assert "Execution Runtime — owns lifecycle" in arch or "Execution owns pause/resume lifecycle" in arch
    assert "Reliability — owns post-admission uncertainty" in arch or "post-admission uncertainty" in norm
    assert "Evidence ≠ authority" in arch or "Evidence is not authority" in norm
    assert "Diagnostics ≠ authority" in arch or "Diagnostics is not authority" in norm
    assert "Nexus — internal orchestration only" in arch or "Nexus is internal" in arch


def test_gov_final_3_hitl_visual_lifecycle_order_preserves_execution_ownership() -> None:
    arch = _read(ARCH_GOVERNED)
    hitl = _hitl_visual_section(arch)
    assert hitl, "Diagram #5 HITL section must exist"
    norm = _normalize(hitl)
    markers = (
        "Governance REQUIRE_HUMAN",
        "GovernedContinuationRequest",
        "ExecutionContinuationPort",
        "PAUSE",
        "WAITING",
        "Human Review",
        "Human result",
        "fresh Governance evaluation",
        "resume",
        "remain blocked",
    )
    for marker in markers:
        assert marker.lower() in norm.lower(), f"HITL visual section missing marker: {marker!r}"
    assert "Execution owns" in hitl or "Execution owns" in norm
    ecp_idx = norm.lower().find("executioncontinuationport")
    hr_idx = norm.lower().find("human review")
    fresh_idx = norm.lower().find("fresh governance evaluation")
    assert ecp_idx != -1 and hr_idx != -1, "order markers missing for ExecutionContinuationPort / Human Review"
    assert ecp_idx < hr_idx, "ExecutionContinuationPort must appear before Human Review in HITL section"
    assert hr_idx < fresh_idx, "Human Review must appear before fresh Governance evaluation"
    resume_idx = norm.lower().find("resume")
    assert fresh_idx != -1 and resume_idx != -1
    assert fresh_idx < resume_idx, "fresh Governance evaluation must precede resume semantics in HITL section"
    canonical = "Canonical lifecycle order"
    assert canonical in hitl
    order_tail = hitl.split(canonical, 1)[1]
    order_norm = _normalize(order_tail)
    for step in (
        "REQUIRE_HUMAN",
        "GovernedContinuationRequest",
        "ExecutionContinuationPort",
        "PAUSE / WAITING",
        "Human Review",
        "Human result",
        "fresh Governance evaluation",
        "ALLOW/DENY",
    ):
        assert step.lower() in order_norm.lower(), f"canonical lifecycle order missing: {step!r}"
    pos = 0
    for step in (
        "REQUIRE_HUMAN",
        "GovernedContinuationRequest",
        "ExecutionContinuationPort",
        "PAUSE / WAITING",
        "Human Review",
        "Human result",
        "fresh Governance evaluation",
    ):
        idx = order_norm.lower().find(step.lower(), pos)
        assert idx != -1, f"canonical order sequence broken at {step!r}"
        pos = idx


def test_gov_final_3_human_approval_not_automatic_allow() -> None:
    arch = _read(ARCH_GOVERNED)
    norm = _normalize(arch)
    assert (
        "Human APPROVED ≠ automatic Governance ALLOW" in arch
        or "Human APPROVED != automatic Governance ALLOW" in norm
        or "Decision accepted ≠ Governance ALLOW" in arch
    )


def test_gov_final_3_reliability_not_governance_authority() -> None:
    arch = _read(ARCH_GOVERNED)
    assert "Reliability ≠ Governance authority" in arch or "Reliability != Governance authority" in arch


def test_gov_final_3_control_plane_not_closed_or_covered() -> None:
    arch = _read(ARCH_GOVERNED)
    visual = _section_after(arch, "Visual Architecture Layer (GOV-FINAL-3)")
    assert "CONTROL_PLANE_MUTATION" in visual
    assert "GAP" in visual
    assert "TARGET" in visual
    cp_lines = [line for line in visual.splitlines() if "CONTROL_PLANE_MUTATION" in line]
    assert cp_lines, "visual layer must name CONTROL_PLANE_MUTATION"
    for line in cp_lines:
        if re.search(r"status\s+remains\s+\*\*GAP\*\*", line, re.IGNORECASE):
            continue
        if re.search(r"\bCLOSED\b", line, re.IGNORECASE) and "not" not in line.lower():
            pytest.fail(f"control plane must not read CLOSED in visual layer: {line!r}")
        if re.search(r"\|\s*\*\*COVERED\*\*\s*\|", line):
            pytest.fail(f"control plane must not read COVERED in visual layer: {line!r}")


def test_gov_final_3_strategy_matrix_shows_open_gaps() -> None:
    arch = _read(ARCH_GOVERNED)
    visual = _section_after(arch, "Visual Architecture Layer (GOV-FINAL-3)")
    assert "INFERENCE" in visual and "AGENTIC" in visual and "ORCHESTRATION" in visual
    assert "GAP" in visual
    assert "PARTIAL" in visual
    assert "GR-10" in visual or "qualification matrix remains" in visual.lower()


def test_gov_final_3_gr8_spine_vs_gep_adoption_honesty() -> None:
    arch = _read(ARCH_GOVERNED)
    visual = _section_after(arch, "Visual Architecture Layer (GOV-FINAL-3)")
    assert "GR-8" in visual
    assert "CANDIDATE CLOSED" in visual
    assert "GR-10" in visual or "GR-13" in visual
    assert "five-ID governance emission as complete" not in visual.lower()


def test_gov_final_3_canonical_ports_named() -> None:
    arch = _read(ARCH_GOVERNED)
    visual = _section_after(arch, "Visual Architecture Layer (GOV-FINAL-3)")
    for port in (
        "RuntimeExecutionPolicyAdmissionPort",
        "CanonicalInnerExecutionGuardPort",
        "DecisionRequirementPolicy",
        "ExecutionContinuationPort",
        "ProviderInvocationStore",
        "MeaningfulSideEffectAuthorizationBoundary",
    ):
        assert port in visual, f"missing port reference in visual layer: {port}"


def test_gov_final_3_no_competing_visual_ssot_file() -> None:
    assert not _FORBIDDEN_COMPETING_SSOT.is_file(), (
        "GOVERNANCE_VISUAL_ARCHITECTURE.md would compete with GOVERNED_EXECUTION.md"
    )


def test_gov_final_3_status_legend_vocabulary() -> None:
    arch = _read(ARCH_GOVERNED)
    visual = _section_after(arch, "Visual Architecture Layer (GOV-FINAL-3)")
    for status in ("COVERED", "PARTIAL", "GAP", "TARGET", "NOT_APPLICABLE"):
        assert status in visual, f"legend must define {status}"
