# © Artur Czarnecki. All rights reserved.

"""GR-10-R12 — ORCHESTRATION Continuation enterprise qualification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_ORCHESTRATION_CONTINUATION_INVENTORY,
    GR10_R12_NEXT_REMEDIATION,
    Gr10CoverageStatus,
    gr10_matrix_orchestration_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NEXUS_LOOP = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"
_HOST = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "nexus_host_execution.py"
_BRIDGE = (
    _REPO_ROOT / "intergrax" / "runtime" / "human" / "governed_continuation_bridge.py"
)
_DURABILITY = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "continuation"
    / "durability_policy.py"
)
_INTERNAL = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "orchestration"
    / "internal_continuation_orchestration.py"
)
_INTAKE = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "orchestration" / "intake_runner.py"
)
_TOPOLOGY = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "orchestration_topology_submission.py"
)


def test_gr10_r12_orchestration_continuation_qualified() -> None:
    row = next(
        row
        for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS
        if row.capability == "Continuation"
    )
    assert row.coverage is Gr10CoverageStatus.QUALIFIED
    assert gr10_matrix_orchestration_status("Continuation") is Gr10CoverageStatus.QUALIFIED


def test_gr10_r12_continuation_inventory_has_no_gap_rows() -> None:
    assert GR10_ORCHESTRATION_CONTINUATION_INVENTORY
    for row in GR10_ORCHESTRATION_CONTINUATION_INVENTORY:
        assert row.coverage in {
            "QUALIFIED — sole pause/wait/resolve/resume authority",
            "QUALIFIED — PAUSE_REQUESTED→PAUSED→WAITING_FOR_HUMAN only",
            "QUALIFIED — canonical-first then Task projection",
            "QUALIFIED — current episode RESUMED only",
            "projection only — not resume authority",
            "QUALIFIED — capability bundle; not parallel engine",
            "QUALIFIED — NEEDS_INPUT projects wait; port owns pause",
            "QUALIFIED — Task CREATED only after canonical RESUMED",
            "QUALIFIED — MSE+HITL gate requires port RESUMED",
            "projection only — eligibility; not resume authority",
            "QUALIFIED — refuses continuable HITL slots",
            "QUALIFIED — active store / injected port only",
            "QUALIFIED — fail-closed without active store",
            "QUALIFIED — resolution only; RESUME_AUTHORIZED ≠ RESUMED",
            "projection only — terminal authority may block; never resume permission",
            "QUALIFIED — production requires durable store; shared canonical store",
            "QUALIFIED — production requires store.is_durable; no type whitelist",
            "QUALIFIED — contract ABC; custom durable provider via composition",
            "QUALIFIED — reference restart durability; named vendor adapter N/A",
            "N/A — lab/test only; production rejects even when explicit",
            "N/A — reconnect only; is_durable=False; not production",
            "QUALIFIED — explicit production_mode; durable store enforced in production",
            "QUALIFIED — mode propagated; host_execution path reuses qualified host",
            "QUALIFIED — no second NexusLoop; host continuation authority reused",
        }
        if row.production and row.coverage.startswith("QUALIFIED"):
            if not row.projection_only and "refuses" not in row.coverage:
                assert row.uses_canonical_port is True or "eligibility" in row.coverage
        if row.production and "GAP" in row.coverage:
            raise AssertionError(f"production GAP row: {row.path}")


def test_gr10_r12_next_remediation_is_reliability() -> None:
    assert GR10_R12_NEXT_REMEDIATION.capability == "Reliability"
    assert "GR-10-R13" in GR10_R12_NEXT_REMEDIATION.task_name


def test_gr10_r12_governance_evidence_qualified_after_r14() -> None:
    assert (
        gr10_matrix_orchestration_status("Governance Evidence")
        is Gr10CoverageStatus.QUALIFIED
    )


def test_gr10_r12_production_forbids_silent_continuation_downgrade() -> None:
    source = _DURABILITY.read_text(encoding="utf-8-sig")
    assert "validate_execution_continuation_for_composition" in source
    assert "silent in-memory" in source
    assert "is_durable" in source
    assert "isinstance" not in source
    loop = _NEXUS_LOOP.read_text(encoding="utf-8-sig")
    assert "validate_execution_continuation_for_composition" in loop
    factory = (
        _REPO_ROOT
        / "intergrax"
        / "applications"
        / "_shared"
        / "nexus_factory.py"
    ).read_text(encoding="utf-8-sig")
    assert "wire_execution_continuation_state_store()" not in factory
    assert "execution_continuation_state_store=execution_continuation_state_store" in factory


def test_gr10_r12_host_shares_nexus_continuation_store() -> None:
    source = _HOST.read_text(encoding="utf-8-sig")
    assert "nexus_loop.execution_continuation_state_store" in source
    assert "wire_execution_continuation_state_store()" not in source


def test_gr10_r12_bridge_fail_closed_without_active_store() -> None:
    source = _BRIDGE.read_text(encoding="utf-8-sig")
    assert "InternalHitlContinuationCapabilityError" in source
    assert "no silent in-memory downgrade" in source
    assert "wire_execution_engine_continuation_dependencies(state_store=None)" not in source
    assert "wire_execution_engine_continuation_dependencies(state_store=active_store)" in source


def test_gr10_r12_internal_continuation_is_capability_bundle_ast() -> None:
    source = _INTERNAL.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_INTERNAL))
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    assert "InternalOrchestrationContinuation" in class_names
    assert "ExecutionContinuationService" not in class_names


def test_gr10_r12_intake_requires_canonical_resumed_before_task_created() -> None:
    source = _INTAKE.read_text(encoding="utf-8-sig")
    assert "canonical_execution_is_resumed" in source
    idx_canon = source.index("canonical_execution_is_resumed")
    idx_created = source.index("task.state = TaskState.CREATED")
    assert idx_canon < idx_created


def test_gr10_r12_recover_failed_slot_refuses_continuable() -> None:
    source = _TOPOLOGY.read_text(encoding="utf-8-sig")
    assert "governed continuation slot must use continue_slot" in source
    assert 'code="require_human"' in source
