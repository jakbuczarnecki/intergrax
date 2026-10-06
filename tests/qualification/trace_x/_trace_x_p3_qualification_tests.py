# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P3 mechanical gates TXP3-Q01..TXP3-Q30 and adversarial cases T/P/S."""

from __future__ import annotations

import subprocess

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_evidence.boundary_event import ExecutionBoundaryEvent
from intergrax.contracts.governed_execution_governance_evidence import GovernanceDecisionEvidenceFact
from intergrax.runtime.attestation.execution_boundary_event import ExecutionBoundaryEventV1
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.task.task import Task
from tests.qualification.trace_x._trace_x_p0_support import repo_root
from tests.qualification.trace_x._trace_x_p3_support import (
    ADVERSARIAL_CASES,
    ATTESTATION_BOUNDARY_MODULE,
    ATTRIBUTION_CHAIN_EFFECT_BLOCKED,
    ATTRIBUTION_CHAIN_PROVIDER_BLOCKED,
    ATTRIBUTION_CHAIN_TOOL,
    ENTERPRISE_AUDIT_MATRIX_P3,
    FRZ_TRC_03_DISPOSITION,
    FRZ_TRC_04_DISPOSITION,
    FRZ_TRC_06_DISPOSITION,
    GOVERNED_BOUNDARY_MODULE,
    GOVERNANCE_EVIDENCE_CONTRACT,
    IDENTITY_MATRIX,
    MANDATORY_FRZ_P3_IDS,
    OUT_OF_SCOPE_FRZ,
    OWNERSHIP_MATRIX,
    P3_IN_SCOPE_BLOCKERS,
    P3_READINESS,
    P3Concern,
    P3GateResult,
    P3ReadinessStatus,
    PROPOSED_CHILD_ON_BLOCKER,
    TENANT_ISOLATION_AUDIT_P3,
    TOOL_INVOKER_MODULE,
    TOOL_TRACE_BRIDGE_MODULE,
    TRACE_X_P3_START_HEAD,
    assert_single_reconstruction_owner,
    attestation_boundary_model_text,
    governed_boundary_model_text,
)

_REPO_ROOT = repo_root()
_TENANT = "tenant-p3"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()


def _task_with_identity() -> tuple[Task, str, str, str, str]:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id="user",
        agent_id="agent",
        message="q",
    )
    return task, task_id, run_id, attempt_id, execution_id


def _tool_trace_event(
    *,
    task_id: str,
    run_id: str,
    seq: int,
    tool_name: str = "probe.tool",
) -> TraceEvent:
    return TraceEvent(
        event_id=f"tool-{seq}",
        run_id=run_id,
        seq=seq,
        ts_utc="2026-06-07T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.TOOLS,
        step="tool_invocation_start",
        message="invoke",
        tags={"task_id": task_id, "tool_name": tool_name},
    )


def test_txp3_q01_correct_start_head_provenance() -> None:
    head = _git_head()
    assert head == TRACE_X_P3_START_HEAD
    merge_base = subprocess.check_output(
        ["git", "merge-base", TRACE_X_P3_START_HEAD, head],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()
    assert merge_base == TRACE_X_P3_START_HEAD


def test_txp3_q02_p3_scope_exactly_trc_03_04_06() -> None:
    assert MANDATORY_FRZ_P3_IDS == ("FRZ-TRC-03", "FRZ-TRC-04", "FRZ-TRC-06")
    assert "FRZ-TRC-01" in OUT_OF_SCOPE_FRZ


def test_txp3_q03_tool_evidence_owner_runtime_event() -> None:
    row = next(r for r in OWNERSHIP_MATRIX if "tool invocation" in r.concern)
    assert row.semantic_owner == "RuntimeEvent"
    assert "RuntimeToolInvoker" in row.producer


def test_txp3_q04_governed_boundary_contract_fields() -> None:
    fields = set(ExecutionBoundaryEvent.model_fields)
    assert "task_id" in fields and "run_id" in fields
    assert "attempt_id" not in fields
    assert "execution_id" not in fields
    provider_fields = governed_boundary_model_text()
    assert "class ProviderInvocationSection" in provider_fields
    assert "invocation_id" in provider_fields


def test_txp3_q05_authorization_evidence_owner() -> None:
    row = next(r for r in OWNERSHIP_MATRIX if "governance decision" in r.concern)
    assert row.semantic_owner == "GovernanceDecisionEvidenceFact"
    text = (_REPO_ROOT / GOVERNANCE_EVIDENCE_CONTRACT).read_text(encoding="utf-8")
    assert "has_full_execution_correlation" in text


def test_txp3_q06_no_duplicate_authority() -> None:
    assert_single_reconstruction_owner()
    attestation = attestation_boundary_model_text()
    assert "Unsigned harness" in attestation or "external receipt" in attestation


def test_txp3_q07_tool_bridge_binds_execution_id() -> None:
    task, task_id, run_id, attempt_id, execution_id = _task_with_identity()
    trace = _tool_trace_event(task_id=task_id, run_id=run_id, seq=1)
    event = trace_event_to_runtime_event(
        trace,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert event.event_type == RuntimeEventType.TOOL_REQUESTED
    assert str(event.execution_id) == execution_id
    assert str(event.attempt_id) == attempt_id


def test_txp3_q08_multi_execution_tool_separation() -> None:
    task, task_id, run_id, attempt_id, _ = _task_with_identity()
    e1 = mint_execution_id()
    e2 = mint_execution_id()
    ev1 = trace_event_to_runtime_event(
        _tool_trace_event(task_id=task_id, run_id=run_id, seq=1),
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=e1,
    )
    ev2 = trace_event_to_runtime_event(
        _tool_trace_event(task_id=task_id, run_id=run_id, seq=2),
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=e2,
    )
    assert ev1.execution_id != ev2.execution_id


def test_txp3_q09_multi_attempt_tool_separation() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    a1 = mint_attempt_id()
    a2 = mint_attempt_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id="u",
        agent_id="a",
        message="m",
    )
    ex = mint_execution_id()
    ev_a1 = trace_event_to_runtime_event(
        _tool_trace_event(task_id=task_id, run_id=run_id, seq=1),
        task,
        run_id=run_id,
        attempt_id=a1,
        execution_id=ex,
    )
    ev_a2 = trace_event_to_runtime_event(
        _tool_trace_event(task_id=task_id, run_id=run_id, seq=2),
        task,
        run_id=run_id,
        attempt_id=a2,
        execution_id=ex,
    )
    assert ev_a1.attempt_id != ev_a2.attempt_id


def test_txp3_q10_tenant_isolation_tool_runtime_store() -> None:
    assert TENANT_ISOLATION_AUDIT_P3["result"] == "PASS"
    task, task_id, run_id, attempt_id, execution_id = _task_with_identity()
    event = trace_event_to_runtime_event(
        _tool_trace_event(task_id=task_id, run_id=run_id, seq=1),
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert event.tenant_id == _TENANT


def test_txp3_q11_provider_execution_id_blocker() -> None:
    tool_row = next(r for r in IDENTITY_MATRIX if r.concern == P3Concern.TOOL_INVOCATION)
    provider_row = next(r for r in IDENTITY_MATRIX if r.concern == P3Concern.PROVIDER_INVOCATION)
    assert tool_row.execution is True
    assert provider_row.execution is False
    assert FRZ_TRC_04_DISPOSITION.value.endswith("BLOCKED")


def test_txp3_q12_provider_retry_semantics_blocked() -> None:
    row = next(r for r in ENTERPRISE_AUDIT_MATRIX_P3 if "retry" in r.area)
    assert row.result == P3GateResult.BLOCKED


def test_txp3_q13_provider_tenant_on_boundary() -> None:
    assert "tenant_id" in ExecutionBoundaryEvent.model_fields


def test_txp3_q14_no_heuristic_provider_join() -> None:
    assert "correlation_id" in ExecutionBoundaryEvent.model_fields
    blocker = next(b for b in P3_IN_SCOPE_BLOCKERS if b.frz_id == "FRZ-TRC-04")
    assert "heuristic" in blocker.summary.lower() or "AttemptId" in blocker.summary


def test_txp3_q15_governance_fact_supports_execution_binding() -> None:
    fields = GovernanceDecisionEvidenceFact.model_fields
    assert "execution_id" in fields
    assert "attempt_id" in fields


def test_txp3_q16_allow_exact_invocation_binding_evidence() -> None:
    case = next(c for c in ADVERSARIAL_CASES if c.case_id == "S1")
    assert "test_fresh_side_effect_authorization" in case.evidence_test


def test_txp3_q17_deny_prevents_effect_evidence() -> None:
    case = next(c for c in ADVERSARIAL_CASES if c.case_id == "S2")
    assert "test_fresh_side_effect_authorization" in case.evidence_test


def test_txp3_q18_stale_approval_evidence() -> None:
    case = next(c for c in ADVERSARIAL_CASES if c.case_id == "S6")
    assert "test_fresh_side_effect_authorization" in case.evidence_test


def test_txp3_q19_cross_agent_evidence() -> None:
    case = next(c for c in ADVERSARIAL_CASES if c.case_id == "S4")
    assert "test_fresh_side_effect_authorization" in case.evidence_test


def test_txp3_q20_invocation_scope_mismatch_evidence() -> None:
    case = next(c for c in ADVERSARIAL_CASES if c.case_id == "S5")
    assert "exact_correlation" in case.evidence_test


def test_txp3_q21_effect_authorization_exact_or_blocker() -> None:
    assert FRZ_TRC_06_DISPOSITION.value.endswith("BLOCKED")
    assert PROPOSED_CHILD_ON_BLOCKER == "TRACE-X-P3-R1"
    assert "boundary" in ATTRIBUTION_CHAIN_EFFECT_BLOCKED.lower()


def test_txp3_q22_optional_observability_not_authority() -> None:
    invoker_text = (_REPO_ROOT / TOOL_INVOKER_MODULE).read_text(encoding="utf-8")
    assert "_emit_tool_invocation_start_non_blocking" in invoker_text
    assert "must not block execution" in invoker_text


def test_txp3_q23_attestation_export_not_truth_owner() -> None:
    v1_fields = set(ExecutionBoundaryEventV1.model_fields)
    assert "attempt_id" not in v1_fields
    assert "execution_id" not in v1_fields
    assert (_REPO_ROOT / ATTESTATION_BOUNDARY_MODULE).is_file()
    governed = GOVERNED_BOUNDARY_MODULE
    assert governed != ATTESTATION_BOUNDARY_MODULE


def test_txp3_q24_governance_not_execution() -> None:
    mse = next(r for r in OWNERSHIP_MATRIX if "meaningful side-effect" in r.concern)
    assert mse.semantic_owner.startswith("Governance")


def test_txp3_q25_evidence_recorder_non_authoritative() -> None:
    row = next(r for r in OWNERSHIP_MATRIX if "governance decision evidence" in r.concern)
    assert "GovernanceDecisionEvidenceFact" in row.semantic_owner
    assert "GovernanceEvidenceRecorder" in row.producer


def test_txp3_q26_effect_certainty_truthful() -> None:
    invoker_text = (_REPO_ROOT / TOOL_INVOKER_MODULE).read_text(encoding="utf-8")
    assert "ToolEffectCertainty" in invoker_text


def test_txp3_q27_contracts_over_implementations() -> None:
    assert (_REPO_ROOT / GOVERNED_BOUNDARY_MODULE).is_file()
    assert (_REPO_ROOT / TOOL_TRACE_BRIDGE_MODULE).is_file()


def test_txp3_q28_strong_typing_identity_matrix() -> None:
    assert all(isinstance(row, type(IDENTITY_MATRIX[0])) for row in IDENTITY_MATRIX)


def test_txp3_q29_no_p4_p5_p6_leakage() -> None:
    assert "FRZ-TRC-05" in OUT_OF_SCOPE_FRZ
    assert "FRZ-TRC-07" in OUT_OF_SCOPE_FRZ


def test_txp3_q30_frz_dispositions_truthful() -> None:
    assert FRZ_TRC_03_DISPOSITION.value.endswith("CLOSURE REVIEW")
    assert FRZ_TRC_04_DISPOSITION.value == "BLOCKED"
    assert FRZ_TRC_06_DISPOSITION.value == "BLOCKED"
    assert P3_READINESS == P3ReadinessStatus.BLOCKED


def test_txp3_adversarial_t1_tool_reconstructs_exact_execution() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        sample_runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
        tenant_id=_TENANT,
    )
    reconstruction = ExecutionReconstructor(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT, task_id, run_id)
    positioned = reconstruction.attempts[0].positioned_events
    assert len(positioned) == 1
    assert str(positioned[0].event.execution_id) == execution_id


def test_txp3_adversarial_t2_multi_execution_tool_separation() -> None:
    test_txp3_q08_multi_execution_tool_separation()


def test_txp3_adversarial_t3_multi_attempt_separation() -> None:
    test_txp3_q09_multi_attempt_tool_separation()


def test_txp3_tool_attribution_chain_documented() -> None:
    assert "RuntimeEvent" in ATTRIBUTION_CHAIN_TOOL
    assert "trace_event_to_runtime_event" in ATTRIBUTION_CHAIN_TOOL


def test_txp3_provider_attribution_blocked_documented() -> None:
    assert "TRACE-X-P3-R1" in ATTRIBUTION_CHAIN_PROVIDER_BLOCKED
