# © Artur Czarnecki. All rights reserved.

"""GEC-4 — Governed Continuation composition over existing Nexus interrupt model."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.agent_decision import AgentDecisionType
from intergrax.contracts.execution_interrupt import InterruptType
from intergrax.contracts.external_work import QuoteAcceptanceEvidence
from intergrax.contracts.governed_continuation import (
    META_CONTINUATION_REASON,
    ContinuationEvidenceRefs,
    ContinuationReason,
    GovernedContinuationRequest,
    attach_continuation_refs_to_quote_acceptance,
    compose_continuation_agent_decision,
    compose_continuation_interrupt,
    continuation_evidence_refs_from_quote_acceptance,
    continuation_reason_from_interrupt,
)
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.human.models import HumanDecisionRecord, HumanResponseVerdict

_DIGEST = "sha256:" + ("ab" * 32)
_T0 = datetime(2026, 7, 20, 16, 0, 0, tzinfo=timezone.utc)
_MODULE = Path("intergrax/contracts/governed_continuation.py")
_FORBIDDEN_RUNTIME_NAMES = {
    "ContinuationRuntime",
    "ContinuationEngine",
    "ContinuationManager",
    "QuoteRuntime",
    "QuoteLifecycleEngine",
    "GovernanceRuntime",
}


def _request(**overrides: object) -> GovernedContinuationRequest:
    payload: dict[str, object] = {
        "reason": ContinuationReason.QUOTE,
        "task_id": "task-gc-1",
        "run_id": "run-gc-1",
        "source_agent_id": "external_contractor_adapter",
        "prompt": "Quote requires governed continuation",
        "correlation": {
            "external_task_id": "ext-1",
            "provider_id": "fake",
            "idempotency_key": "idem-1",
        },
        "context": {"quote_id": "q-1", "quote_version": 1},
    }
    payload.update(overrides)
    return GovernedContinuationRequest.model_validate(payload)


def _acceptance(**overrides: object) -> QuoteAcceptanceEvidence:
    payload: dict[str, object] = {
        "acceptance_id": "acc-gc-1",
        "quote_id": "q-1",
        "quote_version": 1,
        "scope_digest": _DIGEST,
        "actor": ActorIdentity(
            kind=ActorKind.USER, actor_id="u1", tenant_id="tenant-a"
        ),
        "accepted_at": _T0,
    }
    payload.update(overrides)
    return QuoteAcceptanceEvidence.model_validate(payload)


@pytest.mark.unit
@pytest.mark.gate
def test_continuation_reason_is_generic_not_quote_only() -> None:
    values = {r.value for r in ContinuationReason}
    assert "quote" in values
    assert {"security", "legal", "procurement", "compliance", "publication"} <= values


@pytest.mark.unit
@pytest.mark.gate
def test_compose_continuation_interrupt_reuses_execution_interrupt() -> None:
    request = _request()
    interrupt = compose_continuation_interrupt(request, interrupt_id="int_gc_fixed")
    assert interrupt.interrupt_id == "int_gc_fixed"
    assert interrupt.interrupt_type is InterruptType.HUMAN_JUDGMENT_REQUIRED
    assert interrupt.blocking is True
    assert interrupt.task_id == "task-gc-1"
    assert interrupt.run_id == "run-gc-1"
    assert continuation_reason_from_interrupt(interrupt) is ContinuationReason.QUOTE
    assert interrupt.metadata[META_CONTINUATION_REASON] == "quote"
    assert interrupt.metadata["continuation.correlation"]["external_task_id"] == "ext-1"


@pytest.mark.unit
@pytest.mark.gate
def test_interrupt_handler_composition_pause_without_new_runtime() -> None:
    request = _request()
    interrupt = compose_continuation_interrupt(request)
    decision = compose_continuation_agent_decision(request, interrupt)
    assert decision.type is AgentDecisionType.INTERRUPT
    assert decision.interrupt_id == interrupt.interrupt_id

    resolution = ExecutionInterruptHandler().resolve_decision(
        decision,
        task_id=request.task_id,
        run_id=request.run_id,
        agent_id=request.source_agent_id,
    )
    assert resolution.should_pause is True
    assert resolution.interrupt is not None
    assert continuation_reason_from_interrupt(resolution.interrupt) is ContinuationReason.QUOTE


@pytest.mark.unit
@pytest.mark.gate
def test_resume_composition_propagates_evidence_refs() -> None:
    """Human decision + interrupt → QuoteAcceptanceEvidence refs (no Tier-2 decision)."""
    request = _request()
    interrupt = compose_continuation_interrupt(request, interrupt_id="int_resume_1")
    human = HumanDecisionRecord(
        decision_id="hdec_resume_1",
        task_id=request.task_id,
        tenant_id="tenant-a",
        approver=local_development_approver_evidence(tenant_id="tenant-a", actor_id="u1"),
        verdict=HumanResponseVerdict.APPROVE,
        created_at_utc=_T0.isoformat(),
        run_id=request.run_id,
    )
    acceptance = attach_continuation_refs_to_quote_acceptance(
        _acceptance(),
        hitl_decision_id=human.decision_id,
        interrupt_id=interrupt.interrupt_id,
        policy_decision_ref="pol_dec_1",
    )
    refs = continuation_evidence_refs_from_quote_acceptance(acceptance)
    assert refs == ContinuationEvidenceRefs(
        reason=ContinuationReason.QUOTE,
        hitl_decision_id="hdec_resume_1",
        interrupt_id="int_resume_1",
        policy_decision_ref="pol_dec_1",
    )
    assert acceptance.hitl_decision_id == human.decision_id
    assert acceptance.interrupt_id == interrupt.interrupt_id


@pytest.mark.unit
@pytest.mark.gate
def test_quote_acceptance_remains_minimum_continuation_evidence_for_quote() -> None:
    acceptance = _acceptance(
        hitl_decision_id="hdec_min",
        interrupt_id="int_min",
    )
    refs = continuation_evidence_refs_from_quote_acceptance(acceptance)
    assert refs.reason is ContinuationReason.QUOTE
    assert refs.hitl_decision_id == "hdec_min"
    assert refs.interrupt_id == "int_min"


@pytest.mark.unit
@pytest.mark.gate
def test_no_duplicate_continuation_runtime_in_platform_module() -> None:
    source = _MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_MODULE))
    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    assert not (class_names & _FORBIDDEN_RUNTIME_NAMES)
    for forbidden in _FORBIDDEN_RUNTIME_NAMES:
        assert forbidden not in source
    assert "not a runtime" in source.lower() or "**not** a runtime" in source.lower()
