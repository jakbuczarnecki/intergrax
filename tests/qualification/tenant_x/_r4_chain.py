# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R4 E2E continuity chains (context/model, async queue, evidence/events)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present
from tests.qualification.tenant_x._r4_types import R4ChainTransition, R4ChainVerdict

_CHAIN1_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/runtime/events/context_skill_recording.py",
        "peek_active_runtime_event_tenant_id",
    ),
    (
        "intergrax/runtime/llm/model_call_attribution.py",
        "context_assembly_event_id",
    ),
    (
        "intergrax/runtime/llm/model_context_attribution.py",
        "try_attribute_model_call_to_context",
    ),
)

_CHAIN2_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/runtime/vendor_knowledge/sync_runtime.py",
        "_TenantVendorKnowledgeTaskQueue",
    ),
    (
        "intergrax/runtime/background_execution/bootstrap.py",
        "resolve_background_execution",
    ),
    (
        "intergrax/queueing/worker/execution.py",
        "execution_identity.tenant_id",
    ),
)

_CHAIN3_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/contracts/functional_evidence/tenant.py",
        "require_tenant_id_from_exec_ctx",
    ),
    (
        "intergrax/runtime/llm/model_context_attribution.py",
        "ModelContextAttributionEvidence",
    ),
)


def _verdict_from_checks(
    checks: tuple[tuple[str, str], ...],
    *,
    discovery_note: str,
) -> tuple[R4ChainVerdict, tuple[str, ...]]:
    matched = all_structural_patterns_present(checks)
    if len(matched) == len(checks):
        return R4ChainVerdict.PROVEN_CURRENT_HEAD, matched
    return R4ChainVerdict.MISSING, (discovery_note,)


def build_r4_e2e_chain() -> tuple[R4ChainTransition, ...]:
    v1, ev1 = _verdict_from_checks(_CHAIN1_CHECKS, discovery_note="context→model chain incomplete")
    v2, ev2 = _verdict_from_checks(_CHAIN2_CHECKS, discovery_note="queue→worker chain incomplete")
    v3, ev3 = _verdict_from_checks(_CHAIN3_CHECKS, discovery_note="evidence/event chain incomplete")
    return (
        R4ChainTransition(
            chain_id="chain1.context_model",
            transition_id="c1.execution_to_context_assembly",
            source_symbol="active execution / runtime event tenant scope",
            target_symbol="record_context_assembled_from_engine",
            tenant_source="peek_active_runtime_event_tenant_id",
            tenant_target="RuntimeEvent.tenant_id on CONTEXT_ASSEMBLED",
            relation="SAME",
            mismatch_behavior="missing active tenant → event tenant_id may be None; attribution fail-closed",
            fail_closed=True,
            verdict=v1,
            evidence=ev1,
        ),
        R4ChainTransition(
            chain_id="chain1.context_model",
            transition_id="c1.context_event_to_llm_event",
            source_symbol="CONTEXT_ASSEMBLED",
            target_symbol="LLM_CALL",
            tenant_source="context_event.tenant_id",
            tenant_target="llm_event.tenant_id",
            relation="SAME",
            mismatch_behavior="tenant_mismatch → attribution rejected",
            fail_closed=True,
            verdict=v1,
            evidence=ev1,
        ),
        R4ChainTransition(
            chain_id="chain1.context_model",
            transition_id="c1.llm_to_attribution",
            source_symbol="LLM_CALL payload",
            target_symbol="try_attribute_model_call_to_context",
            tenant_source="llm_event.tenant_id",
            tenant_target="ModelContextAttributionEvidence.tenant_id",
            relation="SAME",
            mismatch_behavior="hash / event_id / tenant mismatch → not attributable",
            fail_closed=True,
            verdict=v1,
            evidence=ev1,
        ),
        R4ChainTransition(
            chain_id="chain2.async_queue_worker",
            transition_id="c2.enqueue_task_request",
            source_symbol="scheduler / admission tenant",
            target_symbol="TaskRequest.tenant_id",
            tenant_source="canonical tenant A",
            tenant_target="TaskRequest.tenant_id",
            relation="SAME",
            mismatch_behavior="queue bound A + request B → ValueError",
            fail_closed=True,
            verdict=v2,
            evidence=ev2,
        ),
        R4ChainTransition(
            chain_id="chain2.async_queue_worker",
            transition_id="c2.worker_intake",
            source_symbol="TaskRequest",
            target_symbol="worker handler tenant_id",
            tenant_source="request.tenant_id",
            tenant_target="handler tenant_id parameter",
            relation="SAME",
            mismatch_behavior="decode job tenant mismatch → fail result",
            fail_closed=True,
            verdict=v2,
            evidence=ev2,
        ),
        R4ChainTransition(
            chain_id="chain2.async_queue_worker",
            transition_id="c2.background_identity",
            source_symbol="transport_ref.tenant_id",
            target_symbol="BackgroundExecutionIdentity.tenant_id",
            tenant_source="transport + optional task_tenant_id",
            tenant_target="identity.tenant_id",
            relation="SAME",
            mismatch_behavior="BackgroundExecutionTenantMismatchError",
            fail_closed=True,
            verdict=v2,
            evidence=ev2,
        ),
        R4ChainTransition(
            chain_id="chain2.async_queue_worker",
            transition_id="c2.identity_to_execution",
            source_symbol="BackgroundExecutionIdentity",
            target_symbol="handler / execution context",
            tenant_source="identity.tenant_id",
            tenant_target="execution tenant",
            relation="SAME",
            mismatch_behavior="no run_id tenant inference",
            fail_closed=True,
            verdict=v2,
            evidence=ev2,
        ),
        R4ChainTransition(
            chain_id="chain3.evidence_events",
            transition_id="c3.exec_to_functional_evidence",
            source_symbol="RuntimeExecutionContext",
            target_symbol="functional evidence tenant",
            tenant_source="canonical_request_identity.tenant_id",
            tenant_target="evidence record tenant",
            relation="SAME",
            mismatch_behavior="FunctionalEvidenceTenantIdentityError",
            fail_closed=True,
            verdict=v3,
            evidence=ev3,
        ),
        R4ChainTransition(
            chain_id="chain3.evidence_events",
            transition_id="c3.events_to_attribution_consumer",
            source_symbol="RuntimeEvent.tenant_id",
            target_symbol="ModelContextAttributionEvidence",
            tenant_source="context + llm event tenants",
            tenant_target="evidence.tenant_id",
            relation="SAME",
            mismatch_behavior="tenant_mismatch before evidence emission",
            fail_closed=True,
            verdict=v3,
            evidence=ev3,
        ),
        R4ChainTransition(
            chain_id="chain3.evidence_events",
            transition_id="c3.trace_observational_only",
            source_symbol="trace / diagnostic records",
            target_symbol="attribution verdict",
            tenant_source="observed event tenant",
            tenant_target="comparison only",
            relation="DERIVED",
            mismatch_behavior="no tenant mint / repair",
            fail_closed=True,
            verdict=v3,
            evidence=ev3,
        ),
    )


def r4_chain_missing_count() -> int:
    return sum(1 for t in build_r4_e2e_chain() if t.verdict == R4ChainVerdict.MISSING)


def r4_chain_invalid_count() -> int:
    return sum(1 for t in build_r4_e2e_chain() if t.verdict == R4ChainVerdict.INVALID)


def context_model_chain_proven() -> bool:
    chain = build_r4_e2e_chain()
    c1 = [t for t in chain if t.chain_id == "chain1.context_model"]
    return bool(c1) and all(t.verdict == R4ChainVerdict.PROVEN_CURRENT_HEAD for t in c1)
