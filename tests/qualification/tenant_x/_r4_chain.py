# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R4 E2E continuity chains (per-transition current-HEAD proofs)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present
from tests.qualification.tenant_x._r4_call_sites import (
    context_assembly_production_path_closed,
    llm_call_production_tenant_closed,
)
from tests.qualification.tenant_x._r4_types import R4ChainTransition, R4ChainVerdict

_C1_ENGINE: Final[tuple[tuple[str, str], ...]] = (
    ("intergrax/runtime/events/context_skill_recording.py", "record_context_assembled_from_engine"),
    ("intergrax/runtime/events/context_skill_recording.py", "peek_active_runtime_event_tenant_id"),
    ("intergrax/runtime/nexus/context/context_engine.py", "record_context_assembled_from_engine"),
)

_C1_METADATA: Final[tuple[tuple[str, str], ...]] = (
    ("intergrax/runtime/nexus/context/context_manager.py", '"tenant_id": task.tenant_id'),
    ("intergrax/runtime/events/context_skill_recording.py", "metadata.get(\"tenant_id\")"),
)

_C1_CONTEXT_TO_LLM: Final[tuple[tuple[str, str], ...]] = (
    ("intergrax/runtime/events/context_skill_recording.py", "bind_pending_context_assembly_event_id"),
    ("intergrax/runtime/events/llm_call_recording.py", "peek_pending_context_assembly_event_id"),
    ("intergrax/runtime/events/llm_call_recording.py", "context_assembly_event_id"),
)

_C1_LLM_TO_ATTR: Final[tuple[tuple[str, str], ...]] = (
    ("intergrax/runtime/llm/model_context_attribution.py", "context_event_id_mismatch"),
    ("intergrax/runtime/llm/model_context_attribution.py", "model_input_hash_mismatch"),
    ("intergrax/runtime/llm/model_context_attribution.py", "execution_identity_mismatch"),
    ("intergrax/runtime/llm/model_context_attribution.py", "tenant_mismatch"),
    ("intergrax/runtime/llm/model_context_attribution.py", "tenant_id=llm_event.tenant_id"),
)

_C2_ENQUEUE: Final[tuple[tuple[str, str], ...]] = (
    ("intergrax/runtime/vendor_knowledge/sync_runtime.py", "vendor knowledge task queue tenant mismatch"),
    ("intergrax/queueing/contracts/task_queue.py", "tenant_id"),
)

_C2_WORKER_INTAKE: Final[tuple[tuple[str, str], ...]] = (
    ("intergrax/runtime/vendor_knowledge/sync_worker.py", "vendor_knowledge_sync_tenant_mismatch"),
    ("intergrax/queueing/providers/document_store/document_store_task_queue.py", "tenant_id=request.tenant_id"),
)

_C2_BG_IDENTITY: Final[tuple[tuple[str, str], ...]] = (
    ("intergrax/runtime/background_execution/bootstrap.py", "BackgroundExecutionTenantMismatchError"),
    ("intergrax/runtime/background_execution/bootstrap.py", "resolve_background_execution"),
)

_C2_IDENTITY_TO_EXEC: Final[tuple[tuple[str, str], ...]] = (
    ("intergrax/queueing/worker/execution.py", "resolved_tenant_id = execution_identity.tenant_id"),
    ("intergrax/queueing/worker/execution.py", "execution_identity.tenant_id"),
)

_C3_FUNCTIONAL: Final[tuple[tuple[str, str], ...]] = (
    ("intergrax/contracts/functional_evidence/tenant.py", "require_tenant_id_from_exec_ctx"),
    ("intergrax/contracts/functional_evidence/tenant.py", "canonical_request_identity"),
)

_C3_EVENTS_ATTR: Final[tuple[tuple[str, str], ...]] = (
    ("intergrax/runtime/llm/model_context_attribution.py", "try_attribute_model_call_to_context"),
    ("intergrax/runtime/llm/model_context_attribution.py", "ModelContextAttributionEvidence"),
)

_C3_TRACE_OBS: Final[tuple[tuple[str, str], ...]] = (
    ("intergrax/runtime/llm/model_context_attribution.py", "left_tenant = context_event.tenant_id"),
    ("intergrax/runtime/llm/model_context_attribution.py", "right_tenant = llm_event.tenant_id"),
)


def _verdict_from_checks(
    checks: tuple[tuple[str, str], ...],
    *,
    discovery_note: str,
    extra_gate: bool = True,
) -> tuple[R4ChainVerdict, tuple[str, ...]]:
    matched = all_structural_patterns_present(checks)
    if not extra_gate or len(matched) != len(checks):
        return R4ChainVerdict.MISSING, (discovery_note,)
    return R4ChainVerdict.PROVEN_CURRENT_HEAD, matched


def build_r4_e2e_chain() -> tuple[R4ChainTransition, ...]:
    v1a, ev1a = _verdict_from_checks(
        _C1_ENGINE,
        discovery_note="engine→CONTEXT_ASSEMBLED structural proof incomplete",
        extra_gate=context_assembly_production_path_closed(),
    )
    v1b, ev1b = _verdict_from_checks(
        _C1_METADATA,
        discovery_note="Task.tenant_id→metadata→record_context_assembly proof incomplete",
        extra_gate=context_assembly_production_path_closed(),
    )
    v1c, ev1c = _verdict_from_checks(
        _C1_CONTEXT_TO_LLM,
        discovery_note="context event_id→LLM_CALL linkage incomplete",
        extra_gate=llm_call_production_tenant_closed(),
    )
    v1d, ev1d = _verdict_from_checks(
        _C1_LLM_TO_ATTR,
        discovery_note="LLM→attribution structural checks incomplete",
    )

    v2a, ev2a = _verdict_from_checks(_C2_ENQUEUE, discovery_note="enqueue→TaskRequest tenant incomplete")
    v2b, ev2b = _verdict_from_checks(_C2_WORKER_INTAKE, discovery_note="worker intake tenant incomplete")
    v2c, ev2c = _verdict_from_checks(_C2_BG_IDENTITY, discovery_note="background identity reconcile incomplete")
    v2d, ev2d = _verdict_from_checks(
        _C2_IDENTITY_TO_EXEC,
        discovery_note="BackgroundExecutionIdentity→handler tenant incomplete",
    )

    v3a, ev3a = _verdict_from_checks(_C3_FUNCTIONAL, discovery_note="exec_ctx→functional evidence incomplete")
    v3b, ev3b = _verdict_from_checks(_C3_EVENTS_ATTR, discovery_note="events→attribution evidence incomplete")
    v3c, ev3c = _verdict_from_checks(_C3_TRACE_OBS, discovery_note="trace observational compare incomplete")

    return (
        R4ChainTransition(
            chain_id="chain1.context_model",
            transition_id="c1.engine_to_context_assembly",
            source_symbol="active runtime tenant scope",
            target_symbol="record_context_assembled_from_engine",
            tenant_source="peek_active_runtime_event_tenant_id",
            tenant_target="RuntimeEvent.tenant_id on CONTEXT_ASSEMBLED",
            relation="SAME",
            mismatch_behavior="missing active tenant → event tenant_id may be None; attribution fail-closed",
            fail_closed=True,
            verdict=v1a,
            evidence=ev1a,
        ),
        R4ChainTransition(
            chain_id="chain1.context_model",
            transition_id="c1.task_metadata_to_context_assembly",
            source_symbol="Task.tenant_id",
            target_symbol="record_context_assembly",
            tenant_source="metadata['tenant_id'] after **bundle_metadata",
            tenant_target="RuntimeEvent.tenant_id on CONTEXT_ASSEMBLED",
            relation="SAME",
            mismatch_behavior="raw helper accepts caller metadata; production binds task.tenant_id last",
            fail_closed=True,
            verdict=v1b,
            evidence=ev1b,
        ),
        R4ChainTransition(
            chain_id="chain1.context_model",
            transition_id="c1.context_event_to_llm_event",
            source_symbol="CONTEXT_ASSEMBLED.event_id",
            target_symbol="LLM_CALL.context_assembly_event_id",
            tenant_source="context_event.tenant_id + pending bind scope",
            tenant_target="llm_event.tenant_id (explicit or active scope)",
            relation="SAME",
            mismatch_behavior="missing context ref on PRIMARY → ValueError; tenant_mismatch at attribution",
            fail_closed=True,
            verdict=v1c,
            evidence=ev1c,
        ),
        R4ChainTransition(
            chain_id="chain1.context_model",
            transition_id="c1.llm_to_attribution",
            source_symbol="LLM_CALL payload",
            target_symbol="try_attribute_model_call_to_context",
            tenant_source="llm_event.tenant_id",
            tenant_target="ModelContextAttributionEvidence.tenant_id",
            relation="SAME",
            mismatch_behavior="hash / event_id / identity / tenant mismatch → not attributable",
            fail_closed=True,
            verdict=v1d,
            evidence=ev1d,
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
            verdict=v2a,
            evidence=ev2a,
        ),
        R4ChainTransition(
            chain_id="chain2.async_queue_worker",
            transition_id="c2.worker_intake",
            source_symbol="TaskRequest / decoded job",
            target_symbol="worker handler tenant_id",
            tenant_source="request.tenant_id / job.tenant_id",
            tenant_target="handler tenant_id parameter",
            relation="SAME",
            mismatch_behavior="decode job tenant mismatch → fail result",
            fail_closed=True,
            verdict=v2b,
            evidence=ev2b,
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
            verdict=v2c,
            evidence=ev2c,
        ),
        R4ChainTransition(
            chain_id="chain2.async_queue_worker",
            transition_id="c2.identity_to_execution",
            source_symbol="BackgroundExecutionIdentity",
            target_symbol="handler / execution context",
            tenant_source="identity.tenant_id",
            tenant_target="execution_identity.tenant_id consumed by worker",
            relation="SAME",
            mismatch_behavior="no run_id tenant inference",
            fail_closed=True,
            verdict=v2d,
            evidence=ev2d,
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
            verdict=v3a,
            evidence=ev3a,
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
            verdict=v3b,
            evidence=ev3b,
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
            verdict=v3c,
            evidence=ev3c,
        ),
    )


def r4_chain_missing_count() -> int:
    return sum(1 for t in build_r4_e2e_chain() if t.verdict == R4ChainVerdict.MISSING)


def r4_chain_invalid_count() -> int:
    return sum(1 for t in build_r4_e2e_chain() if t.verdict == R4ChainVerdict.INVALID)


def r4_chain_proven_count() -> int:
    return sum(1 for t in build_r4_e2e_chain() if t.verdict == R4ChainVerdict.PROVEN_CURRENT_HEAD)


def _chain_slice_counts(chain_id: str) -> tuple[int, int, int]:
    rows = [t for t in build_r4_e2e_chain() if t.chain_id == chain_id]
    proven = sum(1 for t in rows if t.verdict == R4ChainVerdict.PROVEN_CURRENT_HEAD)
    missing = sum(1 for t in rows if t.verdict == R4ChainVerdict.MISSING)
    invalid = sum(1 for t in rows if t.verdict == R4ChainVerdict.INVALID)
    return proven, missing, invalid


def context_model_chain_counts() -> tuple[int, int, int]:
    return _chain_slice_counts("chain1.context_model")


def queue_worker_chain_counts() -> tuple[int, int, int]:
    return _chain_slice_counts("chain2.async_queue_worker")


def evidence_event_chain_counts() -> tuple[int, int, int]:
    return _chain_slice_counts("chain3.evidence_events")


def context_model_chain_proven() -> bool:
    proven, missing, invalid = context_model_chain_counts()
    return (
        proven > 0
        and missing == 0
        and invalid == 0
        and context_assembly_production_path_closed()
        and llm_call_production_tenant_closed()
    )
