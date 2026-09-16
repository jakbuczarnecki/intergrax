# © Artur Czarnecki. All rights reserved.

"""Shared governance helpers for specialized memory mutation paths (MEM-ENT-10B)."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityRelationRecord,
)
from intergrax.memory.contracts.long_horizon_memory import LongHorizonSummaryRecord
from intergrax.memory.contracts.memory_control import MemoryControlPlaneScope, MemoryControlScopeRef
from intergrax.memory.contracts.memory_models import MemoryKind, UserProfileMemoryEntry
from intergrax.memory.contracts.entity_temporal_memory import EntityMemoryScope
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceSourceAuthority,
    CanonicalMemoryGovernanceSourceSnapshot,
    CanonicalMemoryGovernanceSourceViolation,
    MemoryGovernanceDecision,
    MemoryGovernanceDenied,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
    MemoryGovernanceRecordSnapshot,
    MemoryGovernanceTarget,
    MemorySecurityContext,
    validate_canonical_governance_source_snapshot,
)
from intergrax.memory.contracts.procedural_memory import ProcedureRecord
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService

__all__ = [
    "enforce_specialized_memory_mutation",
    "governance_snapshot_from_canonical_source",
    "governance_snapshot_from_entity_record",
    "governance_snapshot_from_long_horizon_summary",
    "governance_snapshot_from_procedure_record",
    "governance_source_snapshot_from_user_entry",
    "memory_control_scope_from_entity_scope",
    "memory_security_context_for_mutation",
    "resolve_governance_source_record_snapshot",
    "specialized_mutation_denied_for_canonical_source",
]


def memory_control_scope_from_entity_scope(scope: EntityMemoryScope) -> MemoryControlScopeRef:
    user_id = (scope.user_id or "").strip()
    if not user_id:
        raise MemoryGovernanceDenied(
            "entity memory mutation requires scoped user_id",
            decision=_cross_scope_decision(MemoryGovernanceOperation.PROJECT),
        )
    return MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.USER,
        tenant_id=scope.tenant_id,
        user_id=user_id,
    )


def memory_security_context_for_mutation(
    identity: RequestIdentity,
    scope: EntityMemoryScope,
    operation: MemoryGovernanceOperation,
) -> MemorySecurityContext:
    control_scope = memory_control_scope_from_entity_scope(scope)
    return MemorySecurityContext(
        identity=identity,
        scope=control_scope,
        operation=operation,
    )


def governance_snapshot_from_entity_record(record: EntityRecord) -> MemoryGovernanceRecordSnapshot:
    memory_id = (record.source_memory_id or record.entity_id).strip()
    preview = record.canonical_name[:256] if record.canonical_name else None
    kind_value = record.entity_type.value
    try:
        kind = MemoryKind(kind_value)
    except ValueError:
        kind = MemoryKind.OTHER
    return MemoryGovernanceRecordSnapshot(
        memory_id=memory_id,
        revision=record.revision,
        kind=kind,
        provenance=record.provenance,
        trust=record.trust,
        governance=record.governance,
        content_preview=preview,
    )


def governance_snapshot_from_entity_relation(
    record: EntityRelationRecord,
) -> MemoryGovernanceRecordSnapshot:
    memory_id = (record.source_memory_id or record.relation_id).strip()
    return MemoryGovernanceRecordSnapshot(
        memory_id=memory_id,
        revision=record.revision,
        kind=MemoryKind.OTHER,
        provenance=record.provenance,
        trust=record.trust,
        governance=record.governance,
        content_preview=None,
    )


def governance_snapshot_from_canonical_source(
    snapshot: CanonicalMemoryGovernanceSourceSnapshot,
    *,
    content_preview: str | None = None,
) -> MemoryGovernanceRecordSnapshot:
    kind = snapshot.kind if snapshot.kind is not None else MemoryKind.OTHER
    return MemoryGovernanceRecordSnapshot(
        memory_id=snapshot.memory_id,
        revision=snapshot.revision,
        kind=kind,
        provenance=snapshot.provenance,
        trust=snapshot.trust,
        governance=snapshot.governance,
        content_preview=content_preview,
    )


def specialized_mutation_denied_for_canonical_source(
    operation: MemoryGovernanceOperation,
    message: str,
) -> MemoryGovernanceDenied:
    decision = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
        policy_id="memory.canonical_governance_source",
        policy_version="1.0.0",
        operation=operation,
    )
    return MemoryGovernanceDenied(message, decision=decision)


def resolve_governance_source_record_snapshot(
    authority: CanonicalMemoryGovernanceSourceAuthority,
    scope: EntityMemoryScope,
    memory_id: str,
    revision: int,
    *,
    operation: MemoryGovernanceOperation,
    content_preview: str | None = None,
) -> MemoryGovernanceRecordSnapshot:
    try:
        canonical = authority.resolve_canonical_governance_source(scope, memory_id, revision)
    except CanonicalMemoryGovernanceSourceViolation as exc:
        raise specialized_mutation_denied_for_canonical_source(operation, str(exc)) from exc
    except Exception as exc:
        raise specialized_mutation_denied_for_canonical_source(
            operation,
            f"canonical governance source resolution failed: {exc}",
        ) from exc
    validate_canonical_governance_source_snapshot(memory_id, revision, canonical)
    return governance_snapshot_from_canonical_source(canonical, content_preview=content_preview)


def governance_snapshot_from_procedure_record(
    record: ProcedureRecord,
) -> MemoryGovernanceRecordSnapshot:
    preview = record.title[:256] if record.title else None
    return MemoryGovernanceRecordSnapshot(
        memory_id=record.source_memory_id,
        revision=record.source_memory_revision,
        kind=MemoryKind.PROCEDURAL,
        provenance=record.provenance,
        trust=record.trust,
        governance=record.governance,
        content_preview=preview,
    )


def governance_snapshot_from_long_horizon_summary(
    record: LongHorizonSummaryRecord,
) -> MemoryGovernanceRecordSnapshot:
    preview = record.content[:256] if record.content else None
    return MemoryGovernanceRecordSnapshot(
        memory_id=record.summary_id,
        revision=record.revision,
        kind=MemoryKind.OTHER,
        provenance=record.provenance,
        trust=record.trust,
        governance=record.governance,
        content_preview=preview,
    )


def _cross_scope_decision(operation: MemoryGovernanceOperation) -> MemoryGovernanceDecision:
    return MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=MemoryGovernanceReasonCode.CROSS_SCOPE,
        policy_id="memory.specialized.scope",
        policy_version="1.0.0",
        operation=operation,
    )


def enforce_specialized_memory_mutation(
    governance: MemorySecurityGovernanceService,
    request: MemoryGovernanceEvaluationRequest,
) -> None:
    decision = governance.evaluate(request)
    if not decision.permits_mutation():
        raise MemoryGovernanceDenied(
            f"memory governance denied: {decision.reason_code.value}",
            decision=decision,
        )


def governance_target_for_procedure(procedure_id: str) -> MemoryGovernanceTarget:
    return MemoryGovernanceTarget(memory_id=procedure_id)


def governance_source_snapshot_from_user_entry(
    entry: UserProfileMemoryEntry,
) -> MemoryGovernanceRecordSnapshot:
    return MemoryGovernanceRecordSnapshot.from_user_profile_entry(entry)
