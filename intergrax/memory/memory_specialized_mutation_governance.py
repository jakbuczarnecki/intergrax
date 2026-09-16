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
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDenied,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceRecordSnapshot,
    MemoryGovernanceTarget,
    MemorySecurityContext,
)
from intergrax.memory.contracts.procedural_memory import ProcedureRecord
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService

__all__ = [
    "enforce_specialized_memory_mutation",
    "governance_snapshot_from_entity_record",
    "governance_snapshot_from_long_horizon_summary",
    "governance_snapshot_from_procedure_record",
    "memory_control_scope_from_entity_scope",
    "memory_security_context_for_mutation",
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
    from intergrax.memory.contracts.memory_security_governance import (
        MemoryGovernanceDecision,
        MemoryGovernanceOutcome,
        MemoryGovernanceReasonCode,
    )

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
