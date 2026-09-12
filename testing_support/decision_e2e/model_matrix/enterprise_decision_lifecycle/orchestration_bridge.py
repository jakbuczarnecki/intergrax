# © Artur Czarnecki. All rights reserved.

"""Maps production orchestration results into lifecycle records (DS-E2E-15J-L7)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleActorRef,
    DecisionLifecycleRecord,
    DecisionSourceKind,
    DecisionSourceReference,
    DecisionType,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.engine import (
    DecisionLifecycleEngine,
)
from testing_support.decision_e2e.model_matrix.production_decision_orchestration.contracts import (
    DecisionOrchestrationResult,
)

_ORCHESTRATION_ACTOR = DecisionLifecycleActorRef(
    actor_kind="orchestration",
    reference_id="production_decision_orchestration",
)


def source_references_from_orchestration_result(
    result: DecisionOrchestrationResult,
) -> tuple[DecisionSourceReference, ...]:
    refs: list[DecisionSourceReference] = [
        DecisionSourceReference(
            source_kind=DecisionSourceKind.ORCHESTRATION,
            reference_id=result.lifecycle_metadata.orchestration_task_id,
        ),
        DecisionSourceReference(
            source_kind=DecisionSourceKind.MODEL_SELECTION,
            reference_id=result.selection_result.decision_metadata.selection_task_id,
        ),
    ]
    governance = result.governance_result
    if governance is not None:
        refs.append(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.GOVERNANCE,
                reference_id=governance.audit_metadata.decision_id,
            )
        )
    execution = result.execution_result_reference
    if execution is not None:
        refs.append(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.EXECUTION,
                reference_id=execution.execution_reference_id,
            )
        )
    return tuple(refs)


def lifecycle_record_from_orchestration_result(
    engine: DecisionLifecycleEngine,
    result: DecisionOrchestrationResult,
    *,
    decision_type: DecisionType = DecisionType.PRODUCTION_MODEL_ROUTING,
    reason: str = "orchestration_outcome_recorded",
) -> DecisionLifecycleRecord:
    return engine.begin_decision(
        decision_type=decision_type,
        source_references=source_references_from_orchestration_result(result),
        actor=_ORCHESTRATION_ACTOR,
        reason=reason,
    )


__all__ = [
    "lifecycle_record_from_orchestration_result",
    "source_references_from_orchestration_result",
]
