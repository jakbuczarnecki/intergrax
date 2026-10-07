# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Project POLICY_DECISION runtime events into typed reconstruction provenance."""

from __future__ import annotations

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.contracts.execution_reconstruction_models import (
    ExecutionReconstructionIntegrityError,
)
from intergrax.contracts.execution_reconstruction_policy_provenance import (
    ReconstructedPolicyDecisionProvenance,
)
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.payloads.spine_families import PolicyDecisionSpinePayloadV1
from intergrax.runtime.events.spine_payload_codec import legacy_spine_payload_to_typed


def project_policy_decision_provenance(
    positioned: tuple[PositionedRuntimeEvent, ...],
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
) -> tuple[ReconstructedPolicyDecisionProvenance, ...]:
    """Derive policy provenance in canonical positioned-event order."""
    projected: list[ReconstructedPolicyDecisionProvenance] = []
    for row in positioned:
        event = row.event
        if event.event_type is not RuntimeEventType.POLICY_DECISION:
            continue
        if event.tenant_id != tenant_id:
            raise ExecutionReconstructionIntegrityError(
                "policy decision event tenant mismatch in reconstruction scope"
            )
        if event.task_id != task_id or event.run_id != run_id:
            raise ExecutionReconstructionIntegrityError(
                "policy decision event run scope mismatch in reconstruction boundary"
            )
        typed, _promote = legacy_spine_payload_to_typed(
            RuntimeEventType.POLICY_DECISION,
            dict(event.payload),
        )
        if not isinstance(typed, PolicyDecisionSpinePayloadV1):
            raise ExecutionReconstructionIntegrityError(
                "policy decision payload could not be decoded to typed spine payload"
            )
        _require_non_empty(typed.evidence_id, field="evidence_id")
        _require_non_empty(typed.evaluation_point, field="evaluation_point")
        _require_non_empty(typed.policy_bundle_id, field="policy_bundle_id")
        _require_non_empty(typed.policy_bundle_version, field="policy_bundle_version")
        _require_non_empty(typed.policy_bundle_digest, field="policy_bundle_digest")
        _require_non_empty(typed.policy_rule_id, field="policy_rule_id")
        _require_non_empty(typed.decision, field="decision")
        _require_non_empty(typed.action, field="action")
        projected.append(
            ReconstructedPolicyDecisionProvenance(
                event_id=event.event_id,
                tenant_id=tenant_id,
                task_id=event.task_id,
                run_id=event.run_id,
                attempt_id=event.attempt_id,
                execution_id=event.execution_id,
                evaluation_point=typed.evaluation_point,
                evidence_id=typed.evidence_id,
                policy_bundle_id=typed.policy_bundle_id,
                policy_bundle_version=typed.policy_bundle_version,
                policy_bundle_digest=typed.policy_bundle_digest,
                policy_rule_id=typed.policy_rule_id,
                decision=typed.decision,
                action=typed.action,
            )
        )
    return tuple(projected)


def _require_non_empty(value: str, *, field: str) -> None:
    if not value.strip():
        raise ExecutionReconstructionIntegrityError(
            f"policy decision provenance missing required field: {field}"
        )


__all__ = ["project_policy_decision_provenance"]
