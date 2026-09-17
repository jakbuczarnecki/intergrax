# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Governance evidence persistence adapters (GR-8) — Evidence Plane projection only."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.evaluated_policy_decision import request_digest_for_payload
from intergrax.contracts.execution_evidence.persistence_boundary_errors import (
    EvidencePersistenceBoundaryError,
)
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernanceDecisionEvidenceFact,
    GovernanceEvidencePersistenceOutcome,
    GovernanceEvidencePersistencePort,
    deterministic_runtime_event_id_for_governance_fact,
)
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_event_type import RuntimeEventType


@dataclass
class InMemoryGovernanceEvidencePersistence(GovernanceEvidencePersistencePort):
    """Test/reference store — typed facts only, no policy authority."""

    facts: list[GovernanceDecisionEvidenceFact] = field(default_factory=list)
    fail_on_persist: bool = False

    def persist(self, fact: GovernanceDecisionEvidenceFact) -> GovernanceEvidencePersistenceOutcome:
        if self.fail_on_persist:
            return GovernanceEvidencePersistenceOutcome(
                persisted=False,
                evidence_id=fact.evidence_id,
                error_code="GovernanceEvidencePersistenceError",
            )
        for existing in self.facts:
            if existing.evidence_id == fact.evidence_id:
                if existing == fact:
                    return GovernanceEvidencePersistenceOutcome(
                        persisted=True,
                        evidence_id=fact.evidence_id,
                    )
                return GovernanceEvidencePersistenceOutcome(
                    persisted=False,
                    evidence_id=fact.evidence_id,
                    error_code="GovernanceEvidenceIdempotencyConflict",
                )
        self.facts.append(fact)
        return GovernanceEvidencePersistenceOutcome(
            persisted=True,
            evidence_id=fact.evidence_id,
        )


def _fact_payload(fact: GovernanceDecisionEvidenceFact) -> dict[str, object]:
    payload: dict[str, object] = {
        "governance_evidence_schema": fact.schema_version,
        "evidence_id": fact.evidence_id,
        "evaluation_point": fact.evaluation_point.value,
        "action": fact.action,
        "resource_type": fact.resource_type,
        "resource_scope": fact.resource_scope,
        "decision": fact.decision.value,
        "reason": fact.reason,
        "reason_code": fact.reason_code,
        "policy_bundle_id": fact.policy_bundle_id,
        "policy_bundle_version": fact.policy_bundle_version,
        "policy_bundle_digest": fact.policy_bundle_digest,
        "policy_rule_id": fact.policy_rule_id,
        "request_digest": fact.request_digest,
        "idempotency_key": fact.idempotency_key,
        "workspace_id": fact.workspace_id,
        "principal_id": fact.principal_id,
    }
    if fact.decision_material_ref is not None:
        payload["decision_material_ref"] = fact.decision_material_ref.model_dump(mode="json")
    if fact.human_review_evidence_ref:
        payload["human_review_evidence_ref"] = fact.human_review_evidence_ref
    return payload


@dataclass(frozen=True, slots=True)
class RuntimeEventGovernanceEvidencePersistence(GovernanceEvidencePersistencePort):
    """Projects Governance facts onto canonical ``RuntimeEvent`` via ``EvidencePersistencePort``."""

    evidence_persistence: EvidencePersistencePort

    def persist(self, fact: GovernanceDecisionEvidenceFact) -> GovernanceEvidencePersistenceOutcome:
        if not fact.has_full_execution_correlation:
            return GovernanceEvidencePersistenceOutcome(
                persisted=False,
                evidence_id=fact.evidence_id,
                error_code="GovernanceEvidenceIncompleteCorrelation",
            )
        assert fact.task_id is not None
        assert fact.run_id is not None
        assert fact.attempt_id is not None
        assert fact.execution_id is not None
        event = RuntimeEvent(
            event_id=deterministic_runtime_event_id_for_governance_fact(fact),
            tenant_id=fact.tenant_id,
            task_id=fact.task_id,
            run_id=fact.run_id,
            attempt_id=fact.attempt_id,
            execution_id=fact.execution_id,
            event_type=RuntimeEventType.POLICY_DECISION,
            phase=ExecutionPhase.STEP_EXECUTION,
            timestamp=fact.recorded_at,
            payload=_fact_payload(fact),
        )
        try:
            self.evidence_persistence.append(event, tenant_id=fact.tenant_id)
        except EvidencePersistenceBoundaryError as exc:
            return GovernanceEvidencePersistenceOutcome(
                persisted=False,
                evidence_id=fact.evidence_id,
                error_code=type(exc).__name__,
            )
        return GovernanceEvidencePersistenceOutcome(
            persisted=True,
            evidence_id=fact.evidence_id,
        )


def governance_request_digest_for_admission(
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    execution_operation: str,
    resource_scope: str = "",
) -> str:
    return request_digest_for_payload(
        {
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "principal_id": principal_id,
            "execution_operation": execution_operation,
            "resource_scope": resource_scope,
        }
    )


__all__ = [
    "InMemoryGovernanceEvidencePersistence",
    "RuntimeEventGovernanceEvidencePersistence",
    "governance_request_digest_for_admission",
]
