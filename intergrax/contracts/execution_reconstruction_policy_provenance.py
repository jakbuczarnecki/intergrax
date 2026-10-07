# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed policy decision provenance reconstructed from canonical RuntimeEvents."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import AttemptId, EventId, ExecutionId, RunId, TaskId


@dataclass(frozen=True, slots=True)
class ReconstructedPolicyDecisionProvenance:
    """One POLICY_DECISION event projected for factual reconstruction — not permission."""

    event_id: EventId
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    evaluation_point: str
    evidence_id: str
    policy_bundle_id: str
    policy_bundle_version: str
    policy_bundle_digest: str
    policy_rule_id: str
    decision: str
    action: str


__all__ = ["ReconstructedPolicyDecisionProvenance"]
