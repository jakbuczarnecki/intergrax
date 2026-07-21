# © Artur Czarnecki. All rights reserved.

"""Atomic GovernedExecutionResult (PC-4).

Immutable composition of evaluated policy, provider invocation/outcome, and
descriptive proof for one governed side effect. Host attestation consumes this
object — not loose proof/decision/metadata fragments.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.evaluated_policy_decision import EvaluatedPolicyDecision
from intergrax.contracts.governed_proof import GovernedProofProfile
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.contracts.runtime_policy import PolicyAction

SCHEMA_GOVERNED_EXECUTION_RESULT_V1: Final = "governed_execution_result.v1"
_NON_EMPTY = Field(min_length=1)

# Domain action → provider operation (External Work).
_ACTION_TO_OPERATION: dict[str, str] = {
    "CREATE_EXTERNAL_WORK": "create_work",
    "ACCEPT_QUOTE": "submit_quote_acceptance",
    "CANCEL_EXTERNAL_WORK": "cancel_work",
}


class GovernedExecutionResult(BaseModel):
    """Single atomic post-execution result for host attestation / recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governed_execution_result.v1"] = (
        SCHEMA_GOVERNED_EXECUTION_RESULT_V1
    )
    execution_id: str = _NON_EMPTY
    task_id: str = _NON_EMPTY
    run_id: str = _NON_EMPTY
    principal_id: str = _NON_EMPTY
    tenant_id: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None
    action: str = _NON_EMPTY
    evaluated_policy_decision: EvaluatedPolicyDecision
    provider_invocation: ProviderInvocation
    provider_outcome: ProviderInvocationOutcome
    proof: GovernedProofProfile
    execution_started_at: datetime
    execution_completed_at: datetime

    @field_validator(
        "execution_id",
        "task_id",
        "run_id",
        "principal_id",
        "action",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _assert_atomic_consistency(self) -> GovernedExecutionResult:
        inv = self.provider_invocation
        out = self.provider_outcome
        proof = self.proof
        ev = self.evaluated_policy_decision
        decision = ev.decision

        if inv.task_id != self.task_id or proof.task_id != self.task_id:
            raise ValueError("task_id_inconsistent")
        if inv.run_id != self.run_id or proof.run_id != self.run_id:
            raise ValueError("run_id_inconsistent")
        if proof.principal_id != self.principal_id:
            raise ValueError("principal_id_inconsistent")
        if proof.provider_id != inv.provider_id:
            raise ValueError("provider_id_inconsistent")
        if proof.action != self.action:
            raise ValueError("action_inconsistent_with_proof")
        if decision.action is not PolicyAction.ALLOW:
            raise ValueError("governed_execution_requires_allow")
        if proof.policy_action is not PolicyAction.ALLOW:
            raise ValueError("proof_requires_allow")
        if proof.policy_action is not decision.action:
            raise ValueError("proof_policy_action_mismatch")
        if (
            proof.policy_rule_id.strip()
            and proof.policy_rule_id != ev.matched_rule_id
        ):
            raise ValueError("proof_policy_rule_mismatch")
        if (
            decision.policy_bundle_id != ev.bundle_id
            or decision.policy_bundle_version != ev.bundle_version
            or decision.policy_bundle_digest != ev.bundle_digest
        ):
            raise ValueError("policy_bundle_identity_inconsistent")
        if out.invocation_id != inv.invocation_id:
            raise ValueError("invocation_id_outcome_mismatch")
        if out.status is not ProviderInvocationStatus.SUCCEEDED:
            raise ValueError("governed_execution_requires_succeeded_outcome")
        expected_op = _ACTION_TO_OPERATION.get(self.action)
        if expected_op is not None and inv.operation != expected_op:
            raise ValueError("action_operation_mismatch")
        if self.correlation_id and inv.correlation_id and (
            self.correlation_id != inv.correlation_id
            or (
                proof.correlation_id
                and proof.correlation_id != self.correlation_id
            )
        ):
            raise ValueError("correlation_id_inconsistent")
        if self.idempotency_key and inv.idempotency_key and (
            self.idempotency_key != inv.idempotency_key
            or (
                proof.idempotency_key
                and proof.idempotency_key != self.idempotency_key
            )
        ):
            raise ValueError("idempotency_key_inconsistent")
        if self.execution_completed_at < self.execution_started_at:
            raise ValueError("execution_timestamps_inverted")
        return self
