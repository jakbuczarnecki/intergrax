# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime/Governance execution policy admission contracts (AW-5A corrective).

Independent runtime policy evaluation for trusted root execution authority minting.
Collaborative Work ``effective_authority_decision`` is evidence only — not final policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.execution_authority import validate_authority_scopes
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

WORKER_ROOT_EXECUTION_OPERATION: Final = "worker.root_execution.dispatch"


class RootExecutionAdmissionPolicyRule:
    """Immutable runtime rule for ``RuntimePolicyEngine.evaluate_root_execution_admission``."""

    __slots__ = ("rule_id", "decision", "execution_operation", "approved_scopes", "reason")

    def __init__(
        self,
        *,
        rule_id: str,
        decision: PolicyAction,
        execution_operation: str | None = None,
        approved_scopes: tuple[str, ...] | None = None,
        reason: str = "",
    ) -> None:
        normalized_rule_id = rule_id.strip()
        if not normalized_rule_id:
            raise ValueError("rule_id must be non-empty")
        if not isinstance(decision, PolicyAction):
            raise TypeError("decision must be PolicyAction")
        if execution_operation is not None and not execution_operation.strip():
            raise ValueError("execution_operation must be non-empty when provided")
        if approved_scopes is not None:
            approved_scopes = validate_authority_scopes(approved_scopes)
        self.rule_id = normalized_rule_id
        self.decision = decision
        self.execution_operation = (
            execution_operation.strip() if execution_operation is not None else None
        )
        self.approved_scopes = approved_scopes
        self.reason = reason.strip()


@dataclass(frozen=True, slots=True)
class RuntimeExecutionPolicyAdmissionRequest:
    """Trusted inputs for independent runtime execution policy evaluation."""

    tenant_id: str
    workspace_id: str
    principal_id: str
    collaborative_authority_scopes: tuple[str, ...]
    execution_operation: str = WORKER_ROOT_EXECUTION_OPERATION
    resource_scope: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if not self.workspace_id.strip():
            raise ValueError("workspace_id must be non-empty")
        if not self.principal_id.strip():
            raise ValueError("principal_id must be non-empty")
        if not self.execution_operation.strip():
            raise ValueError("execution_operation must be non-empty")
        object.__setattr__(
            self,
            "collaborative_authority_scopes",
            validate_authority_scopes(self.collaborative_authority_scopes),
        )
        if self.resource_scope is not None and not self.resource_scope.strip():
            raise ValueError("resource_scope must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class RuntimeExecutionPolicyAdmissionResult:
    """Runtime policy outcome — approved scopes may narrow collaborative scopes."""

    policy_decision: PolicyDecision
    approved_scopes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.policy_decision) is not PolicyDecision:
            raise TypeError("policy_decision must be PolicyDecision")
        if self.approved_scopes is not None:
            object.__setattr__(
                self,
                "approved_scopes",
                validate_authority_scopes(self.approved_scopes),
            )


class RuntimeExecutionPolicyAvailability(StrEnum):
    """Evaluator configuration availability."""

    CONFIGURED = "CONFIGURED"
    UNAVAILABLE = "UNAVAILABLE"


@runtime_checkable
class RuntimeExecutionPolicyAdmissionPort(Protocol):
    """Independent runtime/governance policy admission owned by Runtime/Governance."""

    def evaluate(
        self,
        request: RuntimeExecutionPolicyAdmissionRequest,
    ) -> RuntimeExecutionPolicyAdmissionResult:
        """Evaluate applicable runtime policy using trusted request inputs only."""
        ...
