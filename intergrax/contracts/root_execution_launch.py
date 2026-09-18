# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Root execution launch contracts — sole legal production root start (GR-2-R3, MODEL C1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.admitted_root_governance_identity import AdmittedRootGovernanceIdentity
from intergrax.contracts.autonomous_work.execution_authority import validate_authority_scopes
from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.execution_intake import CanonicalExecutionIntakeResult
from intergrax.contracts.root_execution_operation import (
    RootExecutionOperation,
    normalize_root_execution_policy_operation,
)

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


class RootExecutionLaunchDisposition(StrEnum):
    """Launcher outcome — only LAUNCHED performed canonical intake."""

    LAUNCHED = "LAUNCHED"
    DENIED = "DENIED"
    REQUIRE_HUMAN = "REQUIRE_HUMAN"
    ESCALATE = "ESCALATE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class RootExecutionLaunchRequest(Generic[PayloadT]):
    """Trusted launch inputs — caller must not supply admission-provenanced authority."""

    admitted_governance_identity: AdmittedRootGovernanceIdentity
    root_execution_operation: RootExecutionOperation
    collaborative_authority_scopes: tuple[str, ...]
    effective_authority_decision: EffectiveAuthorityDecision
    payload: PayloadT
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None
    task_id: TaskId | None = None
    segment_predecessor_root_execution_id: ExecutionId | None = None

    @property
    def tenant_id(self) -> str:
        return self.admitted_governance_identity.tenant_id

    @property
    def workspace_id(self) -> str:
        return self.admitted_governance_identity.workspace_id

    @property
    def principal_id(self) -> str:
        return self.admitted_governance_identity.principal_id

    def __post_init__(self) -> None:
        if type(self.admitted_governance_identity) is not AdmittedRootGovernanceIdentity:
            raise TypeError("admitted_governance_identity must be AdmittedRootGovernanceIdentity")
        if type(self.root_execution_operation) is not RootExecutionOperation:
            raise TypeError("root_execution_operation must be RootExecutionOperation")
        normalize_root_execution_policy_operation(self.root_execution_operation)
        object.__setattr__(
            self,
            "collaborative_authority_scopes",
            validate_authority_scopes(self.collaborative_authority_scopes),
        )
        if type(self.effective_authority_decision) is not EffectiveAuthorityDecision:
            raise TypeError("effective_authority_decision must be EffectiveAuthorityDecision")
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)
        if self.task_id is not None:
            validate_task_id(self.task_id)
        if self.segment_predecessor_root_execution_id is not None:
            validate_execution_id(self.segment_predecessor_root_execution_id)


@dataclass(frozen=True, slots=True)
class RootExecutionLaunchResult(Generic[ResultT]):
    disposition: RootExecutionLaunchDisposition
    intake_result: CanonicalExecutionIntakeResult[ResultT] | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not RootExecutionLaunchDisposition:
            raise TypeError("disposition must be RootExecutionLaunchDisposition")
        if self.disposition is RootExecutionLaunchDisposition.LAUNCHED:
            if self.intake_result is None:
                raise ValueError("LAUNCHED requires intake_result")
        elif self.intake_result is not None:
            raise ValueError("non-LAUNCHED launch must not expose intake_result")


@runtime_checkable
class RootExecutionLaunchPort(Protocol[PayloadT, ResultT]):
    """Mandatory production root execution entry — admission then intake."""

    async def launch(
        self,
        request: RootExecutionLaunchRequest[PayloadT],
    ) -> RootExecutionLaunchResult[ResultT]:
        """Run governance admission; on ALLOW only, dispatch through canonical intake."""
        ...
