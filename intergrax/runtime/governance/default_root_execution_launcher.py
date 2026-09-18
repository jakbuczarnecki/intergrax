# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default root execution launcher — admission then canonical intake only (GR-2-R3)."""

from __future__ import annotations

from typing import Generic, TypeVar

from intergrax.contracts.execution_intake import (
    CanonicalExecutionIntakePort,
    CanonicalExecutionIntakeRequest,
)
from intergrax.contracts.root_execution_launch import (
    RootExecutionLaunchDisposition,
    RootExecutionLaunchPort,
    RootExecutionLaunchRequest,
    RootExecutionLaunchResult,
)
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionDisposition,
    RootExecutionAuthorityAdmissionPort,
    RootExecutionAuthorityAdmissionRequest,
)

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")

_DISPOSITION_MAP: dict[
    RootExecutionAuthorityAdmissionDisposition,
    RootExecutionLaunchDisposition,
] = {
    RootExecutionAuthorityAdmissionDisposition.ALLOWED: RootExecutionLaunchDisposition.LAUNCHED,
    RootExecutionAuthorityAdmissionDisposition.DENIED: RootExecutionLaunchDisposition.DENIED,
    RootExecutionAuthorityAdmissionDisposition.REQUIRE_HUMAN: (
        RootExecutionLaunchDisposition.REQUIRE_HUMAN
    ),
    RootExecutionAuthorityAdmissionDisposition.ESCALATE: RootExecutionLaunchDisposition.ESCALATE,
    RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE: RootExecutionLaunchDisposition.UNAVAILABLE,
}


class DefaultRootExecutionLauncher(
    RootExecutionLaunchPort[PayloadT, ResultT],
    Generic[PayloadT, ResultT],
):
    """Thin orchestration: governance admission → trusted authority → intake."""

    __slots__ = ("_root_authority_admission", "_execution_intake")

    def __init__(
        self,
        *,
        root_authority_admission: RootExecutionAuthorityAdmissionPort,
        execution_intake: CanonicalExecutionIntakePort[PayloadT, ResultT],
    ) -> None:
        self._root_authority_admission = root_authority_admission
        self._execution_intake = execution_intake

    async def launch(
        self,
        request: RootExecutionLaunchRequest[PayloadT],
    ) -> RootExecutionLaunchResult[ResultT]:
        admission = self._root_authority_admission.authorize(
            RootExecutionAuthorityAdmissionRequest(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                principal_id=request.principal_id,
                collaborative_authority_scopes=request.collaborative_authority_scopes,
                effective_authority_decision=request.effective_authority_decision,
                root_execution_operation=request.root_execution_operation,
                task_id=request.task_id,
                run_id=request.run_id,
                attempt_id=request.attempt_id,
                execution_id=request.execution_id,
            )
        )
        launch_disposition = _DISPOSITION_MAP.get(
            admission.disposition,
            RootExecutionLaunchDisposition.UNAVAILABLE,
        )
        if launch_disposition is not RootExecutionLaunchDisposition.LAUNCHED:
            return RootExecutionLaunchResult(disposition=launch_disposition)
        assert admission.trusted_parent_execution_authority is not None
        intake_result = await self._execution_intake.dispatch(
            CanonicalExecutionIntakeRequest(
                payload=request.payload,
                trusted_parent_execution_authority=admission.trusted_parent_execution_authority,
                admitted_governance_identity=request.admitted_governance_identity,
                run_id=request.run_id,
                attempt_id=request.attempt_id,
                execution_id=request.execution_id,
                task_id=request.task_id,
                segment_predecessor_root_execution_id=request.segment_predecessor_root_execution_id,
            )
        )
        return RootExecutionLaunchResult(
            disposition=RootExecutionLaunchDisposition.LAUNCHED,
            intake_result=intake_result,
        )
