# © Artur Czarnecki. All rights reserved.

"""Shared harness HTTP routes for task control (FLOW-CTL.2, REL-ADV.4)."""

from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from intergrax.applications._shared.async_task_dispatch import get_async_status, run_async
from intergrax.applications._shared.async_task_index_protocol import AsyncTaskIndexProtocol
from intergrax.applications._shared.harness_auth import (
    require_harness_api_key,
    resolve_harness_authenticated_principal,
)
from intergrax.applications._shared.harness_principal import (
    harness_principal_to_approver_evidence,
    harness_principal_to_request_identity,
    reject_identity_assertion_conflicts,
)
from intergrax.applications._shared.task_control import (
    HitlResumeValidationError,
    TaskControlValidationError,
    governed_cancel_active_task,
    governed_resume_checkpoint_task,
    governed_set_task_autonomy,
)
from intergrax.applications._shared.task_control_governance import (
    TaskControlGovernanceBlockedError,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


class HarnessAsyncRunRequest(BaseModel):
    tenant_id: str = "default"
    user_id: str = "user"
    message: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HarnessAutonomyRequest(BaseModel):
    mutation_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    autonomy_level: AutonomyLevel
    tenant_id: str | None = None
    approval_evidence_ref: str | None = None


class HarnessCancelRequest(BaseModel):
    mutation_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    reason: str = "operator_cancel"
    tenant_id: str | None = None
    approval_evidence_ref: str | None = None


class HarnessResumeRequest(BaseModel):
    mutation_id: str = Field(min_length=1)
    tenant_id: str | None = None
    resume_token: str = Field(min_length=1)
    operator_input: dict[str, Any] = Field(default_factory=dict)
    approval_evidence_ref: str | None = None


class HarnessTaskControlResponse(BaseModel):
    task_id: str
    action: str
    accepted: bool
    detail: str = ""
    state: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    authorization_evidence: dict[str, Any] | None = None
    authorization_scope: dict[str, Any] | None = None
    blocker_code: str | None = None
    policy_action: str | None = None


def _raise_task_control_http(exc: Exception) -> None:
    if isinstance(exc, TaskControlGovernanceBlockedError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=exc.governance_http_detail(),
        ) from exc
    if isinstance(exc, TaskControlValidationError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    raise exc


def _task_control_response(result) -> HarnessTaskControlResponse:
    evidence = result.authorization_evidence
    scope = result.authorization_scope
    return HarnessTaskControlResponse(
        task_id=result.task_id,
        action=result.action,
        accepted=result.accepted,
        detail=result.detail,
        state=result.state,
        metadata=dict(result.metadata or {}),
        authorization_evidence=evidence.model_dump(mode="json") if evidence is not None else None,
        authorization_scope=scope.model_dump(mode="json") if scope is not None else None,
        blocker_code=result.blocker_code,
        policy_action=result.policy_action,
    )


def mount_harness_task_routes(
    app: FastAPI,
    *,
    task_runner: UnifiedTaskRunner,
    prefix: str = "/v1/tasks",
    checkpoint_store: TaskCheckpointPersistence | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    task_enricher: Callable[[Task], Task] | None = None,
    async_index: AsyncTaskIndexProtocol | None = None,
    mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
) -> APIRouter:
    router = APIRouter(
        prefix=prefix,
        tags=["harness-tasks"],
        dependencies=[Depends(require_harness_api_key)],
    )

    @router.post("/run-async")
    async def run_async_route(body: HarnessAsyncRunRequest) -> dict[str, Any]:
        run_id = new_run_id()
        task = Task(
            task_id=run_id,
            tenant_id=body.tenant_id,
            user_id=body.user_id,
            message=body.message,
            context=TaskContext(capability=body.capability),
            metadata=dict(body.metadata),
        )
        if task_enricher is not None:
            task = task_enricher(task)
        return await run_async(task_runner, task, index=async_index)

    @router.get("/{task_id}/status")
    async def task_status(task_id: str) -> dict[str, Any]:
        return await get_async_status(task_id, index=async_index)

    @router.post("/{task_id}/cancel", response_model=HarnessTaskControlResponse)
    async def cancel_task(
        task_id: str,
        body: HarnessCancelRequest,
        _request: Request,
        principal=Depends(resolve_harness_authenticated_principal),
    ) -> HarnessTaskControlResponse:
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="authenticated_principal_required",
            )
        identity = harness_principal_to_request_identity(principal)
        if body.tenant_id is not None:
            reject_identity_assertion_conflicts(
                canonical=identity,
                asserted_tenant_id=body.tenant_id,
                asserted_user_id=None,
            )
        try:
            result = await governed_cancel_active_task(
                task_id=task_id,
                run_id=body.run_id,
                mutation_id=body.mutation_id,
                principal=identity,
                mutation_boundary=mutation_boundary,
                reason=body.reason,
                approval_evidence_ref=body.approval_evidence_ref,
            )
        except TaskControlGovernanceBlockedError as exc:
            _raise_task_control_http(exc)
        except TaskControlValidationError as exc:
            _raise_task_control_http(exc)
        if not result.accepted:
            if result.detail == "task_not_active":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result.detail,
                )
            if result.detail in {"run_id_mismatch", "stale_active_binding", "task_not_cancellable"}:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=result.detail,
                )
            if result.blocker_code is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "blocker_code": result.blocker_code,
                        "policy_action": result.policy_action,
                        "authorization_evidence": (
                            result.authorization_evidence.model_dump(mode="json")
                            if result.authorization_evidence is not None
                            else None
                        ),
                        "authorization_scope": (
                            result.authorization_scope.model_dump(mode="json")
                            if result.authorization_scope is not None
                            else None
                        ),
                    },
                )
        return _task_control_response(result)

    @router.post("/{task_id}/autonomy", response_model=HarnessTaskControlResponse)
    async def set_autonomy(
        task_id: str,
        body: HarnessAutonomyRequest,
        _request: Request,
        principal=Depends(resolve_harness_authenticated_principal),
    ) -> HarnessTaskControlResponse:
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="authenticated_principal_required",
            )
        identity = harness_principal_to_request_identity(principal)
        if body.tenant_id is not None:
            reject_identity_assertion_conflicts(
                canonical=identity,
                asserted_tenant_id=body.tenant_id,
                asserted_user_id=None,
            )
        try:
            result = await governed_set_task_autonomy(
                task_id=task_id,
                run_id=body.run_id,
                mutation_id=body.mutation_id,
                target_autonomy_level=body.autonomy_level,
                principal=identity,
                mutation_boundary=mutation_boundary,
                approval_evidence_ref=body.approval_evidence_ref,
            )
        except TaskControlGovernanceBlockedError as exc:
            _raise_task_control_http(exc)
        except TaskControlValidationError as exc:
            _raise_task_control_http(exc)
        if not result.accepted:
            if result.detail == "task_not_active":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result.detail,
                )
            if result.detail in {"run_id_mismatch", "stale_active_binding"}:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=result.detail,
                )
            if result.blocker_code is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "blocker_code": result.blocker_code,
                        "policy_action": result.policy_action,
                        "authorization_evidence": (
                            result.authorization_evidence.model_dump(mode="json")
                            if result.authorization_evidence is not None
                            else None
                        ),
                        "authorization_scope": (
                            result.authorization_scope.model_dump(mode="json")
                            if result.authorization_scope is not None
                            else None
                        ),
                    },
                )
        return _task_control_response(result)

    @router.post("/{task_id}/resume")
    async def resume_task(
        task_id: str,
        body: HarnessResumeRequest,
        _request: Request,
        principal=Depends(resolve_harness_authenticated_principal),
    ) -> dict[str, Any]:
        if checkpoint_store is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="checkpoint_store_not_configured",
            )
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="authenticated_principal_required",
            )
        identity = harness_principal_to_request_identity(principal)
        if body.tenant_id is not None:
            reject_identity_assertion_conflicts(
                canonical=identity,
                asserted_tenant_id=body.tenant_id,
                asserted_user_id=None,
            )
        approver = harness_principal_to_approver_evidence(principal)
        try:
            outcome = await governed_resume_checkpoint_task(
                task_runner,
                task_id=task_id,
                tenant_id=identity.tenant_id,
                resume_token=body.resume_token,
                mutation_id=body.mutation_id,
                principal=identity,
                mutation_boundary=mutation_boundary,
                checkpoint_store=checkpoint_store,
                operator_input=body.operator_input,
                approver=approver,
                approval_evidence_ref=body.approval_evidence_ref,
                execution_terminal=execution_terminal,
            )
        except TaskControlGovernanceBlockedError as exc:
            _raise_task_control_http(exc)
        except TaskControlValidationError as exc:
            _raise_task_control_http(exc)
        except HitlResumeValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        if not outcome.accepted:
            blocked = outcome.blocked
            assert blocked is not None
            if blocked.detail == "invalid_resume_token":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=blocked.detail,
                )
            if blocked.detail in {
                "stale_checkpoint",
                "task_id_mismatch",
                "checkpoint_not_resumable",
                "execution_terminally_cancelled",
            }:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=blocked.detail,
                )
            if blocked.blocker_code is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "blocker_code": blocked.blocker_code,
                        "policy_action": blocked.policy_action,
                        "authorization_evidence": (
                            blocked.authorization_evidence.model_dump(mode="json")
                            if blocked.authorization_evidence is not None
                            else None
                        ),
                        "authorization_scope": (
                            blocked.authorization_scope.model_dump(mode="json")
                            if blocked.authorization_scope is not None
                            else None
                        ),
                    },
                )
        result = outcome.task_result
        assert result is not None
        return {
            "task_id": result.task_id,
            "state": result.state.value,
            "answer": result.answer,
            "resume_token": result.summary.resume_token,
        }

    app.include_router(router)
    return router
