# © Artur Czarnecki. All rights reserved.

"""L3 continuation-aware catalog tool host behind execution-bound L2 (UCA-6C-R6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel

from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    require_active_execution_identity,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.execution.suspended_operation.codec import SuspendedOperationKind
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.payload_catalog import (
    CODE_EXEC_INPUT_SCHEMA_ID,
    ExecutionBoundCatalogToolOperationPayload,
)
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)
from intergrax.runtime.execution.suspended_operation.governed_request import (
    compose_governed_continuation_from_declarative_hitl_pause,
)
from intergrax.runtime.execution.suspended_operation.in_memory_store import (
    InMemorySuspendedExecutionOperationStore,
)
from intergrax.runtime.execution.suspended_operation.codec_registry import (
    DefaultSuspendedOperationCodecRegistry,
)
from intergrax.runtime.execution.suspended_operation.pause_required import (
    ExecutionSuspendedWorkPauseRequired,
)
from intergrax.runtime.execution.suspended_operation.payload_digest import (
    digest_suspended_operation_envelope,
)
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
    establish_canonical_hitl_pause,
)
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativePolicyHitlPauseRequired,
    raise_hitl_pause_from_tool_invocation,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.task.task import Task
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.invocation_wiring import ToolInvocationContext
from intergrax.tools.providers.sandbox.contracts import CodeExecInput


@dataclass(frozen=True, slots=True)
class ContinuationAwareCatalogToolHostDependencies:
    suspended_operation_store: SuspendedExecutionOperationStore
    hitl_continuation: InternalOrchestrationContinuation
    codec_registry: DefaultSuspendedOperationCodecRegistry | None = None


class ContinuationAwareCatalogToolHost:
    """EE-internal L3: RuntimeToolInvoker + durable REQUIRE_HITL materialization."""

    def __init__(
        self,
        *,
        tool_invoker: RuntimeToolInvoker,
        dependencies: ContinuationAwareCatalogToolHostDependencies | None = None,
    ) -> None:
        self._tool_invoker = tool_invoker
        self._deps = dependencies
        self._codecs = (
            dependencies.codec_registry
            if dependencies is not None and dependencies.codec_registry is not None
            else DefaultSuspendedOperationCodecRegistry()
        )

    def invoke(
        self,
        *,
        state: RuntimeState,
        request: ExecutionBoundCatalogToolInvokeRequest,
        runtime_state_builder,
        declarative_grant: DeclarativeHitlApprovalGrant | None,
        task: Task | None = None,
    ) -> ToolExecutionResult[BaseModel]:
        tool_request = _tool_execution_request(request, declarative_grant)
        agent_id = request.agent_id
        try:
            return self._tool_invoker.invoke(
                state=state,
                agent_id=agent_id,
                request=cast(ToolExecutionRequest[BaseModel], tool_request),
            )
        except DeclarativePolicyHitlRequiredError as error:
            if self._deps is None:
                raise
            try:
                raise_hitl_pause_from_tool_invocation(
                    error,
                    state=state,
                    request=cast(ToolExecutionRequest[object], tool_request),
                    agent_id=agent_id,
                )
            except DeclarativePolicyHitlPauseRequired as pause:
                raise self._materialize_pause(
                    pause,
                    request=request,
                    task=task,
                ) from None

    def _materialize_pause(
        self,
        pause: DeclarativePolicyHitlPauseRequired,
        *,
        request: ExecutionBoundCatalogToolInvokeRequest,
        task: Task | None,
    ) -> ExecutionSuspendedWorkPauseRequired:
        deps = self._deps
        if deps is None:
            raise RuntimeError("continuation-aware host dependencies required")

        run_id, attempt_id = require_active_execution_identity()
        execution_id = state_execution_id()
        identity = ExecutionContinuationIdentity(
            task_id=validate_task_id(str(request.task_id)),
            run_id=RunId(str(run_id)),
            attempt_id=AttemptId(str(attempt_id)),
            execution_id=ExecutionId(str(execution_id)),
        )
        governed_request = compose_governed_continuation_from_declarative_hitl_pause(
            pause,
            identity=identity,
        )
        continuation_id = governed_request.continuation_request_id
        payload = _catalog_payload_from_request(
            request,
            invocation_scope_id=pause.signal.invocation_scope_id,
        )
        envelope = self._codecs.encode(
            payload,
            operation_kind=SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
            payload_schema_version=payload.payload_schema_version,
        )
        digest = digest_suspended_operation_envelope(envelope)
        suspended_operation_id = (
            InMemorySuspendedExecutionOperationStore.mint_suspended_operation_id()
        )
        descriptor = SuspendedExecutionOperationDescriptor(
            suspended_operation_id=suspended_operation_id,
            operation_kind=SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
            identity=identity,
            continuation_id=continuation_id,
            invocation_scope_id=pause.signal.invocation_scope_id,
            materialization_state=SuspendedOperationMaterializationState.PREPARED,
            materialization_revision=0,
            claim_ownership=None,
            payload_digest=digest,
            payload=envelope,
        )
        prepared = deps.suspended_operation_store.prepare(descriptor)
        if prepared.descriptor is None:
            raise RuntimeError("suspended operation prepare failed")

        if task is None:
            raise RuntimeError(
                "governed execution task required for canonical HITL pause projection",
            )
        pending = establish_canonical_hitl_pause(
            task,
            identity=identity,
            continuation_id=continuation_id,
            reason=governed_request.reason,
            pause_id=pause.pending.pause_id,
            human_request_id=pause.pending.human_request_id,
            capability=deps.hitl_continuation,
            governed_correlation=governed_request.to_correlation(),
            human_prompt=None,
            execution_interrupt=pause.governance.interrupt,
        )
        if pending.lifecycle_state not in {
            ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
            ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        }:
            raise RuntimeError("canonical pause did not reach human-waiting state")

        blocked = deps.suspended_operation_store.block(
            suspended_operation_id=suspended_operation_id,
            expected_materialization_revision=0,
            continuation=pending,
            governed_correlation=governed_request.to_correlation(),
        )
        if blocked.descriptor is None:
            raise RuntimeError("suspended operation block failed")

        return ExecutionSuspendedWorkPauseRequired(
            pause=pause,
            governed_request=governed_request,
            descriptor=blocked.descriptor,
        )


def state_execution_id() -> str:
    from intergrax.contracts.execution_identity import peek_active_execution_id

    execution_id = peek_active_execution_id()
    if execution_id is None:
        raise RuntimeError("active execution_id required for HITL materialization")
    return str(execution_id)


def _tool_execution_request(
    request: ExecutionBoundCatalogToolInvokeRequest,
    declarative_grant: DeclarativeHitlApprovalGrant | None,
) -> ToolExecutionRequest[BaseModel]:
    invocation_context = ToolInvocationContext(
        run_id=request.run_id,
        step_id=request.step_id,
        tool_id=request.tool_id,
        agent_id=request.agent_id,
        tenant_id=request.tenant_id,
        correlation_request_id=request.correlation_request_id,
        wiring_resolver=request.wiring_resolver,
    )
    return ToolExecutionRequest(
        run_id=request.run_id,
        step_id=request.step_id,
        tool_id=request.tool_id,
        input=request.input,
        invocation_context=invocation_context,
        idempotency_key=request.idempotency_key,
        declarative_hitl_invocation_scope_id=(
            declarative_grant.invocation_scope_id if declarative_grant is not None else None
        ),
    )


def _catalog_payload_from_request(
    request: ExecutionBoundCatalogToolInvokeRequest,
    *,
    invocation_scope_id: str,
) -> ExecutionBoundCatalogToolOperationPayload:
    if type(request.input) is not CodeExecInput:
        raise TypeError("UCA-6C-R6 catalog suspended payload requires CodeExecInput")
    run_id_str = validate_run_id(request.run_id)
    sandbox_session_id = None
    resolver = request.wiring_resolver
    if resolver is not None:
        sandbox = getattr(resolver, "sandbox_session", None)
        if sandbox is not None:
            sandbox_session_id = str(getattr(sandbox, "session_id", "") or "") or None
    return ExecutionBoundCatalogToolOperationPayload(
        tool_id=request.tool_id,
        tool_input_schema_id=CODE_EXEC_INPUT_SCHEMA_ID,
        tool_input=request.input,
        tenant_id=request.tenant_id,
        task_id=str(request.task_id),
        run_id=run_id_str,
        agent_id=request.agent_id,
        step_id=request.step_id,
        invocation_scope_id=invocation_scope_id,
        idempotency_key=request.idempotency_key or f"{run_id_str}:{request.step_id}",
        correlation_request_id=request.correlation_request_id,
        sandbox_session_id=sandbox_session_id,
    )


__all__ = [
    "ContinuationAwareCatalogToolHost",
    "ContinuationAwareCatalogToolHostDependencies",
]
