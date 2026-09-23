# © Artur Czarnecki. All rights reserved.

"""L3 continuation-aware catalog tool host behind execution-bound L2 (UCA-6C-R6)."""

from __future__ import annotations

from dataclasses import dataclass

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
from intergrax.contracts.execution.suspended_operation.codec import (
    SuspendedOperationCodecRegistry,
    SuspendedOperationKind,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.execution.suspended_operation.authority_scope_compat import (
    infer_authority_scope_from_invocation,
)
from intergrax.contracts.execution.suspended_operation.entity_id import (
    mint_suspended_operation_id,
)
from intergrax.contracts.execution.suspended_operation.payload_catalog import (
    CODE_EXEC_INPUT_SCHEMA_ID,
    ExecutionBoundCatalogToolOperationPayload,
)
from intergrax.contracts.execution.suspended_operation.persistence_conflict import (
    SuspendedOperationPersistenceConflictError,
)
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)
from intergrax.runtime.agent_governance.errors import (
    ToolGovernanceApprovalRequiredError,
)
from intergrax.runtime.execution.suspended_operation.governed_request import (
    compose_governed_continuation_from_agent_governance_pause,
    compose_governed_continuation_from_declarative_hitl_pause,
)
from intergrax.runtime.execution.suspended_operation.pause_required import (
    ExecutionSuspendedWorkPauseRequired,
)
from intergrax.runtime.execution.suspended_operation.catalog_tool_invocation_intent import (
    digest_execution_bound_catalog_tool_invocation_intent,
    digest_execution_bound_catalog_tool_invocation_intent_from_request,
    resolved_catalog_tool_idempotency_key,
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
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationMutationOutcome,
    SuspendedOperationMutationResult,
)
from intergrax.runtime.human.agent_governance_pause_projection import (
    AgentGovernancePauseProjectionOutcome,
    TaskAgentGovernancePauseProjectionAdapter,
)
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.nexus.tools.agent_governance_approval_pause_bridge import (
    AgentGovernanceApprovalPauseRequired,
    assert_agent_governance_pause_identity_consistency,
    build_agent_governance_pause_artifacts,
    raise_agent_governance_pause_from_tool_invocation,
)
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativePolicyHitlPauseRequired,
    raise_hitl_pause_from_tool_invocation,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.task.task import Task
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.invocation_wiring import (
    ToolInvocationContext,
    durable_sandbox_session_id_from_resolver,
)
from intergrax.tools.providers.sandbox.contracts import CodeExecInput


def _prepare_suspended_operation_with_persistence_reconciliation(
    store: SuspendedExecutionOperationStore,
    descriptor: SuspendedExecutionOperationDescriptor,
) -> SuspendedOperationMutationResult:
    """Retry prepare once after durable CAS conflict (UCA-6C-R6-R5.5-H2-R1)."""
    try:
        return store.prepare(descriptor)
    except SuspendedOperationPersistenceConflictError:
        return store.prepare(descriptor)


@dataclass(frozen=True, slots=True)
class ContinuationAwareCatalogToolHostDependencies:
    suspended_operation_store: SuspendedExecutionOperationStore
    hitl_continuation: InternalOrchestrationContinuation
    codec_registry: SuspendedOperationCodecRegistry
    task_checkpoint_store: TaskCheckpointPersistence | None = None


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

    def invoke(
        self,
        *,
        state: RuntimeState,
        request: ExecutionBoundCatalogToolInvokeRequest,
        declarative_grant: DeclarativeHitlApprovalGrant | None,
        task: Task | None = None,
    ) -> ToolExecutionResult[BaseModel]:
        tool_request = _tool_execution_request(request, declarative_grant)
        agent_id = request.agent_id
        try:
            return self._tool_invoker.invoke(
                state=state,
                agent_id=agent_id,
                request=tool_request,
            )
        except DeclarativePolicyHitlRequiredError as error:
            if self._deps is None:
                raise
            try:
                raise_hitl_pause_from_tool_invocation(
                    error,
                    state=state,
                    request=tool_request,
                    agent_id=agent_id,
                )
            except DeclarativePolicyHitlPauseRequired as pause:
                raise self._materialize_declarative_pause(
                    pause,
                    request=request,
                    task=task,
                ) from None
        except ToolGovernanceApprovalRequiredError as error:
            if error.governed_continuation_request is not None:
                raise
            if self._deps is None:
                raise
            contract = self._tool_invoker.registry.get(request.tool_id).contract
            try:
                raise_agent_governance_pause_from_tool_invocation(
                    error,
                    state=state,
                    contract=contract,
                    request=tool_request,
                    agent_id=agent_id,
                )
            except AgentGovernanceApprovalPauseRequired as pause:
                raise self._materialize_agent_governance_pause(
                    pause,
                    request=request,
                    task=task,
                ) from None

    def _materialize_declarative_pause(
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
        codec = deps.codec_registry.resolve(
            SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
            payload.payload_schema_version,
        )
        envelope = codec.encode(payload)
        digest = digest_suspended_operation_envelope(envelope)
        suspended_operation_id = mint_suspended_operation_id()
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
            authority_scope=infer_authority_scope_from_invocation(
                pause.signal.invocation_scope_id,
            ),
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
            declarative_pause=pause,
            governed_request=governed_request,
            descriptor=blocked.descriptor,
        )

    def _materialize_agent_governance_pause(
        self,
        pause: AgentGovernanceApprovalPauseRequired,
        *,
        request: ExecutionBoundCatalogToolInvokeRequest,
        task: Task | None,
    ) -> ExecutionSuspendedWorkPauseRequired:
        deps = self._deps
        if deps is None:
            raise RuntimeError("continuation-aware host dependencies required")
        if task is None:
            raise RuntimeError(
                "governed execution task required for agent governance pause projection",
            )
        if deps.task_checkpoint_store is None:
            raise RuntimeError(
                "task_checkpoint_store required for agent governance pause projection",
            )

        run_id, attempt_id = require_active_execution_identity()
        execution_id = state_execution_id()
        identity = ExecutionContinuationIdentity(
            task_id=validate_task_id(str(request.task_id)),
            run_id=RunId(str(run_id)),
            attempt_id=AttemptId(str(attempt_id)),
            execution_id=ExecutionId(str(execution_id)),
        )
        assert_agent_governance_pause_identity_consistency(
            pause.signal,
            request=request,
            run_id=RunId(str(run_id)),
            attempt_id=AttemptId(str(attempt_id)),
            execution_id=ExecutionId(str(execution_id)),
        )
        if task.runtime.governance.agent_governance_hitl_pending is not None:
            existing_scope = task.runtime.governance.agent_governance_hitl_pending.agent_governance_invocation_scope_id
            if not existing_scope.startswith("agr_"):
                raise RuntimeError("incompatible agent governance pending on task")

        from intergrax.contracts.agent_governance_hitl import (
            digest_logical_invocation_fingerprint,
            mint_agent_governance_invocation_scope_id,
        )

        invocation_intent_digest = (
            digest_execution_bound_catalog_tool_invocation_intent_from_request(request)
        )
        resolved_idempotency_key = resolved_catalog_tool_idempotency_key(
            run_id=str(pause.signal.run_id),
            step_id=pause.signal.step_id,
            idempotency_key=pause.signal.idempotency_key,
        )
        fingerprint = digest_logical_invocation_fingerprint(
            task_id=str(pause.signal.task_id),
            run_id=str(pause.signal.run_id),
            attempt_id=str(pause.signal.attempt_id),
            execution_id=str(pause.signal.execution_id),
            tenant_id=pause.signal.tenant_id,
            agent_id=pause.signal.agent_id,
            tool_id=pause.signal.tool_id,
            step_id=pause.signal.step_id,
            idempotency_key=resolved_idempotency_key,
            invocation_intent_digest=invocation_intent_digest,
        )
        existing_descriptor = (
            deps.suspended_operation_store.load_active_for_logical_invocation(
                fingerprint,
            )
        )
        scope_id = (
            existing_descriptor.invocation_scope_id
            if existing_descriptor is not None
            else mint_agent_governance_invocation_scope_id()
        )
        payload = _catalog_payload_from_request(
            request,
            invocation_scope_id=scope_id,
        )
        codec = deps.codec_registry.resolve(
            SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
            payload.payload_schema_version,
        )
        envelope = codec.encode(payload)
        digest = digest_suspended_operation_envelope(envelope)
        intent_from_payload = digest_execution_bound_catalog_tool_invocation_intent(
            payload,
        )
        if intent_from_payload != invocation_intent_digest:
            raise RuntimeError("catalog invocation intent digest mismatch")
        stable_pause_id: str | None = None
        stable_human_request_id: str | None = None
        if existing_descriptor is not None:
            task_pending = task.runtime.governance.agent_governance_hitl_pending
            if (
                task_pending is not None
                and task_pending.agent_governance_invocation_scope_id == scope_id
            ):
                stable_pause_id = task_pending.pause_id
                stable_human_request_id = task_pending.human_request_id
            elif existing_descriptor.materialization_state in {
                SuspendedOperationMaterializationState.BLOCKED,
                SuspendedOperationMaterializationState.CLAIMED,
            }:
                from intergrax.contracts.execution_continuation import (
                    ExecutionContinuationLookup,
                )

                port_pending = deps.hitl_continuation.port.get_pending(
                    ExecutionContinuationLookup(
                        continuation_id=existing_descriptor.continuation_id,
                    ),
                )
                stable_pause_id = port_pending.pause_id
                stable_human_request_id = port_pending.human_request_id
        requirement, pending, human_request = build_agent_governance_pause_artifacts(
            pause.signal,
            logical_invocation_fingerprint=fingerprint,
            payload_digest=digest,
            invocation_scope_id=scope_id,
            pause_id=stable_pause_id,
            human_request_id=stable_human_request_id,
        )
        governed_request = compose_governed_continuation_from_agent_governance_pause(
            pause,
            identity=identity,
            invocation_scope_id=scope_id,
        )
        if existing_descriptor is not None:
            governed_request = governed_request.model_copy(
                update={
                    "continuation_request_id": existing_descriptor.continuation_id,
                },
            )
        continuation_id = governed_request.continuation_request_id
        if existing_descriptor is None:
            suspended_operation_id = mint_suspended_operation_id()
            descriptor = SuspendedExecutionOperationDescriptor(
                suspended_operation_id=suspended_operation_id,
                operation_kind=SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
                identity=identity,
                continuation_id=continuation_id,
                invocation_scope_id=scope_id,
                materialization_state=SuspendedOperationMaterializationState.PREPARED,
                materialization_revision=0,
                claim_ownership=None,
                payload_digest=digest,
                payload=envelope,
                pause_generation=1,
                logical_invocation_fingerprint=requirement.logical_invocation_fingerprint,
                authority_scope=SuspendedOperationAuthorityScope.AGENT_RUNTIME_GOVERNANCE,
            )
            prepared = _prepare_suspended_operation_with_persistence_reconciliation(
                deps.suspended_operation_store,
                descriptor,
            )
            if prepared.descriptor is None:
                raise RuntimeError("suspended operation prepare failed")
            if prepared.outcome is SuspendedOperationMutationOutcome.ALREADY_ACTIVE:
                existing_descriptor = prepared.descriptor
            else:
                existing_descriptor = prepared.descriptor

        assert existing_descriptor is not None
        blocked_descriptor = existing_descriptor
        if (
            existing_descriptor.materialization_state
            is SuspendedOperationMaterializationState.PREPARED
        ):
            canonical_pending = establish_canonical_hitl_pause(
                task,
                identity=identity,
                continuation_id=continuation_id,
                reason=governed_request.reason,
                pause_id=pending.pause_id,
                human_request_id=pending.human_request_id,
                capability=deps.hitl_continuation,
                governed_correlation=governed_request.to_correlation(),
                human_prompt=human_request.prompt,
                execution_interrupt=None,
            )
            if canonical_pending.lifecycle_state not in {
                ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
                ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
            }:
                raise RuntimeError("canonical pause did not reach human-waiting state")

            blocked = deps.suspended_operation_store.block(
                suspended_operation_id=existing_descriptor.suspended_operation_id,
                expected_materialization_revision=existing_descriptor.materialization_revision,
                continuation=canonical_pending,
                governed_correlation=governed_request.to_correlation(),
            )
            if blocked.descriptor is None:
                raise RuntimeError("suspended operation block failed")
            blocked_descriptor = blocked.descriptor
        elif existing_descriptor.materialization_state in {
            SuspendedOperationMaterializationState.BLOCKED,
            SuspendedOperationMaterializationState.CLAIMED,
        }:
            establish_canonical_hitl_pause(
                task,
                identity=identity,
                continuation_id=continuation_id,
                reason=governed_request.reason,
                pause_id=pending.pause_id,
                human_request_id=pending.human_request_id,
                capability=deps.hitl_continuation,
                governed_correlation=governed_request.to_correlation(),
                human_prompt=human_request.prompt,
                execution_interrupt=None,
            )
        else:
            raise RuntimeError("unexpected suspended operation state for agent pause")

        projection = TaskAgentGovernancePauseProjectionAdapter(
            task=task,
            checkpoint_store=deps.task_checkpoint_store,
        ).persist_pause_projection(
            pending=pending,
            human_request=human_request,
        )
        if projection.outcome is AgentGovernancePauseProjectionOutcome.CONFLICT:
            raise RuntimeError("agent governance pause projection conflict")
        if projection.outcome is AgentGovernancePauseProjectionOutcome.STALE_REVISION:
            raise RuntimeError("agent governance pause checkpoint stale")

        return ExecutionSuspendedWorkPauseRequired(
            agent_governance_pause=pause,
            governed_request=governed_request,
            descriptor=blocked_descriptor,
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
            declarative_grant.invocation_scope_id
            if declarative_grant is not None
            else None
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
    sandbox_session_id = durable_sandbox_session_id_from_resolver(
        request.wiring_resolver
    )
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
