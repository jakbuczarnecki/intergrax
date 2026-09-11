# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Optional, Protocol, Type, cast, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
    from intergrax.runtime.sandbox.isolation_gate import SandboxAvailabilityProvider
    from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
        IdempotencyPreEffectCoordinator,
        PreEffectClaimContext,
    )

from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
    DeclarativePolicyViolationError,
)
from intergrax.runtime.policy.side_effect_authorization_errors import (
    MeaningfulSideEffectAuthorizationRequiredError,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tracing.tools.tool_invocation import ToolInvocationEndDiagV1, ToolInvocationErrorDiagV1, ToolInvocationStartDiagV1
from intergrax.runtime.observability.modality_tool_trace import (
    consume_modality_metrics_for_tool,
    modality_metrics_dict,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.policy.policy_trace_diagnostics import (
    DeclarativePolicyEvaluationDiagV1,
    MeaningfulSideEffectAuthorizationRequiredDiagV1,
)
from intergrax.runtime.policy.declarative_enforcer import resolve_declarative_policy_enforcer
from intergrax.runtime.policy.declarative_tool_authorization_gate import (
    require_meaningful_side_effect_authorization,
)
from intergrax.runtime.sandbox.isolation_gate import (
    SandboxIsolationAvailability,
    require_sandbox_isolation,
)
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.policy.rules.schema import PolicyRuleAction
from intergrax.contracts.idempotency_store import ClaimOutcome
from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyAdmissionTimeoutError,
    DependencyConcurrencyExceededError,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyPolicyMissingError,
)
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
)
from intergrax.runtime.cancellation.coordinator import (
    CancellationCoordinator,
    CooperativeCancellationAbort,
    cooperative_delay_seconds,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationCancellationPort,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ProcessLocalExternalOperationOwner,
)
from intergrax.runtime.external_operations.tool_external_operation_attempt import (
    ExternalOperationStartSuppressedError,
    ToolExternalOperationAttempt,
    tool_external_operation_identity,
)
from intergrax.runtime.tools.scope_policy import ToolScopePolicy
from intergrax.tools.core.contracts import SideEffectRetrySafety, ToolContract
from intergrax.tools.execution_models import (
    ToolEffectCertainty,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolExecutor


@runtime_checkable
class TraceEmitter(Protocol):
    def trace_event(
        self,
        *,
        component: TraceComponent,
        step: str,
        message: str,
        level: TraceLevel,
        payload: Optional[object] = None,
        artifact_refs: Optional[list] = None,
    ) -> None: ...


class _ExternalEffectBoundary:
    """Tracks whether ToolExecutor may have been entered for post-claim safety."""

    __slots__ = ("may_have_started",)

    def __init__(self) -> None:
        self.may_have_started = False


class RuntimeToolInvoker:
    """
    Nexus-owned enforcement wrapper for tool invocation.

    Enforces:
    - tool registry presence
    - input schema type
    - output schema validation
    - error mapping -> RuntimeErrorCode
    - trace start/end/error
    - optional idempotency coordination before external effects
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        executor: ToolExecutor,
        scope_policy: Optional[ToolScopePolicy] = None,
        pre_effect_coordinator: Optional[IdempotencyPreEffectCoordinator] = None,
        sandbox_availability: Optional["SandboxAvailabilityProvider"] = None,
        agent_runtime_governance: Optional["AgentRuntimeGovernancePort"] = None,
        dependency_attempt_boundary: DependencyAttemptExecutionBoundary | None = None,
        external_operation_store: ExternalOperationStateStore | None = None,
        external_operation_owner: ProcessLocalExternalOperationOwner | None = None,
        external_operation_cancellation_port: ExternalOperationCancellationPort | None = None,
    ) -> None:
        self._registry = registry
        self._executor = executor
        self._scope_policy = scope_policy
        self._pre_effect_coordinator = pre_effect_coordinator
        self._sandbox_availability = sandbox_availability
        self._agent_runtime_governance = agent_runtime_governance
        self._dependency_attempt_boundary = dependency_attempt_boundary
        self._external_operation_store = external_operation_store
        if external_operation_store is not None and external_operation_owner is None:
            external_operation_owner = ProcessLocalExternalOperationOwner.mint()
        self._external_operation_owner = external_operation_owner
        self._external_operation_cancellation_port = external_operation_cancellation_port
        # Shared pool for timeout-isolated tool execution; default worker count
        # preserves concurrent independent invocations (not max_workers=1).
        self._execution_pool = ThreadPoolExecutor()
        self._execution_pool_closed = False

    def close(self) -> None:
        """Shut down admission boundary (when configured) and the execution pool."""
        if self._execution_pool_closed:
            return
        if self._dependency_attempt_boundary is not None:
            self._dependency_attempt_boundary.begin_shutdown()
        self._execution_pool.shutdown(wait=True)
        if self._dependency_attempt_boundary is not None:
            self._dependency_attempt_boundary.drain_and_close()
        self._execution_pool_closed = True

    @property
    def registry(self) -> ToolRegistry:
        """Read-only access to the runtime tool catalog (Phase O.5)."""
        return self._registry

    def invoke(
        self,
        *,
        state: "RuntimeState",
        agent_id: str,
        request: ToolExecutionRequest[BaseModel],
    ) -> ToolExecutionResult[BaseModel]:
        preparation = self._prepare_invocation(
            state=state,
            agent_id=agent_id,
            request=request,
        )
        if isinstance(preparation, ToolExecutionResult):
            return preparation

        contract = preparation
        claim_context: PreEffectClaimContext | None = None

        if self._requires_idempotency_coordination(contract, request):
            coordinator = self._pre_effect_coordinator
            if coordinator is None:
                raise RuntimeError(
                    "Side-effect tool with idempotency key requires a pre-effect coordinator.",
                )
            coordination = coordinator.before_external_effect(
                state=state,
                contract=contract,
                request=request,
            )
            if coordination.outcome == ClaimOutcome.REPLAY_COMPLETED:
                replay = coordination.replay_result
                if replay is None:
                    raise RuntimeError(
                        "Ledger inconsistency: REPLAY_COMPLETED without stored result.",
                    )
                return replay
            claim_context = coordination.claim_context
            if claim_context is None:
                raise RuntimeError("Ledger inconsistency: ACQUIRED without claim context.")

        if claim_context is not None:
            coordinator = self._pre_effect_coordinator
            if coordinator is None:
                raise RuntimeError(
                    "Pre-effect claim context present without coordinator.",
                )
            boundary = _ExternalEffectBoundary()
            try:
                result = self._execute_external_effect(
                    state=state,
                    agent_id=agent_id,
                    contract=contract,
                    request=request,
                    boundary=boundary,
                )
            except Exception:
                coordinator.on_post_claim_exception(
                    claim_context=claim_context,
                    contract=contract,
                    effect_may_have_started=boundary.may_have_started,
                )
                raise
            coordinator.after_external_effect(
                claim_context=claim_context,
                contract=contract,
                result=result,
            )
            return result

        return self._execute_external_effect(
            state=state,
            agent_id=agent_id,
            contract=contract,
            request=request,
        )

    def _prepare_invocation(
        self,
        *,
        state: "RuntimeState",
        agent_id: str,
        request: ToolExecutionRequest[BaseModel],
    ) -> ToolContract | ToolExecutionResult[BaseModel]:
        # 1) registry check + contract bind
        try:
            reg = self._registry.get(request.tool_id)
        except KeyError as exc:
            msg = str(exc)
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="tool_invocation_error",
                message="Tool not registered.",
                level=TraceLevel.ERROR,
                payload=ToolInvocationErrorDiagV1(
                    tool_id=request.tool_id,
                    step_id=str(request.step_id),
                    error_code=RuntimeErrorCode.TOOL_ERROR,
                    error_message=msg,
                ),
            )
            return ToolExecutionResult.fail(
                RuntimeErrorCode.TOOL_ERROR,
                msg,
                effect_certainty=ToolEffectCertainty.NOT_STARTED,
            )

        contract = reg.contract

        self._require_current_attempt_authorization(
            state=state,
            agent_id=agent_id,
            contract=contract,
            request=request,
        )

        # 2) input type enforcement
        if not isinstance(request.input, contract.input_schema):
            msg = (
                f"Tool input must be {contract.input_schema.__name__} "
                f"(got {type(request.input).__name__})."
            )
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="tool_invocation_error",
                message="Tool input validation failed.",
                level=TraceLevel.ERROR,
                payload=ToolInvocationErrorDiagV1(
                    tool_id=request.tool_id,
                    step_id=str(request.step_id),
                    error_code=RuntimeErrorCode.VALIDATION_ERROR,
                    error_message=msg,
                ),
            )
            result = ToolExecutionResult.fail(
                RuntimeErrorCode.VALIDATION_ERROR,
                msg,
                effect_certainty=ToolEffectCertainty.NOT_STARTED,
            )
            self._emit_boundary_event(
                state=state,
                agent_id=agent_id,
                contract=contract,
                request=request,
                result=result,
            )
            return result

        return contract

    def _require_current_attempt_authorization(
        self,
        *,
        state: "RuntimeState",
        agent_id: str,
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
    ) -> None:
        """Attempt-scoped authorization before each physical tool execution."""
        self._require_agent_runtime_governance(
            state=state,
            agent_id=agent_id,
            contract=contract,
            request=request,
        )

        if self._scope_policy is not None:
            allowed = self._scope_policy.is_allowed(
                agent_id=agent_id,
                tool_id=request.tool_id,
            )
            if not allowed:
                msg = "Tool execution denied by scope policy."
                state.trace_event(
                    component=TraceComponent.TOOLS,
                    step="tool_invocation_denied",
                    message=msg,
                    level=TraceLevel.ERROR,
                    payload=ToolInvocationErrorDiagV1(
                        tool_id=request.tool_id,
                        step_id=str(request.step_id),
                        error_code=RuntimeErrorCode.TOOL_ERROR,
                        error_message=msg,
                    ),
                )
                from intergrax.runtime.nexus.errors.tool_scope_violation_error import (
                    ToolScopeViolationError,
                )

                raise ToolScopeViolationError(
                    run_id=state.run_id,
                    agent_id=agent_id,
                    tool_id=request.tool_id,
                )

        if contract.requires_sandbox_isolation:
            if self._sandbox_availability is None:
                availability = SandboxIsolationAvailability(
                    session_configured=False,
                    host_configured=False,
                    healthy=True,
                )
            else:
                availability = self._sandbox_availability()
            try:
                require_sandbox_isolation(
                    contract=contract,
                    availability=availability,
                    run_id=state.run_id,
                    agent_id=agent_id,
                )
            except Exception as exc:
                from intergrax.runtime.sandbox.isolation_errors import (
                    SandboxIsolationRequiredError,
                )

                if isinstance(exc, SandboxIsolationRequiredError):
                    try:
                        state.trace_event(
                            component=TraceComponent.TOOLS,
                            step="sandbox_isolation_required",
                            message="Sandbox isolation is required but unavailable.",
                            level=TraceLevel.ERROR,
                            payload=ToolInvocationErrorDiagV1(
                                tool_id=exc.tool_id,
                                step_id=str(request.step_id),
                                error_code=RuntimeErrorCode.PERMISSION_ERROR,
                                error_message=str(exc),
                            ),
                        )
                    except Exception:
                        pass
                raise

        declarative_enforcer = resolve_declarative_policy_enforcer(state)
        try:
            require_meaningful_side_effect_authorization(
                contract=contract,
                enforcer=declarative_enforcer,
                run_id=state.run_id,
                agent_id=agent_id,
            )
        except MeaningfulSideEffectAuthorizationRequiredError as exc:
            try:
                state.trace_event(
                    component=TraceComponent.TOOLS,
                    step="meaningful_side_effect_authorization_required",
                    message="Meaningful side-effect authorization is required.",
                    level=TraceLevel.ERROR,
                    payload=MeaningfulSideEffectAuthorizationRequiredDiagV1(
                        tool_id=exc.tool_id,
                        agent_id=exc.agent_id,
                        run_id=exc.run_id,
                        reason=exc.reason.value,
                    ),
                )
            except Exception:
                pass
            raise

        if declarative_enforcer is not None:
            task_id = state.task_id
            policy_context = PolicyEvaluationContext(
                tool_id=request.tool_id,
                tenant_id=state.tenant_id,
                agent_id=agent_id,
                task_id=task_id,
                run_id=request.run_id,
                step_id=str(request.step_id),
                idempotency_key=request.idempotency_key,
                invocation_scope_id=request.declarative_hitl_invocation_scope_id,
                approval_grant=state.declarative_hitl_grant,
            )
            decision = declarative_enforcer.evaluate_tool_invocation(context=policy_context)
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="declarative_policy_evaluation",
                message="Declarative policy evaluated for tool invocation.",
                level=TraceLevel.INFO,
                payload=DeclarativePolicyEvaluationDiagV1(
                    tool_id=request.tool_id,
                    action=decision.action.value,
                    matched_rule_ids=decision.matched_rule_ids,
                    enforcement_mode=decision.enforcement_mode.value,
                    enforced=decision.enforced,
                    would_deny=decision.would_deny,
                    reasons=decision.reasons,
                    unknown_handler_ids=decision.unknown_handler_ids,
                    provenance_digest=decision.provenance_digest,
                ),
            )
            if decision.should_block_execution:
                if decision.action is PolicyRuleAction.REQUIRE_HITL:
                    raise DeclarativePolicyHitlRequiredError(
                        run_id=state.run_id,
                        agent_id=agent_id,
                        tool_id=request.tool_id,
                        matched_rule_ids=decision.matched_rule_ids,
                        reasons=decision.reasons,
                    )
                raise DeclarativePolicyViolationError(
                    run_id=state.run_id,
                    agent_id=agent_id,
                    tool_id=request.tool_id,
                    matched_rule_ids=decision.matched_rule_ids,
                    reasons=decision.reasons,
                )

    def _require_agent_runtime_governance(
        self,
        *,
        state: "RuntimeState",
        agent_id: str,
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
    ) -> None:
        """NPSC-4 governance boundary — mandatory on production tool execution."""
        governance = self._agent_runtime_governance
        if governance is None:
            if state.context.config.production_mode:
                from intergrax.runtime.agent_governance.errors import ToolGovernanceDeniedError

                capability = contract.category.strip() or contract.tool_id
                raise ToolGovernanceDeniedError(
                    run_id=state.run_id,
                    agent_id=agent_id,
                    tool_id=request.tool_id,
                    capability=capability,
                    reason="agent_runtime_governance_not_configured",
                    policy_results=(),
                )
            return

        from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort
        from intergrax.runtime.agent_governance.request_builder import (
            build_tool_authorization_request,
        )
        from intergrax.runtime.agent_governance.errors import (
            ToolGovernanceApprovalRequiredError,
            ToolGovernanceDeniedError,
        )
        from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
        from intergrax.runtime.nexus.tracing.tools.tool_invocation import (
            ToolInvocationErrorDiagV1,
        )
        from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode

        if not isinstance(governance, AgentRuntimeGovernancePort):
            raise RuntimeError("agent_runtime_governance_invalid_type")

        auth_request = build_tool_authorization_request(
            state=state,
            agent_id=agent_id,
            contract=contract,
            request=request,
        )
        try:
            governance.authorize_tool(auth_request)
        except ToolGovernanceDeniedError as exc:
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="agent_runtime_governance_denied",
                message="Agent runtime governance denied tool invocation.",
                level=TraceLevel.ERROR,
                payload=ToolInvocationErrorDiagV1(
                    tool_id=exc.tool_id,
                    step_id=str(request.step_id),
                    error_code=RuntimeErrorCode.PERMISSION_ERROR,
                    error_message=str(exc),
                ),
            )
            raise
        except ToolGovernanceApprovalRequiredError as exc:
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="agent_runtime_governance_approval_required",
                message="Agent runtime governance requires human approval.",
                level=TraceLevel.ERROR,
                payload=ToolInvocationErrorDiagV1(
                    tool_id=exc.tool_id,
                    step_id=str(request.step_id),
                    error_code=RuntimeErrorCode.PERMISSION_ERROR,
                    error_message=str(exc),
                ),
            )
            raise

    @staticmethod
    def _requires_idempotency_coordination(
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
    ) -> bool:
        return contract.side_effects and request.idempotency_key is not None

    @staticmethod
    def _emit_tool_invocation_start_non_blocking(
        *,
        state: "RuntimeState",
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
    ) -> None:
        """Emit tool_invocation_start; observability failures must not block execution."""
        try:
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="tool_invocation_start",
                message="Tool invocation started.",
                level=TraceLevel.INFO,
                payload=ToolInvocationStartDiagV1(
                    tool_id=contract.tool_id,
                    step_id=str(request.step_id),
                    side_effects=contract.side_effects,
                    input_payload=request.input.model_dump(),
                    risk_level=contract.risk_level.value,
                    injects_context=contract.injects_context,
                    category=contract.category,
                    timeout_ms=contract.timeout_ms,
                ),
            )
        except Exception:
            # Observability-only: execution and idempotency path continue.
            try:
                state.trace_event(
                    component=TraceComponent.TOOLS,
                    step="tool_invocation_start_trace_error",
                    message="Tool invocation start trace failed (non-blocking).",
                    level=TraceLevel.WARNING,
                )
            except Exception:
                # Best-effort warning; do not mask ToolExecutor failures.
                pass

    def _execute_external_effect(
        self,
        *,
        state: "RuntimeState",
        agent_id: str,
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
        boundary: _ExternalEffectBoundary | None = None,
    ) -> ToolExecutionResult[BaseModel]:
        # 3) trace start (non-blocking after idempotency claim / replay resolution)
        self._emit_tool_invocation_start_non_blocking(
            state=state,
            contract=contract,
            request=request,
        )

        # 4) execute + normalize (timeout + runtime-managed retries)
        return self._execute_with_policy(
            state=state,
            agent_id=agent_id,
            contract=contract,
            request=request,
            boundary=boundary,
        )

    def _execute_with_policy(
        self,
        *,
        state: "RuntimeState",
        agent_id: str,
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
        boundary: _ExternalEffectBoundary | None = None,
    ) -> ToolExecutionResult[BaseModel]:
        policy = contract.retry_policy
        attempts = self._effective_max_attempts(contract)
        last_exc: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            if attempt > 1:
                if self._cooperative_cancellation_requested(state):
                    return self._tool_result_task_cancelled(
                        state=state,
                        contract=contract,
                        request=request,
                        agent_id=agent_id,
                    )
                if policy.backoff_ms > 0:
                    try:
                        cooperative_delay_seconds(
                            policy.backoff_ms / 1000.0,
                            should_abort=lambda: self._cooperative_cancellation_requested(
                                state
                            ),
                        )
                    except CooperativeCancellationAbort:
                        return self._tool_result_task_cancelled(
                            state=state,
                            contract=contract,
                            request=request,
                            agent_id=agent_id,
                        )
                self._require_current_attempt_authorization(
                    state=state,
                    agent_id=agent_id,
                    contract=contract,
                    request=request,
                )

            start_perf = time.perf_counter()
            try:
                raw_out = self._execute_once(
                    state,
                    contract,
                    request,
                    effect_boundary=boundary,
                    physical_attempt_sequence=attempt,
                )
                out = self._validate_output(contract.output_schema, raw_out)
                duration_ms = max(0, int((time.perf_counter() - start_perf) * 1000))

                modality_metrics = modality_metrics_dict(
                    consume_modality_metrics_for_tool(self._registry, contract.tool_id)
                )
                state.trace_event(
                    component=TraceComponent.TOOLS,
                    step="tool_invocation_end",
                    message="Tool invocation finished.",
                    level=TraceLevel.INFO,
                    payload=ToolInvocationEndDiagV1(
                        tool_id=contract.tool_id,
                        step_id=str(request.step_id),
                        success=True,
                        output_preview=self._preview_output(out),
                        duration_ms=duration_ms,
                        modality_metrics=modality_metrics,
                    ),
                )
                result = ToolExecutionResult.ok(out)
                self._emit_boundary_event(
                    state=state,
                    agent_id=agent_id,
                    contract=contract,
                    request=request,
                    result=result,
                )
                return result

            except ExternalOperationStartSuppressedError:
                return self._tool_result_task_cancelled(
                    state=state,
                    contract=contract,
                    request=request,
                    agent_id=agent_id,
                )

            except DependencyConcurrencyPolicyMissingError:
                raise

            except (
                DependencyConcurrencyExceededError,
                DependencyConcurrencyAdmissionTimeoutError,
            ) as exc:
                duration_ms = max(0, int((time.perf_counter() - start_perf) * 1000))
                msg = f"{type(exc).__name__}: {exc}"
                state.trace_event(
                    component=TraceComponent.TOOLS,
                    step="tool_invocation_error",
                    message="Tool dependency admission rejected.",
                    level=TraceLevel.ERROR,
                    payload=ToolInvocationErrorDiagV1(
                        tool_id=contract.tool_id,
                        step_id=str(request.step_id),
                        error_code=RuntimeErrorCode.DEPENDENCY_ERROR,
                        error_message=msg,
                    ),
                )
                result = ToolExecutionResult.fail(
                    RuntimeErrorCode.DEPENDENCY_ERROR,
                    msg,
                    effect_certainty=ToolEffectCertainty.NOT_STARTED,
                )
                self._emit_boundary_event(
                    state=state,
                    agent_id=agent_id,
                    contract=contract,
                    request=request,
                    result=result,
                )
                return result

            except FuturesTimeoutError:
                duration_ms = max(0, int((time.perf_counter() - start_perf) * 1000))
                msg = f"Tool execution timed out after {contract.timeout_ms}ms"
                state.trace_event(
                    component=TraceComponent.TOOLS,
                    step="tool_invocation_error",
                    message="Tool invocation timed out.",
                    level=TraceLevel.ERROR,
                    payload=ToolInvocationErrorDiagV1(
                        tool_id=contract.tool_id,
                        step_id=str(request.step_id),
                        error_code=RuntimeErrorCode.TIMEOUT,
                        error_message=msg,
                    ),
                )
                result = ToolExecutionResult.fail(RuntimeErrorCode.TIMEOUT, msg)
                self._emit_boundary_event(
                    state=state,
                    agent_id=agent_id,
                    contract=contract,
                    request=request,
                    result=result,
                )
                return result

            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    state.trace_event(
                        component=TraceComponent.TOOLS,
                        step="tool_invocation_retry",
                        message=f"Tool invocation failed (attempt {attempt}/{attempts}); retrying.",
                        level=TraceLevel.WARNING,
                        payload=ToolInvocationErrorDiagV1(
                            tool_id=contract.tool_id,
                            step_id=str(request.step_id),
                            error_code=self._map_error(contract, exc),
                            error_message=f"{type(exc).__name__}: {exc}",
                        ),
                    )
                    continue

                code = self._map_error(contract, exc)
                msg = f"{type(exc).__name__}: {exc}"
                duration_ms = max(0, int((time.perf_counter() - start_perf) * 1000))

                state.trace_event(
                    component=TraceComponent.TOOLS,
                    step="tool_invocation_error",
                    message="Tool invocation failed.",
                    level=TraceLevel.ERROR,
                    payload=ToolInvocationErrorDiagV1(
                        tool_id=contract.tool_id,
                        step_id=str(request.step_id),
                        error_code=code,
                        error_message=msg,
                    ),
                )
                result = ToolExecutionResult.fail(code, msg)
                self._emit_boundary_event(
                    state=state,
                    agent_id=agent_id,
                    contract=contract,
                    request=request,
                    result=result,
                )
                return result

        # Unreachable if attempts >= 1; satisfies type checker.
        if last_exc is not None:
            code = self._map_error(contract, last_exc)
            result = ToolExecutionResult.fail(code, str(last_exc))
            self._emit_boundary_event(
                state=state,
                agent_id=agent_id,
                contract=contract,
                request=request,
                result=result,
            )
            return result
        result = ToolExecutionResult.fail(RuntimeErrorCode.TOOL_ERROR, "Tool execution failed.")
        self._emit_boundary_event(
            state=state,
            agent_id=agent_id,
            contract=contract,
            request=request,
            result=result,
        )
        return result

    @staticmethod
    def _cooperative_cancellation_requested(state: "RuntimeState") -> bool:
        return CancellationCoordinator.is_requested(state.request.metadata)

    def _tool_result_task_cancelled(
        self,
        *,
        state: "RuntimeState",
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
        agent_id: str,
    ) -> ToolExecutionResult[BaseModel]:
        state.trace_event(
            component=TraceComponent.TOOLS,
            step="tool_invocation_cancelled",
            message="Tool invocation aborted after cooperative cancellation.",
            level=TraceLevel.INFO,
            payload=ToolInvocationErrorDiagV1(
                tool_id=contract.tool_id,
                step_id=str(request.step_id),
                error_code=RuntimeErrorCode.RUNTIME_ERROR,
                error_message="task_cancelled",
            ),
        )
        result = ToolExecutionResult.fail(
            RuntimeErrorCode.RUNTIME_ERROR,
            "task_cancelled",
            effect_certainty=ToolEffectCertainty.NOT_STARTED,
        )
        self._emit_boundary_event(
            state=state,
            agent_id=agent_id,
            contract=contract,
            request=request,
            result=result,
        )
        return result

    @staticmethod
    def _effective_max_attempts(contract: ToolContract) -> int:
        """Side-effect tools require positive retry-safety proof before automatic retry."""
        policy_attempts = contract.retry_policy.max_attempts
        if not contract.side_effects:
            return policy_attempts
        if contract.side_effect_retry_safety == SideEffectRetrySafety.NOT_RETRY_SAFE:
            return 1
        return policy_attempts

    @staticmethod
    def _emit_boundary_event(
        *,
        state: "RuntimeState",
        agent_id: str,
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
        result: ToolExecutionResult[BaseModel],
    ) -> None:
        from intergrax.runtime.attestation.boundary_emitter import ExecutionBoundaryEmitter

        try:
            ExecutionBoundaryEmitter.maybe_emit(
                state=state,
                agent_id=agent_id,
                contract=contract,
                request=request,
                result=result,
            )
        except Exception:
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="execution_boundary_export_error",
                message="Execution boundary export failed (non-blocking).",
                level=TraceLevel.WARNING,
            )

    def _build_tool_external_operation_attempt(
        self,
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
        *,
        physical_attempt_sequence: int,
    ) -> ToolExternalOperationAttempt:
        store = self._external_operation_store
        if store is None:
            return ToolExternalOperationAttempt(
                store=None,
                owner=None,
                identity=None,
            )
        identity = tool_external_operation_identity(
            tool_id=contract.tool_id,
            step_id=str(request.step_id),
            physical_attempt_sequence=physical_attempt_sequence,
        )
        return ToolExternalOperationAttempt(
            store=store,
            owner=self._external_operation_owner,
            identity=identity,
            cancellation_port=self._external_operation_cancellation_port,
        )

    def _execute_once(
        self,
        state: "RuntimeState",
        contract: ToolContract,
        request: ToolExecutionRequest[BaseModel],
        *,
        effect_boundary: _ExternalEffectBoundary | None = None,
        physical_attempt_sequence: int = 1,
    ) -> BaseModel:
        timeout_s = contract.timeout_ms / 1000.0
        dep_boundary = self._dependency_attempt_boundary
        ext_op = self._build_tool_external_operation_attempt(
            contract,
            request,
            physical_attempt_sequence=physical_attempt_sequence,
        )
        ext_op.before_physical_submit()

        if dep_boundary is None:
            ext_op.mark_running()
            future = self._execution_pool.submit(self._executor.execute, request)
            if effect_boundary is not None:
                effect_boundary.may_have_started = True
            try:
                return future.result(timeout=timeout_s)
            except FuturesTimeoutError:
                ext_op.mark_unknown()
                raise
            except BaseException:
                if self._cooperative_cancellation_requested(state):
                    ext_op.request_cancel()
                    ext_op.mark_cancelled()
                else:
                    ext_op.mark_failed()
                raise
            else:
                ext_op.mark_succeeded()

        admission_request = DependencyConcurrencyAdmissionRequest(
            dependency=DependencyConcurrencyIdentity(
                kind=DependencyConcurrencyKind.TOOL,
                value=contract.tool_id,
            ),
            tenant_id=state.tenant_id,
        )
        attempt_handle = dep_boundary.acquire(admission_request)
        try:
            ext_op.mark_running()
            future = self._execution_pool.submit(self._executor.execute, request)
        except BaseException:
            ext_op.mark_failed()
            dep_boundary.release_after_submit_failure(attempt_handle)
            raise

        dep_boundary.bind_worker(
            attempt_handle,
            cast(ConcurrentFuture[object], future),
        )
        if effect_boundary is not None:
            effect_boundary.may_have_started = True

        try:
            result = future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            if dep_boundary.detach_if_still_running(attempt_handle):
                ext_op.mark_unknown()
                raise
            dep_boundary.complete_attached(attempt_handle)
            if future.done():
                ext_op.mark_succeeded()
                return future.result()
            ext_op.mark_unknown()
            raise
        except BaseException:
            if self._cooperative_cancellation_requested(state):
                ext_op.request_cancel()
                ext_op.mark_cancelled()
            else:
                ext_op.mark_failed()
            dep_boundary.complete_attached(attempt_handle)
            raise
        else:
            ext_op.mark_succeeded()
            dep_boundary.complete_attached(attempt_handle)
            return result

    @staticmethod
    def _map_error(contract: ToolContract, exc: Exception) -> RuntimeErrorCode:
        # contract.error_mapping: Mapping[type[Exception], RuntimeErrorCode]
        for exc_type, code in contract.error_mapping.items():
            if isinstance(exc, exc_type):
                return cast(RuntimeErrorCode, code)
        return RuntimeErrorCode.TOOL_ERROR


    @staticmethod
    def _validate_output(schema: Type[BaseModel], raw_out: BaseModel) -> BaseModel:
        """
        Strict runtime boundary check.

        Tool handler MUST return exactly the declared output_schema type.
        """
        if not isinstance(raw_out, schema):
            raise TypeError(
                f"Tool returned invalid output type. "
                f"Expected {schema.__name__}, got {type(raw_out).__name__}."
            )
        return raw_out


    @staticmethod
    def _preview_output(out: BaseModel, *, limit: int = 300) -> str:
        s = out.model_dump_json()
        if len(s) <= limit:
            return s
        return s[:limit] + "…"
