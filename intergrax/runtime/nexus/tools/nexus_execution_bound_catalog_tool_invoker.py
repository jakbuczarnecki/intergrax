# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus RuntimeToolInvoker-backed execution-bound catalog tool gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from pydantic import BaseModel

from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import validate_run_id, validate_task_id
from intergrax.prompts.registry.prompt_registry_resolver import (
    resolve_yaml_prompt_registry,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.agents.catalog_declarative_invoker import (
    CatalogDeclarativeRunBinding,
    _CatalogDispatchLLMStub,
)
from intergrax.runtime.nexus.budget.production_budget_policy import (
    ensure_production_run_budget,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import (
    InMemorySessionStorage,
)
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tools.continuation_aware_catalog_tool_host import (
    ContinuationAwareCatalogToolHost,
    ContinuationAwareCatalogToolHostDependencies,
)
from intergrax.runtime.nexus.tools.governance_approval_evidence_adapter import (
    declarative_hitl_grant_from_invocation_evidence,
    require_invocation_evidence_matches_request,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.governance.active_governed_execution_task import (
    peek_governed_execution_task,
)
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.invocation_wiring import ToolInvocationContext


@dataclass
class NexusExecutionBoundCatalogToolInvoker:
    """Build trusted RuntimeState internally; consumers invoke via typed request only."""

    tool_invoker: RuntimeToolInvoker
    policy_bundle: RuntimePolicyBundle
    caller_agent_id: str
    binding: CatalogDeclarativeRunBinding = field(
        default_factory=CatalogDeclarativeRunBinding,
    )
    production_mode: bool = False
    continuation_aware_dependencies: ContinuationAwareCatalogToolHostDependencies | None = (
        None
    )
    _last_trace_steps: tuple[str, ...] = field(default=(), init=False, repr=False)
    _continuation_host: ContinuationAwareCatalogToolHost | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def last_invocation_trace_steps(self) -> tuple[str, ...]:
        """Trace steps recorded on the RuntimeState during the last invoke (observability)."""
        return self._last_trace_steps

    def invoke(
        self,
        request: ExecutionBoundCatalogToolInvokeRequest,
    ) -> ToolExecutionResult[BaseModel]:
        state = self.build_runtime_state(request)
        declarative_grant = self._declarative_hitl_grant_for_request(request)
        host = self._continuation_host_instance()
        if host is not None:
            from intergrax.runtime.nexus.errors.tool_scope_violation_error import (
                ToolScopeViolationError,
            )

            try:
                result = host.invoke(
                    state=state,
                    request=request,
                    runtime_state_builder=None,
                    declarative_grant=declarative_grant,
                    task=peek_governed_execution_task(),
                )
            except ToolScopeViolationError as exc:
                result = ToolExecutionResult.fail("permission_error", str(exc))
            self._last_trace_steps = tuple(event.step for event in state.trace_events)
            return result

        invocation_context = ToolInvocationContext(
            run_id=request.run_id,
            step_id=request.step_id,
            tool_id=request.tool_id,
            agent_id=request.agent_id,
            tenant_id=request.tenant_id,
            correlation_request_id=request.correlation_request_id,
            wiring_resolver=request.wiring_resolver,
        )
        invocation_scope_id = (
            request.governance_approval_evidence.invocation_scope_id
            if request.governance_approval_evidence is not None
            else None
        )
        tool_request = ToolExecutionRequest(
            run_id=request.run_id,
            step_id=request.step_id,
            tool_id=request.tool_id,
            input=request.input,
            invocation_context=invocation_context,
            idempotency_key=request.idempotency_key,
            declarative_hitl_invocation_scope_id=invocation_scope_id,
        )
        from intergrax.runtime.nexus.errors.tool_scope_violation_error import (
            ToolScopeViolationError,
        )

        try:
            result = self.tool_invoker.invoke(
                state=state,
                agent_id=request.agent_id,
                request=cast(ToolExecutionRequest[BaseModel], tool_request),
            )
        except ToolScopeViolationError as exc:
            result = ToolExecutionResult.fail("permission_error", str(exc))
        self._last_trace_steps = tuple(event.step for event in state.trace_events)
        return result

    def _continuation_host_instance(self) -> ContinuationAwareCatalogToolHost | None:
        if self.continuation_aware_dependencies is None:
            return None
        if self._continuation_host is None:
            self._continuation_host = ContinuationAwareCatalogToolHost(
                tool_invoker=self.tool_invoker,
                dependencies=self.continuation_aware_dependencies,
            )
        return self._continuation_host

    def build_runtime_state(
        self, request: ExecutionBoundCatalogToolInvokeRequest
    ) -> RuntimeState:
        return self._runtime_state(request)

    def _runtime_state(
        self, request: ExecutionBoundCatalogToolInvokeRequest
    ) -> RuntimeState:
        agent_id = _require_request_identity_field(request.agent_id, "agent_id")
        tenant_id = _require_request_identity_field(request.tenant_id, "tenant_id")
        config = RuntimeConfig(
            llm_adapter=cast(LLMAdapter, _CatalogDispatchLLMStub()),
            production_mode=self.production_mode,
            enable_rag=False,
            enable_websearch=False,
            tool_invoker=self.tool_invoker,
            tenant_id=tenant_id,
            policy_bundle=self.policy_bundle,
        )
        config.validate()
        ensure_production_run_budget(config)
        context = RuntimeContext(
            config=config,
            session_manager=SessionManager(storage=InMemorySessionStorage()),
            prompt_registry=resolve_yaml_prompt_registry(
                catalog_path=config.prompt_catalog_path,
            ),
        )
        resolved_run_id = validate_run_id(request.run_id)
        resolved_task_id = validate_task_id(request.task_id)
        declarative_grant = self._declarative_hitl_grant_for_request(request)
        return RuntimeState(
            context=context,
            request=RuntimeRequest(
                agent_id=agent_id,
                user_id=self.binding.user_id,
                session_id=str(resolved_run_id),
                tenant_id=tenant_id,
                message="execution_bound.catalog",
                task_id=resolved_task_id,
                run_id=resolved_run_id,
            ),
            run_id=resolved_run_id,
            tool_traces=[],
            declarative_hitl_grant=declarative_grant,
        )

    def _declarative_hitl_grant_for_request(
        self,
        request: ExecutionBoundCatalogToolInvokeRequest,
    ) -> DeclarativeHitlApprovalGrant | None:
        if request.governance_approval_evidence is not None:
            require_invocation_evidence_matches_request(
                request.governance_approval_evidence,
                request,
            )
            return declarative_hitl_grant_from_invocation_evidence(
                request.governance_approval_evidence,
            )
        if self.binding.declarative_hitl_grant is not None:
            return self.binding.declarative_hitl_grant
        return None


def _require_request_identity_field(value: str, label: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"execution-bound catalog {label} must be set on invoke request")
    return value.strip()


__all__ = [
    "NexusExecutionBoundCatalogToolInvoker",
]
