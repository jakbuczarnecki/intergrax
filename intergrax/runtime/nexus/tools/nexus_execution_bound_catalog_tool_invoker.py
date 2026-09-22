# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus RuntimeToolInvoker-backed execution-bound catalog tool gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from pydantic import BaseModel

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
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
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
    _last_trace_steps: tuple[str, ...] = field(default=(), init=False, repr=False)

    @property
    def last_invocation_trace_steps(self) -> tuple[str, ...]:
        """Trace steps recorded on the RuntimeState during the last invoke (observability)."""
        return self._last_trace_steps

    def bind_execution_identity(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
    ) -> None:
        self.binding.run_id = run_id
        self.binding.task_id = task_id
        self.binding.agent_id = agent_id
        self.binding.tenant_id = tenant_id

    def invoke(
        self,
        request: ExecutionBoundCatalogToolInvokeRequest,
    ) -> ToolExecutionResult[BaseModel]:
        _require_bound_identity_matches(request, self.binding)
        state = self._runtime_state(request)
        invocation_context = ToolInvocationContext(
            run_id=request.run_id,
            step_id=request.step_id,
            tool_id=request.tool_id,
            agent_id=request.agent_id,
            tenant_id=request.tenant_id,
            correlation_request_id=request.correlation_request_id,
            wiring_resolver=request.wiring_resolver,
        )
        tool_request = ToolExecutionRequest(
            run_id=request.run_id,
            step_id=request.step_id,
            tool_id=request.tool_id,
            input=request.input,
            invocation_context=invocation_context,
            idempotency_key=request.idempotency_key,
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

    def _runtime_state(
        self, request: ExecutionBoundCatalogToolInvokeRequest
    ) -> RuntimeState:
        agent_id = _require_bound_identity_field(self.binding.agent_id, "agent_id")
        tenant_id = _require_bound_identity_field(self.binding.tenant_id, "tenant_id")
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
        )


def _require_bound_identity_field(value: str, label: str) -> str:
    if not value or not value.strip():
        raise ValueError(
            f"execution-bound catalog {label} must be set via bind_execution_identity",
        )
    return value.strip()


def _require_bound_identity_matches(
    request: ExecutionBoundCatalogToolInvokeRequest,
    binding: CatalogDeclarativeRunBinding,
) -> None:
    expected_run = validate_run_id(binding.run_id)
    expected_task = validate_task_id(binding.task_id)
    if request.run_id != expected_run:
        raise ValueError("execution-bound catalog run_id does not match bound identity")
    if request.task_id != expected_task:
        raise ValueError(
            "execution-bound catalog task_id does not match bound identity"
        )
    if request.tenant_id.strip() != binding.tenant_id.strip():
        raise ValueError(
            "execution-bound catalog tenant_id does not match bound identity"
        )
    if request.agent_id.strip() != binding.agent_id.strip():
        raise ValueError(
            "execution-bound catalog agent_id does not match bound identity"
        )


__all__ = [
    "NexusExecutionBoundCatalogToolInvoker",
]
