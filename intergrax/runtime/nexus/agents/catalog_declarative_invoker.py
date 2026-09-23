# © Artur Czarnecki. All rights reserved.

"""Catalog-backed ``DeclarativeToolInvoker`` for ACP host wiring (ACP-PROD-2 depth)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from intergrax.contracts.declarative_tool_invoke_result import DeclarativeToolInvokeResult
from intergrax.knowledge.contracts.validation import JsonObject
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import (
    InMemorySessionStorage,
)
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tools.catalog_dispatch import invoke_catalog_tool_request
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker


class _CatalogDispatchLLMStub(BaseLLMAdapter):
    """Minimal LLM adapter for catalog-only dispatch (no generation)."""

    provider = "acp_catalog_dispatch"
    model = "catalog-dispatch-stub"

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        _ = messages, temperature, max_tokens, run_id
        return build_adapter_response(content="")


@dataclass
class CatalogDeclarativeRunBinding:
    """Immutable session-scoped metadata for declarative catalog dispatch (not execution identity)."""

    user_id: str = ""
    declarative_hitl_grant: DeclarativeHitlApprovalGrant | None = None


@dataclass
class CatalogDeclarativeToolInvoker:
    """Invoke declarative actions through the Tier-1 catalog tool gateway."""

    tool_invoker: RuntimeToolInvoker
    binding: CatalogDeclarativeRunBinding = field(
        default_factory=CatalogDeclarativeRunBinding
    )
    production_mode: bool = False

    def bind_run(
        self,
        *,
        run_id: str = "",
        task_id: str = "",
        agent_id: str = "",
        tenant_id: str = "",
        user_id: str = "",
    ) -> None:
        """Bind session metadata only (execution identity is per-call on ``invoke``)."""
        _ = run_id, task_id, agent_id, tenant_id
        self.binding.user_id = user_id

    def _runtime_state(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        user_id: str,
    ) -> RuntimeState:
        from intergrax.contracts.execution_identity import (
            validate_run_id,
            validate_task_id,
        )

        agent_id = _require_invoke_identity_field(agent_id, "agent_id")
        tenant_id = _require_invoke_identity_field(tenant_id, "tenant_id")
        host_tool_invoker = self.tool_invoker
        from intergrax.prompts.registry.prompt_registry_resolver import (
            resolve_yaml_prompt_registry,
        )
        from intergrax.runtime.nexus.budget.production_budget_policy import (
            ensure_production_run_budget,
        )

        config = RuntimeConfig(
            llm_adapter=_CatalogDispatchLLMStub(),
            production_mode=self.production_mode,
            enable_rag=False,
            enable_websearch=False,
            tool_invoker=host_tool_invoker,
            tenant_id=tenant_id,
        )
        config.validate()
        ensure_production_run_budget(config)
        # Do not use RuntimeContext.build() — it materializes a fresh catalog and
        # replaces config.tool_invoker; host-bound invoker must be preserved.
        context = RuntimeContext(
            config=config,
            session_manager=SessionManager(storage=InMemorySessionStorage()),
            prompt_registry=resolve_yaml_prompt_registry(
                catalog_path=config.prompt_catalog_path,
            ),
        )
        resolved_run_id = validate_run_id(run_id)
        resolved_task_id = validate_task_id(task_id)
        grant = self.binding.declarative_hitl_grant
        return RuntimeState(
            context=context,
            request=RuntimeRequest(
                agent_id=agent_id,
                user_id=user_id,
                session_id=str(resolved_run_id),
                tenant_id=tenant_id,
                message="acp.declarative",
                task_id=resolved_task_id,
                run_id=resolved_run_id,
            ),
            run_id=resolved_run_id,
            tool_traces=[],
            declarative_hitl_grant=grant,
        )

    async def invoke(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        tool_id: str,
        args: JsonObject,
        idempotency_key: str | None,
    ) -> DeclarativeToolInvokeResult:
        resolved_tenant_id = _require_invoke_identity_field(tenant_id, "tenant_id")
        resolved_run_id = _require_invoke_identity_field(run_id, "run_id")
        resolved_task_id = _require_invoke_identity_field(task_id, "task_id")
        resolved_agent_id = _require_invoke_identity_field(agent_id, "agent_id")
        user_id = self.binding.user_id
        request = ToolRequest(
            tool_name=tool_id,
            agent_id=resolved_agent_id,
            step_id="acp.declarative",
            input=args,
            idempotency_key=idempotency_key,
        )
        response = invoke_catalog_tool_request(
            state=self._runtime_state(
                tenant_id=resolved_tenant_id,
                run_id=resolved_run_id,
                task_id=resolved_task_id,
                agent_id=resolved_agent_id,
                user_id=user_id,
            ),
            request=request,
            trace_step="AcpDeclarativeTool",
        )
        if response.status == ToolResponseStatus.SUCCESS:
            external_ref: str | None = None
            if response.output:
                for key in ("external_ref", "id", "message_id", "ref"):
                    value = response.output.get(key)
                    if value is not None:
                        external_ref = str(value)
                        break
            return DeclarativeToolInvokeResult(
                status="success",
                output=response.output,
                external_ref=external_ref,
                duration_ms=response.duration_ms,
            )
        if response.status == ToolResponseStatus.DENIED:
            return DeclarativeToolInvokeResult(
                status="denied",
                error=response.error,
                duration_ms=response.duration_ms,
            )
        return DeclarativeToolInvokeResult(
            status="failed",
            error=response.error,
            duration_ms=response.duration_ms,
        )


def _require_invoke_identity_field(value: str, label: str) -> str:
    if not value or not value.strip():
        raise ValueError(
            f"catalog declarative invoke requires explicit {label}",
        )
    return value.strip()
