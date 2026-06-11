# © Artur Czarnecki. All rights reserved.

"""Catalog-backed ``DeclarativeToolInvoker`` for ACP host wiring (ACP-PROD-2 depth)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from intergrax.agents.persistence.declarative_tool_executor import (
    DeclarativeToolInvokeResult,
    DeclarativeToolInvoker,
)
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tools.catalog_dispatch import invoke_catalog_tool_request
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.registry import ToolRegistry


class _CatalogDispatchLLMStub(LLMAdapter):
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
    """Mutable run scope rebound before each ACP session or graph node."""

    run_id: str = ""
    agent_id: str = ""
    tenant_id: str = "default"
    user_id: str = ""


@dataclass
class CatalogDeclarativeToolInvoker:
    """Invoke declarative actions through the Tier-1 catalog tool gateway."""

    tool_invoker: object
    binding: CatalogDeclarativeRunBinding = field(default_factory=CatalogDeclarativeRunBinding)

    def bind_run(
        self,
        *,
        run_id: str,
        agent_id: str,
        tenant_id: str = "default",
        user_id: str = "",
    ) -> None:
        self.binding.run_id = run_id
        self.binding.agent_id = agent_id
        self.binding.tenant_id = tenant_id
        self.binding.user_id = user_id

    def _runtime_state(self) -> RuntimeState:
        from intergrax.prompts.registry.prompt_registry_resolver import (
            resolve_yaml_prompt_registry,
        )
        from intergrax.runtime.nexus.budget.production_budget_policy import (
            ensure_production_run_budget,
        )

        config = RuntimeConfig(
            llm_adapter=_CatalogDispatchLLMStub(),
            production_mode=False,
            enable_rag=False,
            enable_websearch=False,
            tool_invoker=self.tool_invoker,
            tenant_id=self.binding.tenant_id,
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
        return RuntimeState(
            context=context,
            request=RuntimeRequest(
                agent_id=self.binding.agent_id or "agent",
                user_id=self.binding.user_id or "user",
                session_id=self.binding.run_id or "session",
                tenant_id=self.binding.tenant_id,
                message="acp.declarative",
            ),
            run_id=self.binding.run_id or "run",
            tool_traces=[],
        )

    async def invoke(
        self,
        *,
        tool_id: str,
        args: dict[str, Any],
        idempotency_key: str | None,
    ) -> DeclarativeToolInvokeResult:
        request = ToolRequest(
            tool_name=tool_id,
            agent_id=self.binding.agent_id or "agent",
            step_id="acp.declarative",
            input=args,
            idempotency_key=idempotency_key,
        )
        response = invoke_catalog_tool_request(
            state=self._runtime_state(),
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


def build_catalog_declarative_invoker_from_registry(
    registry: ToolRegistry,
) -> CatalogDeclarativeToolInvoker:
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    return CatalogDeclarativeToolInvoker(tool_invoker=invoker)


def resolve_declarative_tool_invoker(
    candidate: object | None,
) -> DeclarativeToolInvoker | None:
    if candidate is None:
        return None
    if isinstance(candidate, DeclarativeToolInvoker):
        return candidate
    raise TypeError("declarative tool invoker metadata must implement DeclarativeToolInvoker")
