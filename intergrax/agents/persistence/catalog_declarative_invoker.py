# © Artur Czarnecki. All rights reserved.

"""Catalog-backed ``DeclarativeToolInvoker`` for ACP host wiring (ACP-PROD-2 depth)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock  # minimal RuntimeContext shim — tools only

from intergrax.agents.persistence.declarative_tool_executor import (
    DeclarativeToolInvokeResult,
    DeclarativeToolInvoker,
)
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.catalog_dispatch import invoke_catalog_tool_request
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.registry import ToolRegistry


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
        config = RuntimeConfig(
            llm_adapter=MagicMock(),
            production_mode=False,
            enable_rag=False,
            enable_websearch=False,
            tool_invoker=self.tool_invoker,
        )
        ctx = RuntimeContext(
            config=config,
            session_manager=MagicMock(),
            prompt_registry=MagicMock(),
        )
        return RuntimeState(
            context=ctx,
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
