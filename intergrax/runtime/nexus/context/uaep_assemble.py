# © Artur Czarnecki. All rights reserved.

"""UAEP session context assembly via ContextEngine (CE-UAEP-ASM)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextProviderContext,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.context.protocols import ContextEngine
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.graph_assembly import text_from_assembled_messages
from intergrax.runtime.nexus.context.provider_handles import build_graph_provider_handles
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


def build_uaep_assembly_request(
    request: RuntimeRequest,
    *,
    agent_id: str,
    assembly_options: TaskContextAssemblyOptions | None = None,
) -> ContextAssemblyRequest:
    options = assembly_options or TaskContextAssemblyOptions()
    run_id = str(request.metadata.get("run_id") or request.run_id)
    task_id = str(request.metadata.get("task_id") or request.task_id)
    return ContextAssemblyRequest(
        trace_id=run_id,
        run_id=run_id,
        task_id=task_id,
        tenant_id=str(request.tenant_id or request.metadata.get("tenant_id") or "default"),
        assembly_scope="uaep_turn",
        objective=request.message or "",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_chars=max(options.max_prior_chars, 4000)),
        assembly_options=options,
        step_index=0,
        step_kind="uaep_session",
    )


async def assemble_uaep_session_messages(
    request: RuntimeRequest,
    *,
    agent_id: str,
    engine: ContextEngine,
    llm_adapter: LLMAdapter,
    event_bus: RuntimeEventBus | None = None,
    assembly_options: TaskContextAssemblyOptions | None = None,
) -> tuple[ChatMessage, ...]:
    """Run ``ContextEngine.assemble`` for a UAEP session turn and return exact messages."""
    assembly_request = build_uaep_assembly_request(
        request,
        agent_id=agent_id,
        assembly_options=assembly_options,
    )
    runtime_config = RuntimeConfig(llm_adapter=llm_adapter, production_mode=False)
    base_message = request.message or ""
    engine_id = engine.engine_id
    provider_ctx = ContextProviderContext(
        engine_id=engine_id,
        handles=build_graph_provider_handles(
            _task_stub_from_request(request),
            runtime_config=runtime_config,
            messages=[ChatMessage(role="user", content=base_message)],
            event_bus=event_bus,
            node_id=agent_id,
            agent_id=agent_id,
            engine_id=engine_id,
        ),
    )
    assembled = await engine.assemble(assembly_request, provider_ctx=provider_ctx)
    return assembled.messages


async def assemble_uaep_session_prompt(
    request: RuntimeRequest,
    *,
    agent_id: str,
    engine: ContextEngine,
    llm_adapter: LLMAdapter,
    event_bus: RuntimeEventBus | None = None,
    assembly_options: TaskContextAssemblyOptions | None = None,
) -> str:
    """Compatibility wrapper — string projection only when losslessly allowed."""
    assembled_messages = await assemble_uaep_session_messages(
        request,
        agent_id=agent_id,
        engine=engine,
        llm_adapter=llm_adapter,
        event_bus=event_bus,
        assembly_options=assembly_options,
    )
    base_message = request.message or ""
    if not assembled_messages:
        return base_message
    projected = text_from_assembled_messages(assembled_messages)
    return projected or base_message


def _task_stub_from_request(request: RuntimeRequest):
    """Minimal task view for provider handle extraction."""
    from intergrax.runtime.task.task import Task, TaskContext

    task_id = str(request.metadata.get("task_id") or request.task_id)
    metadata = {
        k: v
        for k, v in request.metadata.items()
        if k
        in {
            "workspace_files",
            "memory_profile",
            "session_vector_hits",
            "session_history_snapshot",
            "session_context_revision_id",
            "session_history_messages",
            "rag_chunks",
            "ltm_entries",
            "websearch_blocks",
            "tool_output_blocks",
            "system_instructions",
            "policy_overlay_fragments",
            "attachment_summaries",
        }
    }
    return Task(
        tenant_id=str(request.tenant_id or "default"),
        user_id=str(request.metadata.get("user_id") or "user"),
        session_id=request.session_id,
        message=request.message or "",
        context=TaskContext(),
        metadata=metadata,
        task_id=task_id,
    )
