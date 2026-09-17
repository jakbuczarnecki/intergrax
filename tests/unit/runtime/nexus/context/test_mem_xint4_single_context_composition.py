# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-4 — single canonical Context Engine composition (no legacy direct injection)."""

from __future__ import annotations

import inspect
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intergrax.context.bootstrap import materialize_context_plugin_registry
from intergrax.context.contracts import (
    AssembledContext,
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextProviderContext,
    IterativeToolOutputBlock,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.memory_control import (
    MemoryControlPlaneScope,
    MemoryControlRecallItem,
    MemoryControlRecallResult,
)
from intergrax.memory.user_profile_memory import MemoryKind
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.canonical_context_composition import (
    enforce_context_engine_when_provider_sources_active,
)
from intergrax.runtime.nexus.context.context_builder import BuiltContext, RetrievedChunk
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.memory_context_invocation import run_longterm_memory_context
from intergrax.context.source_inputs import ContextMemoryEntryInput, ContextProviderSourceInputs
from intergrax.runtime.nexus.context.provider_handles import (
    LTM_ENTRIES_METADATA_KEY,
    RAG_CHUNKS_METADATA_KEY,
)
from intergrax.runtime.nexus.context.runtime_state_handle_bridge import (
    merge_provider_metadata_into_request,
)
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState, ToolCallTrace
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.tools.plan_context_invocation import run_rag_context, run_tools_context
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationResult
from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan, ToolRuntime
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    build_runtime_request_for_tests,
    build_runtime_state_for_tests,
    canonical_execution_identity_scope,
    canonical_governed_execution_scope,
    canonical_run_id_for_tests,
)
from testing_support.memory_control_plane_test_stub import MemoryControlPlaneTestStub

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CANONICAL_TENANT = "tenant-a"
_CANONICAL_USER = "user-u1"
_LTM_SNIPPET = "prefers dark mode"


def _canonical_identity() -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_CANONICAL_TENANT,
        user_id=_CANONICAL_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_CANONICAL_USER,
    )


class _LtPlane(MemoryControlPlaneTestStub):
    async def recall(self, identity, scope, request):
        return MemoryControlRecallResult(
            scope=MemoryControlPlaneScope.USER,
            items=(
                MemoryControlRecallItem(
                    entry_id="ltm-1",
                    content=_LTM_SNIPPET,
                    kind=MemoryKind.USER_FACT,
                    score=0.9,
                ),
            ),
            reason="hits",
        )


class RecordingContextEngine:
    """Injectable ContextEngine contract double for assemble invocation proofs."""

    def __init__(self) -> None:
        self.assemble_calls = 0
        self._inner = DefaultNexusContextEngine(
            engine_id="recording",
            registry=materialize_context_plugin_registry(["intergrax.builtin"]),
        )

    @property
    def engine_id(self) -> str:
        return "recording"

    @property
    def registry(self):
        return self._inner.registry

    async def assemble(
        self,
        request: ContextAssemblyRequest,
        *,
        provider_ctx: ContextProviderContext,
    ) -> AssembledContext:
        self.assemble_calls += 1
        return await self._inner.assemble(request, provider_ctx=provider_ctx)


def test_ast_guards_no_canonical_direct_injection_helpers() -> None:
    memory_src = inspect.getsource(
        __import__(
            "intergrax.runtime.nexus.context.memory_context_invocation",
            fromlist=["*"],
        )
    )
    assert "insert_context_before_last_user" not in memory_src
    assert "build_user_longterm_memory_prompt" not in memory_src

    plan_src = inspect.getsource(
        __import__(
            "intergrax.runtime.nexus.tools.plan_context_invocation",
            fromlist=["*"],
        )
    )
    assert "insert_context_before_last_user" not in plan_src
    assert "build_rag_prompt" not in plan_src
    assert "inject_tool_traces_system_context" not in plan_src

    catalog_src = inspect.getsource(
        __import__(
            "intergrax.runtime.nexus.tools.catalog_context",
            fromlist=["*"],
        )
    )
    assert "insert_context_before_last_user" not in catalog_src


@pytest.mark.asyncio
async def test_ltm_recall_does_not_mutate_messages_for_llm() -> None:
    plane = _LtPlane()
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_CANONICAL_TENANT,
            user_id=_CANONICAL_USER,
            message="prefs",
        ),
        canonical_identity=_canonical_identity(),
    )
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_user_longterm_memory=True,
        tool_wiring_context=ToolWiringContext(extras={"memory_control_plane": plane}),
    )
    ctx = RuntimeContext.build(config=config, session_manager=build_in_memory_session_manager())
    run_id = canonical_run_id_for_tests("run-ltm")
    state = RuntimeState(
        context=ctx,
        request=request,
        run_id=run_id,
        messages_for_llm=[ChatMessage(role="user", content="prefs")],
    )
    builder = MagicMock()
    builder.build_user_longterm_memory_prompt = MagicMock()
    state.context.user_longterm_memory_prompt_builder = builder

    with canonical_execution_identity_scope(run_id):
        await run_longterm_memory_context(state)

    builder.build_user_longterm_memory_prompt.assert_not_called()
    assert len(state.messages_for_llm) == 1
    merge_provider_metadata_into_request(state)
    assert LTM_ENTRIES_METADATA_KEY in state.request.metadata


@pytest.mark.asyncio
async def test_rag_step_stages_chunks_without_message_injection() -> None:
    request = build_runtime_request_for_tests(message="docs?")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        embedding_manager=MagicMock(),
        vectorstore_manager=MagicMock(),
        production_mode=False,
        enable_rag=True,
    )
    ctx = RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
        context_builder=MagicMock(),
        rag_prompt_builder=MagicMock(),
    )
    chunk = RetrievedChunk(id="c1", text="chunk body", metadata={}, score=0.8)
    run_id = canonical_run_id_for_tests("run-rag")
    state = RuntimeState(
        context=ctx,
        request=request,
        run_id=run_id,
        session=ChatSession(id="s1", tenant_id="t", user_id="u"),
        messages_for_llm=[ChatMessage(role="user", content="docs?")],
        context_builder_result=BuiltContext(
            history_messages=[],
            retrieved_chunks=[chunk],
            rag_used=True,
            rag_reason="hits",
        ),
    )

    with canonical_governed_execution_scope(run_id):
        await run_rag_context(state)

    ctx.rag_prompt_builder.build_rag_prompt.assert_not_called()
    assert len(state.messages_for_llm) == 1
    merge_provider_metadata_into_request(state)
    assert RAG_CHUNKS_METADATA_KEY in state.request.metadata


@pytest.mark.asyncio
async def test_ltm_fragment_appears_once_in_ce_assembly() -> None:
    engine = DefaultNexusContextEngine(
        registry=materialize_context_plugin_registry(["intergrax.builtin"]),
    )
    request = ContextAssemblyRequest(
        trace_id="t1",
        run_id="t1",
        task_id="task",
        tenant_id=_CANONICAL_TENANT,
        assembly_scope="test",
        objective="hello",
        decision_profile=ContextDecisionSnapshot(prefer_longterm_memory=True),
        budget_policy=ContextBudgetSnapshot(max_chars=8000, max_tokens_estimate=2000),
        assembly_options=TaskContextAssemblyOptions(),
    )
    runtime_config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    sources = ContextProviderSourceInputs(
        memory=(
            ContextMemoryEntryInput(
                entry_id="e1",
                content=_LTM_SNIPPET,
                kind="user_fact",
            ),
        ),
    )
    handles = {
        "messages": [ChatMessage(role="user", content="hello")],
        "runtime_config": runtime_config,
    }
    provider_ctx = ContextProviderContext(engine_id="default", sources=sources, handles=handles)
    assembled = await engine.assemble(request, provider_ctx=provider_ctx)
    joined = "\n".join(m.content or "" for m in assembled.messages)
    assert joined.count(_LTM_SNIPPET) == 1


@pytest.mark.asyncio
async def test_fail_closed_without_context_engine_when_sources_active() -> None:
    state = build_runtime_state_for_tests(run_id="run-fail")
    state.context.config.context_engine = None
    from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry

    state.user_longterm_memory_result = {
        "used_longterm": True,
        "hits": [
            UserProfileMemoryEntry(entry_id="e1", content="x", kind=MemoryKind.USER_FACT),
        ],
        "scores": [0.9],
        "debug": {"used": True},
    }
    with pytest.raises(RuntimeError, match="context_engine is required"):
        enforce_context_engine_when_provider_sources_active(state)


@pytest.mark.asyncio
async def test_custom_recording_context_engine_used_by_ce_tool_loop() -> None:
    recording = RecordingContextEngine()
    state = build_runtime_state_for_tests(run_id="run-ce")
    state.context.config.llm_adapter = FakeLLMAdapter()
    state.context.config.context_engine = recording
    state.context.config.max_tool_iterations = 2
    state.iterative_tool_output_blocks.append(
        IterativeToolOutputBlock(
            content="tool observation",
            tool_call_id="tc-1",
            tool_name="probe",
            step_id="s1",
        )
    )
    messages = [ChatMessage(role="user", content="go")]
    from intergrax.runtime.nexus.context.iterative_tool_context_assembly import (
        assemble_iterative_tool_planner_messages,
    )

    await assemble_iterative_tool_planner_messages(state, recording, messages)
    assert recording.assemble_calls == 1


@pytest.mark.asyncio
async def test_tool_runtime_enforces_context_engine_when_ltm_active() -> None:
    plane = _LtPlane()
    request = replace(
        build_runtime_request_for_tests(
            tenant_id=_CANONICAL_TENANT,
            user_id=_CANONICAL_USER,
            message="prefs",
        ),
        canonical_identity=_canonical_identity(),
    )
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_user_longterm_memory=True,
        tool_wiring_context=ToolWiringContext(extras={"memory_control_plane": plane}),
    )
    ctx = RuntimeContext.build(config=config, session_manager=build_in_memory_session_manager())
    run_id = canonical_run_id_for_tests("run-enforce")
    state = RuntimeState(context=ctx, request=request, run_id=run_id)

    plan = ToolInvocationPlan.from_tool_ids([])

    with (
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_rag_context",
            new_callable=AsyncMock,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_websearch_context",
            new_callable=AsyncMock,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_tools_context",
            new_callable=AsyncMock,
        ),
        canonical_governed_execution_scope("run-enforce"),
        pytest.raises(RuntimeError, match="context_engine is required"),
    ):
        await ToolRuntime.invoke(state=state, plan=plan)


@pytest.mark.asyncio
async def test_run_tools_context_stages_tool_output_for_ce_not_system_injection() -> None:
    state = build_runtime_state_for_tests(run_id="run-tools")
    state.context.config.tools_mode = "auto"
    state.context.config.tool_invoker = object()
    state.context.config.tool_planner = object()
    state.messages_for_llm = [ChatMessage(role="user", content="use tool")]
    trace = ToolCallTrace(
        tool_name="probe",
        arguments={},
        output_preview="full output",
        success=True,
        error_message=None,
        raw_trace={},
    )
    loop_result = ToolInvocationResult(
        tool_traces=[trace],
        loop_iterations=1,
        stop_reason="legacy_single_pass",
    )

    with (
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.resolve_tool_planner_input",
            return_value=state.messages_for_llm,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.resolve_tool_registry",
            return_value=None,
        ),
        patch(
            "intergrax.runtime.nexus.tools.plan_context_invocation.run_bounded_tool_loop_async",
            return_value=loop_result,
        ),
        canonical_governed_execution_scope("run-tools"),
    ):
        await run_tools_context(state)

    assert len(state.messages_for_llm) == 1
    assert state.tools_context_parts
    assert "full output" in state.tools_context_parts[0]


@pytest.mark.asyncio
async def test_ce_budget_excludes_ltm_when_tight() -> None:
    engine = DefaultNexusContextEngine(
        registry=materialize_context_plugin_registry(["intergrax.builtin"]),
    )
    request = ContextAssemblyRequest(
        trace_id="t2",
        run_id="t2",
        task_id="task",
        tenant_id=_CANONICAL_TENANT,
        assembly_scope="test",
        objective="hello",
        decision_profile=ContextDecisionSnapshot(
            prefer_longterm_memory=True,
            max_memory_entries_in_context=4,
        ),
        budget_policy=ContextBudgetSnapshot(max_chars=12, max_tokens_estimate=3),
        assembly_options=TaskContextAssemblyOptions(),
    )
    runtime_config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    sources = ContextProviderSourceInputs(
        memory=(
            ContextMemoryEntryInput(
                entry_id="e1",
                content="x" * 200,
                kind="user_fact",
            ),
        ),
    )
    handles = {
        "messages": [ChatMessage(role="user", content="hello")],
        "runtime_config": runtime_config,
    }
    provider_ctx = ContextProviderContext(engine_id="default", sources=sources, handles=handles)
    assembled = await engine.assemble(request, provider_ctx=provider_ctx)
    joined = "\n".join(m.content or "" for m in assembled.messages)
    assert assembled.degradation_steps or len(joined) < 80
