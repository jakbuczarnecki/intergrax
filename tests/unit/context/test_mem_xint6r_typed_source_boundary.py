# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-6-R typed provider source boundary guards and collector behavior."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.context.bootstrap import materialize_context_plugin_registry
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragmentSource,
    ContextProviderContext,
    IterativeToolOutputBlock,
)
from intergrax.context.policy.canonicalization import canonicalize_fragment_for_policy
from intergrax.context.providers import builtin as builtin_mod
from intergrax.context.source_inputs import (
    ContextAttachmentSummaryInput,
    ContextMemoryEntryInput,
    ContextPolicyOverlayInput,
    ContextPriorOutputInput,
    ContextProviderSourceInputs,
    ContextRagChunkInput,
    ContextRagCitationField,
    ContextSessionSourceInput,
    ContextSharedContextReadInput,
    ContextSystemInstructionsInput,
    ContextWebSearchResultInput,
)
from intergrax.context.session_history import (
    build_session_history_snapshot,
    fragments_from_session_history_snapshot,
    session_history_content_hash_for_fragment,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_SEMANTIC_HANDLE_CONSTANTS = (
    "LTM_ENTRIES_HANDLE",
    "RAG_CHUNKS_HANDLE",
    "TOOL_OUTPUT_BLOCKS_HANDLE",
    "WEBSEARCH_BLOCKS_HANDLE",
    "SESSION_HISTORY_MESSAGES_HANDLE",
    "SYSTEM_INSTRUCTIONS_HANDLE",
    "POLICY_OVERLAY_FRAGMENTS_HANDLE",
    "ATTACHMENT_SUMMARIES_HANDLE",
    "SHARED_CONTEXT_READS_HANDLE",
    "PRIOR_OUTPUT_RECORDS_HANDLE",
)


def _request() -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace",
        run_id="run",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="typed boundary",
        decision_profile=ContextDecisionSnapshot(
            prefer_longterm_memory=True,
            prefer_rag_when_enabled=True,
            include_session_history=True,
        ),
        budget_policy=ContextBudgetSnapshot(max_chars=8000),
        assembly_options=TaskContextAssemblyOptions(),
    )


def _aux_handles() -> dict[str, object]:
    return {
        "runtime_config": RuntimeConfig(llm_adapter=object(), production_mode=False),
        "messages": [ChatMessage(role="user", content="typed boundary")],
    }


def _sources_full() -> ContextProviderSourceInputs:
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="session", entry_id="m1")],
    )
    return ContextProviderSourceInputs(
        memory=(
            ContextMemoryEntryInput(entry_id="m1", content="memory fact", kind="user_fact"),
        ),
        rag=(
            ContextRagChunkInput(
                chunk_id="c1",
                content="rag body",
                citation_fields=(ContextRagCitationField(key="doc_id", value="d1"),),
            ),
        ),
        tools=(
            IterativeToolOutputBlock(
                content="tool output",
                tool_call_id="tc-1",
                tool_name="probe",
                step_id=None,
            ),
        ),
        web=(ContextWebSearchResultInput(source_id="w1", content="web hit"),),
        session=ContextSessionSourceInput(
            snapshot=snapshot,
            binding_context_scope_id="scope",
            binding_revision_id="rev",
        ),
        system=ContextSystemInstructionsInput(text="system policy"),
        attachments=(
            ContextAttachmentSummaryInput(attachment_id="a1", summary="attachment summary"),
        ),
        shared_context=(ContextSharedContextReadInput(entry_key="dep", content="shared"),),
        policy_overlay=(ContextPolicyOverlayInput(overlay_id="p1", content="overlay", priority=10),),
        graph_prior=(ContextPriorOutputInput(node_id="n1", content="prior", agent_id="agent"),),
    )


def test_builtin_has_zero_semantic_handle_reads() -> None:
    source = inspect.getsource(builtin_mod)
    for literal in (
        "ltm_entries",
        "rag_chunks",
        "tool_output_blocks",
        "websearch_blocks",
        "session_history_messages",
        "system_instructions",
    ):
        assert f'handles.get("{literal}"' not in source
    for constant in _SEMANTIC_HANDLE_CONSTANTS:
        assert constant not in source
    assert "legacy_bridge" not in source


def test_builtin_guard_script_passes() -> None:
    script = _REPO / "scripts" / "maintenance" / "check_mem_xint6_typed_source_boundary.py"
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
async def test_semantic_handles_ignored_when_only_handles_populated() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    providers = {provider.provider_id: provider for provider in registry.list_providers()}
    request = _request()
    ctx = ContextProviderContext(
        engine_id="default",
        handles={
            **_aux_handles(),
            "ltm_entries": [{"entry_id": "x", "content": "must-not-appear"}],
            "rag_chunks": [{"text": "must-not-appear"}],
        },
    )
    assert await providers["builtin.longterm_memory"].collect(request, ctx) == []
    assert await providers["builtin.rag"].collect(request, ctx) == []


@pytest.mark.asyncio
async def test_memory_and_rag_collectors_read_typed_slots() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    providers = {provider.provider_id: provider for provider in registry.list_providers()}
    request = _request()
    ctx = ContextProviderContext(engine_id="default", sources=_sources_full(), handles=_aux_handles())
    memory = await providers["builtin.longterm_memory"].collect(request, ctx)
    rag = await providers["builtin.rag"].collect(request, ctx)
    assert memory and memory[0].source is ContextFragmentSource.LONGTERM_MEMORY
    assert rag and rag[0].source is ContextFragmentSource.RAG
    assert "memory fact" in memory[0].content
    assert "rag body" in rag[0].content


@pytest.mark.asyncio
async def test_provider_isolation_memory_does_not_emit_rag() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    providers = {provider.provider_id: provider for provider in registry.list_providers()}
    request = _request()
    sources = ContextProviderSourceInputs(
        memory=(ContextMemoryEntryInput(entry_id="m1", content="only-memory"),),
        rag=(ContextRagChunkInput(chunk_id="c1", content="only-rag"),),
    )
    ctx = ContextProviderContext(engine_id="default", sources=sources, handles=_aux_handles())
    memory = await providers["builtin.longterm_memory"].collect(request, ctx)
    assert memory and all(fragment.source is ContextFragmentSource.LONGTERM_MEMORY for fragment in memory)
    assert "only-rag" not in memory[0].content


@pytest.mark.asyncio
async def test_empty_sources_safe_for_all_builtin_collectors() -> None:
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    request = _request()
    ctx = ContextProviderContext(engine_id="default", sources=ContextProviderSourceInputs(), handles=_aux_handles())
    for provider in registry.list_providers():
        if provider.provider_id.startswith("builtin.") and provider.provider_id != "builtin.workspace":
            fragments = await provider.collect(request, ctx)
            assert isinstance(fragments, list)


def test_typed_session_fragment_survives_policy_canonicalization_hash() -> None:
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="  session body  \n", entry_id="m1")],
    )
    fragment = fragments_from_session_history_snapshot(snapshot)[0]
    canonical = canonicalize_fragment_for_policy(fragment)
    assert canonical.content_hash == session_history_content_hash_for_fragment(canonical)
    assert canonical.metadata.get("content_hash") == canonical.content_hash
