# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-6 cross-layer certification helpers (tests only)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intergrax.context.bootstrap import materialize_context_plugin_registry
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextAuthorityClass,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
    IterativeToolOutputBlock,
)
from intergrax.context.source_inputs import (
    ContextMemoryEntryInput,
    ContextProviderSourceInputs,
    ContextRagChunkInput,
    ContextSessionSourceInput,
    ContextWebSearchResultInput,
)
from intergrax.context.session_history import (
    SessionHistoryMessage,
    SessionHistorySnapshot,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_RUNTIME_SCAN_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context",
    REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools",
    REPO_ROOT / "intergrax" / "runtime" / "nexus" / "engine",
    REPO_ROOT / "intergrax" / "runtime" / "execution",
)

LEGACY_BYPASS_SYMBOLS: tuple[str, ...] = (
    "insert_context_before_last_user",
    "append_native_tool_messages",
    "inject_tool_traces_system_context",
    "format_rag_context",
    "build_rag_prompt",
    "build_user_longterm_memory_prompt",
    "search_user_longterm_memory",
)

DEFINITION_ONLY_MODULES: frozenset[str] = frozenset(
    {
        "tool_context_helpers.py",
        "tool_loop.py",
        "rag_prompt_builder.py",
        "user_longterm_memory_prompt_builder.py",
        "session_manager.py",
    }
)


@dataclass(slots=True)
class BypassInventoryRow:
    symbol: str
    call_count: int
    classification: str


@dataclass(slots=True)
class CrossLayerCertificationEvidence:
    """Typed snapshot for certification assertions and final report excerpts."""

    scenario_id: str
    assemble_calls: int = 0
    plane_recall_calls: int = 0
    manager_direct_recall_calls: int = 0
    included_fragment_ids: tuple[str, ...] = ()
    policy_stage_count: int = 0
    model_message_count: int = 0
    notes: tuple[str, ...] = ()


def build_certification_engine(engine_id: str = "mem-xint6") -> DefaultNexusContextEngine:
    return DefaultNexusContextEngine(
        engine_id=engine_id,
        registry=materialize_context_plugin_registry(["intergrax.builtin"]),
    )


def certification_assembly_request(
    *,
    tenant_id: str = "tenant-cert",
    user_id: str = "user-cert",
    run_id: str = "run-cert",
    task_id: str = "task-cert",
    max_tokens: int = 8000,
    objective: str = "certify cross-layer context",
) -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id=run_id,
        run_id=run_id,
        task_id=task_id,
        tenant_id=tenant_id,
        user_id=user_id,
        assembly_scope="acp_step",
        objective=objective,
        decision_profile=ContextDecisionSnapshot(
            prefer_longterm_memory=True,
            include_session_history=True,
        ),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=max_tokens),
        assembly_options=TaskContextAssemblyOptions(),
    )


def build_provider_source_inputs(
    *,
    ltm_entries: tuple[ContextMemoryEntryInput, ...] | None = None,
    rag_chunks: tuple[ContextRagChunkInput, ...] | None = None,
    tool_blocks: tuple[IterativeToolOutputBlock, ...] | None = None,
    web_blocks: tuple[ContextWebSearchResultInput, ...] | None = None,
    session_snapshot: SessionHistorySnapshot | None = None,
) -> ContextProviderSourceInputs:
    session = None
    if session_snapshot is not None:
        session = ContextSessionSourceInput(
            snapshot=session_snapshot,
            binding_context_scope_id=session_snapshot.context_scope_id,
            binding_revision_id=session_snapshot.revision_id,
        )
    return ContextProviderSourceInputs(
        memory=ltm_entries or (),
        rag=rag_chunks or (),
        tools=tool_blocks or (),
        web=web_blocks or (),
        session=session,
    )


def build_provider_handles(
    *,
    runtime_config: RuntimeConfig,
    messages: list[ChatMessage] | None = None,
) -> dict[str, Any]:
    """Auxiliary handles only — semantic payloads must use ``sources``."""
    return {
        "runtime_config": runtime_config,
        "messages": messages or [ChatMessage(role="user", content="certify")],
    }


def build_certification_provider_context(
    *,
    runtime_config: RuntimeConfig,
    messages: list[ChatMessage] | None = None,
    sources: ContextProviderSourceInputs,
    engine_id: str = "mem-xint6",
) -> ContextProviderContext:
    return ContextProviderContext(
        engine_id=engine_id,
        sources=sources,
        handles=build_provider_handles(runtime_config=runtime_config, messages=messages),
    )


def session_snapshot_for_cert(
    *,
    tenant_id: str,
    session_id: str,
    revision_id: str,
    episodic_fact: str,
) -> SessionHistorySnapshot:
    return SessionHistorySnapshot(
        tenant_id=tenant_id,
        context_scope_id=session_id,
        revision_id=revision_id,
        messages=(
            SessionHistoryMessage(
                message_id="sh-1",
                sequence=0,
                role="user",
                content=episodic_fact,
            ),
        ),
    )


def authority_for_source(source: ContextFragmentSource) -> ContextAuthorityClass:
    mapping = {
        ContextFragmentSource.LONGTERM_MEMORY: ContextAuthorityClass.CANONICAL_MEMORY,
        ContextFragmentSource.SESSION_HISTORY: ContextAuthorityClass.SESSION_EPISODIC,
        ContextFragmentSource.RAG: ContextAuthorityClass.RAG_EVIDENCE,
        ContextFragmentSource.TOOL_OUTPUT: ContextAuthorityClass.TOOL_OBSERVATION,
        ContextFragmentSource.WEBSEARCH: ContextAuthorityClass.RAG_EVIDENCE,
        ContextFragmentSource.SYSTEM_INSTRUCTIONS: ContextAuthorityClass.SYSTEM_CONTEXT,
    }
    return mapping.get(source, ContextAuthorityClass.UNASSIGNED)


def assert_included_fragments_have_provenance(fragments: tuple[ContextFragment, ...]) -> None:
    for fragment in fragments:
        if fragment.source is ContextFragmentSource.SYSTEM_INSTRUCTIONS:
            continue
        assert fragment.provider_provenance is not None or fragment.source_id


def count_symbol_calls_in_file(path: Path, symbol: str) -> int:
    source = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"\b{re.escape(symbol)}\s*\(")
    return len(pattern.findall(source))


def inventory_legacy_bypass_symbols() -> list[BypassInventoryRow]:
    rows: list[BypassInventoryRow] = []
    for symbol in LEGACY_BYPASS_SYMBOLS:
        total_calls = 0
        definition_only = False
        for root in CANONICAL_RUNTIME_SCAN_ROOTS:
            if not root.is_dir():
                continue
            for path in root.rglob("*.py"):
                if path.name in DEFINITION_ONLY_MODULES:
                    calls = count_symbol_calls_in_file(path, symbol)
                    if calls == 1 and symbol in path.read_text(encoding="utf-8"):
                        definition_only = True
                    continue
                total_calls += count_symbol_calls_in_file(path, symbol)
        if total_calls == 0:
            classification = "canonical_runtime_bypass_absent"
        elif definition_only and symbol in {
            "append_native_tool_messages",
            "insert_context_before_last_user",
            "format_rag_context",
            "build_rag_prompt",
            "build_user_longterm_memory_prompt",
            "search_user_longterm_memory",
        }:
            classification = "legacy_definition_only"
        else:
            classification = "review_required"
        rows.append(BypassInventoryRow(symbol=symbol, call_count=total_calls, classification=classification))
    return rows


@dataclass(slots=True)
class RecordingAssembleEngine:
    """ContextEngine double that counts ``assemble`` while delegating to the default engine."""

    inner: DefaultNexusContextEngine
    assemble_calls: int = 0

    @property
    def engine_id(self) -> str:
        return self.inner.engine_id

    @property
    def registry(self):
        return self.inner.registry

    async def assemble(self, request: ContextAssemblyRequest, *, provider_ctx: ContextProviderContext):
        self.assemble_calls += 1
        return await self.inner.assemble(request, provider_ctx=provider_ctx)


def recording_engine() -> RecordingAssembleEngine:
    return RecordingAssembleEngine(inner=build_certification_engine())


@dataclass
class RecordingRecallPlaneCalls:
    recall_calls: int = 0

    def record_recall(self) -> None:
        self.recall_calls += 1