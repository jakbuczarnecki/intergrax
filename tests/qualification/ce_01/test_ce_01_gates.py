# © Artur Czarnecki. All rights reserved.

"""CE-01 qualification gates (CE-Q1..CE-Q15 primary evidence)."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.context.contracts import (
    ASSEMBLED_CONTEXT_SCHEMA,
    CONTEXT_CONTRACTS_SCHEMA,
    AssembledContext,
    ContextAssemblyRequest,
    ContextAuthorityClass,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentScopeRef,
    ContextFragmentSource,
    ContextPolicyReasonCode,
    ContextProviderContext,
)
from intergrax.context.policy.exact_dedup import exact_dedup_fragments
from intergrax.context.policy.scope_isolation import isolate_assembly_scope
from intergrax.context.provider_descriptor import build_provider_descriptor
from intergrax.context.protocols import ContextEngine, ContextSourceProvider
from intergrax.context.registry import ContextPluginRegistry
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.assembly_runtime_deps import (
    ContextAssemblyRuntimeDependencies,
    build_context_assembly_runtime_dependencies,
)
from intergrax.runtime.nexus.context.iterative_tool_context_assembly import (
    assemble_iterative_tool_planner_messages,
)
from intergrax.runtime.nexus.context.uaep_assemble import assemble_uaep_session_messages

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTEXT_TIER0 = _REPO_ROOT / "intergrax" / "context"
_CE_CORE_SCAN_ROOTS = (
    _CONTEXT_TIER0,
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "context_engine.py",
)
_FORBIDDEN_CE_TOKENS = ("service_locator", "GLOBAL_REGISTRY")
_FORBIDDEN_VENDOR_PREFIXES = (
    "openai",
    "anthropic",
    "google.genai",
    "google.generativeai",
    "boto3",
    "azure.ai",
    "bedrock",
)
_NEXUS_CANONICAL_CONTEXT_MODULES = (
    "intergrax.runtime.nexus.context.uaep_assemble",
    "intergrax.runtime.nexus.context.iterative_tool_context_assembly",
    "intergrax.runtime.nexus.context.context_manager",
)


class _SmallWindowAdapter(BaseLLMAdapter):
    provider = "fake"
    model = "fake-gate"

    def __init__(self, window: int = 4096) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


def _assembly_request(tenant_id: str = "tenant-a") -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id=tenant_id,
        assembly_scope="acp_step",
        objective="ce-01 gate",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=2000),
        assembly_options=TaskContextAssemblyOptions(),
    )


def _scan_python_files(roots: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        files.extend(root.rglob("*.py"))
    return files


def test_ce_q1_canonical_context_engine_entry_surfaces() -> None:
    assert issubclass(DefaultNexusContextEngine, object)
    engine_protocol = inspect.getsource(ContextEngine)
    assert "assemble" in engine_protocol

    for module_name in _NEXUS_CANONICAL_CONTEXT_MODULES:
        mod = __import__(module_name, fromlist=["*"])
        source = inspect.getsource(mod)
        assert "engine.assemble" in source or "ContextEngine" in source

    uaep_sig = inspect.signature(assemble_uaep_session_messages)
    assert "engine" in uaep_sig.parameters
    iterative_sig = inspect.signature(assemble_iterative_tool_planner_messages)
    assert "engine" in iterative_sig.parameters


class _Q1SpyEngine:
    engine_id = "spy"

    def __init__(self) -> None:
        self.calls = 0
        self.last_request: ContextAssemblyRequest | None = None

    async def assemble(
        self,
        request: ContextAssemblyRequest,
        *,
        provider_ctx: ContextProviderContext | None = None,
    ) -> AssembledContext:
        self.calls += 1
        self.last_request = request
        adapter = _SmallWindowAdapter()
        config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
        runtime = build_context_assembly_runtime_dependencies(
            runtime_config=config,
            messages=[ChatMessage(role="user", content="spy")],
            max_output_tokens=32,
        )
        inner = DefaultNexusContextEngine(engine_id="spy")
        return await inner.assemble(
            request,
            provider_ctx=ContextProviderContext(engine_id="spy", runtime=runtime),
        )


@pytest.mark.asyncio
async def test_ce_q1_behavioral_engine_assemble_on_canonical_surface() -> None:
    from testing_support.builder import build_runtime_request_for_tests

    spy = _Q1SpyEngine()
    request = build_runtime_request_for_tests(
        seed="ce-q1-spy",
        tenant_id="t",
        user_id="u",
        session_id="s",
        agent_id="a",
        message="m",
    )
    await assemble_uaep_session_messages(
        request,
        agent_id="a",
        engine=spy,
        llm_adapter=_SmallWindowAdapter(),
    )
    assert spy.calls == 1
    assert spy.last_request is not None
    assert spy.last_request.assembly_scope == "uaep_turn"


@pytest.mark.asyncio
async def test_ce_q2_typed_context_contracts_are_assembly_abi() -> None:
    """Typed ContextAssemblyRequest / ContextProvider runtime / AssembledContext ABI."""
    request = _assembly_request()
    assert isinstance(request, ContextAssemblyRequest)
    assert request.schema_version == CONTEXT_CONTRACTS_SCHEMA
    assert "runtime" in ContextProviderContext.__dataclass_fields__

    adapter = _SmallWindowAdapter()
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    engine = DefaultNexusContextEngine()
    runtime = build_context_assembly_runtime_dependencies(
        runtime_config=config,
        messages=[ChatMessage(role="user", content="hi")],
        max_output_tokens=64,
    )
    provider_ctx = ContextProviderContext(engine_id="default", runtime=runtime)

    assembled = await engine.assemble(request, provider_ctx=provider_ctx)
    assert isinstance(assembled, AssembledContext)
    assert assembled.schema_version == ASSEMBLED_CONTEXT_SCHEMA
    assert isinstance(assembled.messages, tuple)
    assert isinstance(runtime, ContextAssemblyRuntimeDependencies)

    engine_source = (_REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "context_engine.py").read_text(
        encoding="utf-8"
    )
    for literal in ("runtime_config", "max_output_tokens", "nexus_ucl_runtime", "context_optimization_policy"):
        assert f'handles.get("{literal}"' not in engine_source
    assert 'handles.get("messages"' not in engine_source
    assert "ensure_context_assembly_runtime" not in engine_source
    assert provider_ctx.runtime is not None


def test_ce_q3_foreign_tenant_fragment_rejected() -> None:
    request = _assembly_request(tenant_id="tenant-a")
    foreign = ContextFragment(
        fragment_id="f-foreign",
        source=ContextFragmentSource.RAG,
        source_id="rag-1",
        content="foreign fact",
        token_estimate=4,
        relevance_score=0.8,
        freshness_score=0.8,
        confidence_score=0.8,
        mandatory=False,
        scope_ref=ContextFragmentScopeRef(tenant_id="tenant-b"),
    )
    kept, excluded = isolate_assembly_scope([foreign], request=request)
    assert kept == []
    assert excluded[0][1] == ContextPolicyReasonCode.SCOPE_INCOMPATIBLE.value


def test_ce_q8_authority_precedence_under_ranking() -> None:
    shared_hash = "same-body-hash"
    high = ContextFragment(
        fragment_id="high",
        source=ContextFragmentSource.RAG,
        source_id="s1",
        content="duplicate body",
        token_estimate=4,
        relevance_score=0.2,
        freshness_score=0.8,
        confidence_score=0.8,
        mandatory=False,
        authority_class=ContextAuthorityClass.RAG_EVIDENCE,
        content_hash=shared_hash,
    )
    low = ContextFragment(
        fragment_id="low",
        source=ContextFragmentSource.RAG,
        source_id="s2",
        content="duplicate body",
        token_estimate=4,
        relevance_score=0.9,
        freshness_score=0.8,
        confidence_score=0.8,
        mandatory=False,
        authority_class=ContextAuthorityClass.UNASSIGNED,
        content_hash=shared_hash,
    )
    kept, _, _ = exact_dedup_fragments([low, high])
    assert len(kept) == 1
    assert kept[0].fragment_id == "high"


class _Ce01CustomProvider:
    provider_id = "ce01.custom"

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]:
        return frozenset({ContextFragmentSource.CUSTOM})

    @property
    def descriptor(self):
        return build_provider_descriptor(
            self.provider_id,
            provider_version="1.0.0",
            supported_sources=self.supported_sources,
        )

    async def collect(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        _ = request, ctx
        return [
            ContextFragment(
                fragment_id="custom-1",
                source=ContextFragmentSource.CUSTOM,
                source_id=self.provider_id,
                content="plugin source",
                token_estimate=3,
                relevance_score=0.85,
                freshness_score=0.85,
                confidence_score=0.85,
                mandatory=False,
            )
        ]


@pytest.mark.asyncio
async def test_ce_q9_custom_provider_without_engine_core_change() -> None:
    registry = ContextPluginRegistry()
    registry.add_provider(_Ce01CustomProvider())
    assert isinstance(_Ce01CustomProvider(), ContextSourceProvider)

    adapter = _SmallWindowAdapter()
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    engine = DefaultNexusContextEngine(registry=registry)
    runtime = build_context_assembly_runtime_dependencies(
        runtime_config=config,
        messages=[ChatMessage(role="user", content="hi")],
        max_output_tokens=64,
    )
    provider_ctx = ContextProviderContext(engine_id="default", runtime=runtime)
    base = _assembly_request()
    request = ContextAssemblyRequest(
        trace_id=base.trace_id,
        run_id=base.run_id,
        task_id=base.task_id,
        tenant_id=base.tenant_id,
        assembly_scope="graph_node",
        objective=base.objective,
        decision_profile=base.decision_profile,
        budget_policy=base.budget_policy,
        assembly_options=base.assembly_options,
        graph_node_id="node-ce01",
    )
    assembled = await engine.assemble(request, provider_ctx=provider_ctx)
    assert any(f.source == ContextFragmentSource.CUSTOM for f in assembled.fragments_included)


def test_ce_q11_context_tier0_vendor_import_gate() -> None:
    violations: list[str] = []
    for path in _scan_python_files((_CONTEXT_TIER0,)):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            for prefix in _FORBIDDEN_VENDOR_PREFIXES:
                root = prefix.split(".")[0]
                if f"import {root}" in stripped or f"from {prefix}" in stripped or f"from {root}" in stripped:
                    violations.append(f"{rel}: {stripped}")
                    break
    assert not violations, "\n".join(violations)


def test_ce_q12_llm_adapter_core_no_memory_or_retrieval_fetch() -> None:
    adapter_src = inspect.getsource(LLMAdapter)
    lowered = adapter_src.lower()
    for forbidden in ("vectorstore", "retrievalrequest", "memorycontrol", "rag.retrieve"):
        assert forbidden not in lowered


@pytest.mark.asyncio
async def test_ce_q13_assembled_context_carries_provenance_fields() -> None:
    adapter = _SmallWindowAdapter()
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    registry = ContextPluginRegistry()
    registry.add_provider(_Ce01CustomProvider())
    engine = DefaultNexusContextEngine(registry=registry)
    runtime = build_context_assembly_runtime_dependencies(
        runtime_config=config,
        messages=[ChatMessage(role="user", content="hi")],
        max_output_tokens=64,
    )
    provider_ctx = ContextProviderContext(engine_id="default", runtime=runtime)
    assembled = await engine.assemble(_assembly_request(), provider_ctx=provider_ctx)
    assert assembled.fragments_included
    fragment = assembled.fragments_included[0]
    assert assembled.provenance
    record = assembled.provenance[0]
    assert record.fragment_id == fragment.fragment_id
    assert record.source_type == fragment.source
    assert record.source_id == fragment.source_id
    assert fragment.provider_provenance is not None
    assert record.provider_id == fragment.provider_provenance.provider_id
    assert record.provider_version == fragment.provider_provenance.provider_version
    assert record.content_hash == fragment.content_hash


def test_ce_q14_ce_core_forbidden_integration_gate() -> None:
    script = _REPO_ROOT / "scripts" / "maintenance" / "check_context_tier0_import_boundary.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    semantic_script = _REPO_ROOT / "scripts" / "maintenance" / "check_ce_canonical_semantic_handles.py"
    semantic = subprocess.run(
        [sys.executable, str(semantic_script)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert semantic.returncode == 0, semantic.stdout + semantic.stderr

    violations: list[str] = []
    for path in _scan_python_files(_CE_CORE_SCAN_ROOTS):
        posix = path.as_posix()
        if "legacy_bridge" in posix or "legacy_assembly_runtime_bridge" in posix:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_CE_TOKENS:
            if token in text:
                violations.append(f"{rel}: {token}")
    assert not violations


def test_ce_q15_nexus_canonical_paths_forbid_direct_prompt_injection() -> None:
    forbidden_calls = (
        "build_rag_prompt",
        "build_user_longterm_memory_prompt",
        "insert_context_before_last_user",
        "inject_tool_traces_system_context",
    )
    modules = (
        "intergrax.runtime.nexus.tools.plan_context_invocation",
        "intergrax.runtime.nexus.context.memory_context_invocation",
        "intergrax.runtime.nexus.tools.catalog_context",
    )
    for module_name in modules:
        mod = __import__(module_name, fromlist=["*"])
        source = inspect.getsource(mod)
        for call in forbidden_calls:
            assert call not in source, f"{module_name} still references {call}"

    apps_root = _REPO_ROOT / "intergrax" / "applications"
    if apps_root.is_dir():
        for path in apps_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "DefaultPromptBuilder" in text or "legacy.rag_answers.builders.prompt_builder" in text:
                rel = path.relative_to(_REPO_ROOT).as_posix()
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and "legacy.rag_answers" in node.module:
                        pytest.fail(f"application imports legacy rag prompt pipeline: {rel}")
