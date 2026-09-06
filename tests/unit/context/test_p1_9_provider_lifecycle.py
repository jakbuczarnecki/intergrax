# © Artur Czarnecki. All rights reserved.

"""P1.9 — context provider lifecycle, provenance, and execution pinning."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

import pytest

from intergrax.applications._shared.runtime_inspection.service import RuntimeInspectionService
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
    ContextProviderProvenance,
)
from intergrax.context.errors import (
    ContextProviderContractViolationError,
    ContextProviderRegistrationError,
    RequiredContextSourceUnavailableError,
)
from intergrax.context.contracts import BUILTIN_PROVIDER_VERSION
from intergrax.context.provider_descriptor import build_provider_descriptor, compute_provider_set_fingerprint
from intergrax.context.provider_lifecycle import (
    CONTEXT_PROVIDER_PINNING_STORE_HANDLE,
    BoundContextProvider,
    BoundContextProviderSet,
    InMemoryContextProviderExecutionPinningStore,
    bind_context_provider_set_from_registry,
    is_provider_eligible,
    pin_context_provider_set_for_execution,
    resolve_bound_context_provider_set,
    snapshot_context_provider_set,
)
from intergrax.context.providers.session_semantic_recall import SessionSemanticRecallProvider
from intergrax.context.providers.workspace import WorkspaceContextProvider
from intergrax.context.registry import ContextPluginRegistry
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _SmallWindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake-small"

    def __init__(self, window: int = 4096) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


def _assembly_request(
    *,
    required_sources: frozenset[ContextFragmentSource] = frozenset(),
    excluded_sources: frozenset[ContextFragmentSource] = frozenset(),
    run_id: str = "run_test00000000000000000000000001",
) -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-1",
        run_id=run_id,
        task_id="task_test00000000000000000000000001",
        tenant_id="tenant-a",
        assembly_scope="graph_node",
        objective="objective",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=500),
        assembly_options=TaskContextAssemblyOptions(),
        required_sources=required_sources,
        excluded_sources=excluded_sources,
    )


def _provider_ctx(
    *,
    registry: ContextPluginRegistry | None = None,
    engine_id: str = "default",
    pinning_store: InMemoryContextProviderExecutionPinningStore | None = None,
    extra_handles: dict | None = None,
) -> ContextProviderContext:
    adapter = _SmallWindowAdapter()
    handles = {
        "runtime_config": RuntimeConfig(llm_adapter=adapter, production_mode=False),
        "messages": [ChatMessage(role="user", content="hello")],
        "max_output_tokens": 64,
    }
    if pinning_store is not None:
        handles[CONTEXT_PROVIDER_PINNING_STORE_HANDLE] = pinning_store
    if extra_handles:
        handles.update(extra_handles)
    return ContextProviderContext(engine_id=engine_id, handles=handles)


@dataclass
class _RecordingProvider:
    provider_id: str
    supported: frozenset[ContextFragmentSource]
    version: str = "1.0.0"
    origin: str = "builtin"
    collect_fn: Callable[..., list[ContextFragment]] | None = None
    collect_calls: int = 0

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]:
        return self.supported

    @property
    def descriptor(self):
        return build_provider_descriptor(
            self.provider_id,
            provider_version=self.version,
            supported_sources=self.supported,
            origin=self.origin,
        )

    async def collect(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        _ = request, ctx
        self.collect_calls += 1
        if self.collect_fn is None:
            return []
        result = self.collect_fn(request, ctx)
        if asyncio.iscoroutine(result):
            return await result
        return result


def test_provider_set_fingerprint_order_independent() -> None:
    a = build_provider_descriptor("a.provider", provider_version="1.0.0", supported_sources=frozenset({ContextFragmentSource.WORKSPACE}))
    b = build_provider_descriptor("b.provider", provider_version="2.0.0", supported_sources=frozenset({ContextFragmentSource.RAG}))
    assert compute_provider_set_fingerprint((a, b)) == compute_provider_set_fingerprint((b, a))


def test_provider_version_change_changes_fingerprint() -> None:
    v1 = build_provider_descriptor("a.provider", provider_version="1.0.0", supported_sources=frozenset({ContextFragmentSource.WORKSPACE}))
    v2 = build_provider_descriptor("a.provider", provider_version="2.0.0", supported_sources=frozenset({ContextFragmentSource.WORKSPACE}))
    assert compute_provider_set_fingerprint((v1,)) != compute_provider_set_fingerprint((v2,))


def test_supported_sources_change_changes_fingerprint() -> None:
    ws = build_provider_descriptor("a.provider", provider_version="1.0.0", supported_sources=frozenset({ContextFragmentSource.WORKSPACE}))
    rag = build_provider_descriptor("a.provider", provider_version="1.0.0", supported_sources=frozenset({ContextFragmentSource.RAG}))
    assert compute_provider_set_fingerprint((ws,)) != compute_provider_set_fingerprint((rag,))


def test_registry_rejects_empty_version() -> None:
    registry = ContextPluginRegistry()
    provider = _RecordingProvider(
        provider_id="bad.provider",
        supported=frozenset({ContextFragmentSource.CUSTOM}),
        version="unknown",
    )
    with pytest.raises(ValueError, match="provider_version must be explicit"):
        registry.add_provider(provider)


def test_registry_list_providers_deterministic() -> None:
    registry = ContextPluginRegistry()
    b = _RecordingProvider("b.provider", frozenset({ContextFragmentSource.RAG}))
    a = _RecordingProvider("a.provider", frozenset({ContextFragmentSource.WORKSPACE}))
    registry.add_provider(b)
    registry.add_provider(a)
    first = [item.provider_id for item in registry.list_providers()]
    registry2 = ContextPluginRegistry()
    registry2.add_provider(a)
    registry2.add_provider(b)
    second = [item.provider_id for item in registry2.list_providers()]
    assert first == second == ["a.provider", "b.provider"]


@pytest.mark.asyncio
async def test_excluded_provider_collect_calls_zero() -> None:
    workspace = _RecordingProvider("builtin.workspace", frozenset({ContextFragmentSource.WORKSPACE}))
    registry = ContextPluginRegistry()
    registry.add_provider(workspace)
    engine = DefaultNexusContextEngine(registry=registry)
    request = _assembly_request(excluded_sources=frozenset({ContextFragmentSource.WORKSPACE}))
    await engine.assemble(request, provider_ctx=_provider_ctx())
    assert workspace.collect_calls == 0


@pytest.mark.asyncio
async def test_irrelevant_provider_collect_calls_zero() -> None:
    rag = _RecordingProvider("builtin.rag", frozenset({ContextFragmentSource.RAG}))
    registry = ContextPluginRegistry()
    registry.add_provider(rag)
    engine = DefaultNexusContextEngine(registry=registry)
    request = _assembly_request(required_sources=frozenset({ContextFragmentSource.WORKSPACE}))
    with pytest.raises(RequiredContextSourceUnavailableError, match="required_source.no_provider"):
        await engine.assemble(request, provider_ctx=_provider_ctx())
    assert rag.collect_calls == 0


@pytest.mark.asyncio
async def test_required_source_no_provider_fail_closed() -> None:
    engine = DefaultNexusContextEngine(registry=ContextPluginRegistry())
    request = _assembly_request(required_sources=frozenset({ContextFragmentSource.WORKSPACE}))
    with pytest.raises(RequiredContextSourceUnavailableError, match="required_source.no_provider"):
        await engine.assemble(request, provider_ctx=_provider_ctx())


@pytest.mark.asyncio
async def test_required_provider_failure_fail_closed() -> None:
    async def _fail(_request, _ctx):
        raise RuntimeError("synthetic failure")

    workspace = _RecordingProvider(
        "builtin.workspace",
        frozenset({ContextFragmentSource.WORKSPACE}),
        collect_fn=_fail,
    )
    registry = ContextPluginRegistry()
    registry.add_provider(workspace)
    engine = DefaultNexusContextEngine(registry=registry)
    request = _assembly_request(required_sources=frozenset({ContextFragmentSource.WORKSPACE}))
    with pytest.raises(RequiredContextSourceUnavailableError, match="required_source.provider_failed"):
        await engine.assemble(request, provider_ctx=_provider_ctx())


@pytest.mark.asyncio
async def test_optional_provider_failure_visible() -> None:
    async def _fail(_request, _ctx):
        raise RuntimeError("optional failure")

    optional = _RecordingProvider(
        "optional.provider",
        frozenset({ContextFragmentSource.CUSTOM}),
        collect_fn=_fail,
    )
    registry = ContextPluginRegistry()
    registry.add_provider(optional)
    engine = DefaultNexusContextEngine(registry=registry)
    assembled = await engine.assemble(_assembly_request(), provider_ctx=_provider_ctx())
    failed = [item for item in assembled.provider_outcomes if item.status == "failed"]
    assert failed
    assert failed[0].descriptor.provider_id == "optional.provider"


@pytest.mark.asyncio
async def test_provider_spoof_protection() -> None:
    def _forged(_request, _ctx):
        return [
            ContextFragment(
                fragment_id="frag-1",
                source=ContextFragmentSource.WORKSPACE,
                source_id="path",
                content="body",
                token_estimate=1,
                relevance_score=0.5,
                freshness_score=0.5,
                confidence_score=0.5,
                mandatory=False,
                provider_provenance=ContextProviderProvenance(
                    provider_id="other.provider",
                    provider_version="9.0.0",
                ),
            ),
        ]

    workspace = _RecordingProvider(
        "builtin.workspace",
        frozenset({ContextFragmentSource.WORKSPACE}),
        collect_fn=_forged,
    )
    registry = ContextPluginRegistry()
    registry.add_provider(workspace)
    engine = DefaultNexusContextEngine(registry=registry)
    with pytest.raises(ContextProviderContractViolationError, match="forged provider provenance"):
        await engine.assemble(
            _assembly_request(),
            provider_ctx=_provider_ctx(
                extra_handles={"workspace_files": {"a.py": "x = 1\n"}},
            ),
        )


@pytest.mark.asyncio
async def test_unsupported_source_contract_violation() -> None:
    def _bad_source(_request, _ctx):
        return [
            ContextFragment(
                fragment_id="frag-1",
                source=ContextFragmentSource.RAG,
                source_id="doc-1",
                content="rag body",
                token_estimate=1,
                relevance_score=0.5,
                freshness_score=0.5,
                confidence_score=0.5,
                mandatory=False,
            ),
        ]

    workspace = _RecordingProvider(
        "builtin.workspace",
        frozenset({ContextFragmentSource.WORKSPACE}),
        collect_fn=_bad_source,
    )
    registry = ContextPluginRegistry()
    registry.add_provider(workspace)
    engine = DefaultNexusContextEngine(registry=registry)
    with pytest.raises(ContextProviderContractViolationError):
        await engine.assemble(_assembly_request(), provider_ctx=_provider_ctx())


@pytest.mark.asyncio
async def test_workspace_and_session_providers_adopted() -> None:
    registry = ContextPluginRegistry()
    registry.add_provider(WorkspaceContextProvider())
    registry.add_provider(SessionSemanticRecallProvider())
    workspace_desc = registry.get_provider_descriptor("builtin.workspace")
    session_desc = registry.get_provider_descriptor("builtin.session_history_semantic")
    assert workspace_desc.provider_version == BUILTIN_PROVIDER_VERSION
    assert session_desc.provider_version == BUILTIN_PROVIDER_VERSION


@pytest.mark.asyncio
async def test_fragment_and_assembly_provenance() -> None:
    registry = ContextPluginRegistry()
    registry.add_provider(WorkspaceContextProvider())
    engine = DefaultNexusContextEngine(registry=registry)
    assembled = await engine.assemble(
        _assembly_request(),
        provider_ctx=_provider_ctx(
            extra_handles={"workspace_files": {"handler.py": "def run(): pass\n"}},
        ),
    )
    assert assembled.fragments_included
    fragment = assembled.fragments_included[0]
    assert fragment.provider_provenance is not None
    assert fragment.provider_provenance.provider_id == "builtin.workspace"
    provenance = assembled.provenance[0]
    assert provenance.provider_id == "builtin.workspace"
    assert provenance.provider_version == BUILTIN_PROVIDER_VERSION
    assert provenance.content_hash == fragment.content_hash


@pytest.mark.asyncio
async def test_execution_pinning_registry_replace_does_not_rebind_e1() -> None:
    v1_provider = WorkspaceContextProvider()
    registry = ContextPluginRegistry()
    registry.add_provider(v1_provider)
    pinning_store = InMemoryContextProviderExecutionPinningStore()
    execution_id = mint_execution_id()
    run_id = mint_run_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        request = _assembly_request(run_id=run_id)
        ctx = _provider_ctx(pinning_store=pinning_store, extra_handles={"workspace_files": {"a.py": "x=1\n"}})
        engine = DefaultNexusContextEngine(registry=registry)
        first = await engine.assemble(request, provider_ctx=ctx)
        pinned_version = first.provider_set_snapshot.providers[0].provider_version

        class _WorkspaceV2(WorkspaceContextProvider):
            _PROVIDER_VERSION = "2.0.0"

        registry.remove_provider("builtin.workspace")
        registry.add_provider(_WorkspaceV2(), override=True)

        second = await engine.assemble(request, provider_ctx=ctx)
        assert second.provider_set_snapshot.providers[0].provider_version == pinned_version

        ctx_new = _provider_ctx(
            pinning_store=InMemoryContextProviderExecutionPinningStore(),
            extra_handles={"workspace_files": {"a.py": "x=1\n"}},
        )
        engine_new = DefaultNexusContextEngine(registry=registry)
        token2 = bind_active_execution_identity(
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        )
        try:
            fresh = await engine_new.assemble(_assembly_request(), provider_ctx=ctx_new)
            assert fresh.provider_set_snapshot.providers[0].provider_version == "2.0.0"
        finally:
            reset_active_execution_identity(token2)
    finally:
        reset_active_execution_identity(token)


@pytest.mark.asyncio
async def test_execution_pinning_registry_remove_does_not_rebind_e1() -> None:
    registry = ContextPluginRegistry()
    registry.add_provider(WorkspaceContextProvider())
    pinning_store = InMemoryContextProviderExecutionPinningStore()
    execution_id = mint_execution_id()
    run_id = mint_run_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        request = _assembly_request(run_id=run_id)
        ctx = _provider_ctx(
            pinning_store=pinning_store,
            extra_handles={"workspace_files": {"a.py": "x=1\n"}},
        )
        engine = DefaultNexusContextEngine(registry=registry)
        first = await engine.assemble(request, provider_ctx=ctx)
        assert first.fragments_included
        registry.remove_provider("builtin.workspace")
        second = await engine.assemble(request, provider_ctx=ctx)
        assert second.fragments_included
    finally:
        reset_active_execution_identity(token)


def test_inspection_shows_bound_provider_set_not_current_registry() -> None:
    registry = ContextPluginRegistry()
    registry.add_provider(WorkspaceContextProvider())
    pinning_store = InMemoryContextProviderExecutionPinningStore()
    bound = bind_context_provider_set_from_registry(registry, engine_id="default")
    pin_context_provider_set_for_execution(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        bound_set=bound,
        pinning_store=pinning_store,
    )
    binding = next(iter(pinning_store._bindings.values()))
    registry.remove_provider("builtin.workspace")
    service = RuntimeInspectionService(context_provider_pinning_store=pinning_store)
    result = service.inspect_execution(
        tenant_id="tenant-a",
        execution_id=binding.execution_id,
        scope_application_id="app.test",
        scope_tenant_id="tenant-a",
    )
    provider_ids = {item.subject for item in result.explanations if item.subject.startswith("builtin.")}
    assert "builtin.workspace" in provider_ids


def test_tenant_isolation_for_pinning_store() -> None:
    store = InMemoryContextProviderExecutionPinningStore()
    execution_id = mint_execution_id()
    registry_a = ContextPluginRegistry()
    registry_a.add_provider(_RecordingProvider("tenant.a.provider", frozenset({ContextFragmentSource.CUSTOM})))
    bound_a = bind_context_provider_set_from_registry(registry_a, engine_id="default")
    pin_context_provider_set_for_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        bound_set=bound_a,
        pinning_store=store,
    )
    assert store.get(tenant_id="tenant-a", execution_id=execution_id) is not None
    assert store.get(tenant_id="tenant-b", execution_id=execution_id) is None


def test_lazy_eligibility_helpers() -> None:
    descriptor = build_provider_descriptor(
        "builtin.workspace",
        provider_version="1.0.0",
        supported_sources=frozenset({ContextFragmentSource.WORKSPACE}),
    )
    assert not is_provider_eligible(
        descriptor,
        _assembly_request(excluded_sources=frozenset({ContextFragmentSource.WORKSPACE})),
    )
    assert is_provider_eligible(descriptor, _assembly_request())
