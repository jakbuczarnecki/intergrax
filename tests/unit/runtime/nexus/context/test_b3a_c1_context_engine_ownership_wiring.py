# © Artur Czarnecki. All rights reserved.

"""MP-5F-B3A-C1: ContextEngine → UCL artifact ownership composition wiring."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextProviderContext,
)
from intergrax.context.registry import ContextPluginRegistry
from intergrax.context.session_history import (
    HandleSessionHistoryProvider,
    SESSION_HISTORY_CONTEXT_SCOPE_HANDLE,
    SESSION_HISTORY_REVISION_HANDLE,
    SESSION_HISTORY_SNAPSHOT_HANDLE,
    build_session_history_snapshot,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import UclArtifactOwnershipScope
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.ucl_artifact_ownership_composition import (
    resolve_ucl_artifact_ownership_scope,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from dataclasses import dataclass


@dataclass(slots=True)
class _RuntimeConfigStub:
    llm_adapter: LLMAdapter
    production_mode: bool = False
    context_budget_policy: ContextBudgetPolicy | None = None
    context_decision_profile: dict | None = None
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class _SmallWindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake-small"

    @property
    def context_window_tokens(self) -> int:
        return 512

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


def _provider_ctx(handles: dict[str, object]) -> ContextProviderContext:
    from intergrax.runtime.nexus.context.legacy_assembly_runtime_bridge import (
        build_context_assembly_runtime_from_legacy_handles,
    )

    runtime = build_context_assembly_runtime_from_legacy_handles(handles)
    if runtime is None:
        raise AssertionError("test handles must include runtime_config")
    return ContextProviderContext(engine_id="default", handles=handles, runtime=runtime)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_COMPOSITION_WIRING_PATHS = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "context_engine.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "context"
    / "ucl_artifact_ownership_composition.py",
)
_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.contracts.context_view",
    "intergrax.collaborative_work",
    "applications.",
    "agents.",
)
_FORBIDDEN_AST_NAMES = frozenset({"getattr", "setattr", "hasattr"})


def _assembly_request(
    *,
    workspace_id: str | None = "ws-A",
    tenant_id: str = "tenant1",
) -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="t1",
        run_id="r1",
        task_id="task1",
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        assembly_scope="acp_step",
        objective="test",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=200),
        assembly_options=TaskContextAssemblyOptions(),
    )


def test_context_engine_wiring_passes_artifact_ownership_kwarg() -> None:
    engine_source = _COMPOSITION_WIRING_PATHS[0].read_text(encoding="utf-8-sig")
    assert "resolve_ucl_artifact_ownership_scope" in engine_source
    assert "artifact_ownership=artifact_ownership" in engine_source


def test_resolve_ucl_artifact_ownership_scope_uses_request_contract_only() -> None:
    request = _assembly_request(workspace_id="ws-canonical", tenant_id="tenant-x")
    assert (
        resolve_ucl_artifact_ownership_scope(
            request,
            context_plan=SimpleNamespace(optimization_required=False),
        )
        is None
    )
    scope = resolve_ucl_artifact_ownership_scope(
        request,
        context_plan=SimpleNamespace(optimization_required=True),
    )
    assert scope == UclArtifactOwnershipScope(tenant_id="tenant-x", workspace_id="ws-canonical")


def test_workspace_independent_from_context_scope_in_ownership_scope() -> None:
    request = _assembly_request(workspace_id="ws-1")
    scope = resolve_ucl_artifact_ownership_scope(
        request,
        context_plan=SimpleNamespace(optimization_required=True),
    )
    assert scope is not None
    assert scope.workspace_id == "ws-1"
    assert scope.workspace_id != "ctx-77"


@pytest.mark.asyncio
async def test_engine_selection_only_without_workspace_succeeds() -> None:
    adapter = _SmallWindowAdapter()
    config = _RuntimeConfigStub(llm_adapter=adapter, production_mode=False)
    registry = ContextPluginRegistry()
    registry.add_provider(HandleSessionHistoryProvider())
    engine = DefaultNexusContextEngine(registry=registry)
    snapshot = build_session_history_snapshot(
        tenant_id="tenant1",
        context_scope_id="ctx-only",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="short", entry_id="m1")],
    )
    request = _assembly_request(workspace_id=None)
    provider_ctx = _provider_ctx({
        "runtime_config": config,
        "messages": [ChatMessage(role="user", content="current", entry_id="current")],
        SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot,
        SESSION_HISTORY_CONTEXT_SCOPE_HANDLE: snapshot.context_scope_id,
        SESSION_HISTORY_REVISION_HANDLE: snapshot.revision_id,
    })
    assembled = await engine.assemble(request, provider_ctx=provider_ctx)
    assert assembled.context_plan is not None
    assert assembled.context_plan.optimization_required is False


def test_composition_wiring_gate_forbidden_imports_and_reflection() -> None:
    violations: list[str] = []
    for path in _COMPOSITION_WIRING_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_AST_NAMES:
                violations.append(f"{path.name}:{node.lineno} uses {node.id}")
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                    if node.module == prefix or node.module.startswith(f"{prefix}."):
                        violations.append(f"{path.name}:{node.lineno} imports {node.module}")
    composition_source = _COMPOSITION_WIRING_PATHS[1].read_text(encoding="utf-8")
    if "context_scope_id" in composition_source:
        violations.append("composition must not reference context_scope_id for workspace")
    assert not violations, "\n".join(violations)
