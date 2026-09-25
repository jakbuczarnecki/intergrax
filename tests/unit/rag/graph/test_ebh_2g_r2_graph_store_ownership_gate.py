# © Artur Czarnecki. All rights reserved.

"""EBH-2G-R2-R1 — Integration-owned graph store selection and typed RAG adaptation gates."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.applications._shared.rag_runtime_bridge import resolve_rag_profile_for_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.providers.graph_store.neo4j.integration import Neo4jGraphStoreIntegration
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.bootstrap.rag_stack_bootstrap import create_default_rag_stack
from intergrax.rag.graph.providers.cypher_rag_graph_store import CypherRagGraphStore
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile
from tests.unit.rag.graph.fixtures.fake_cypher_graph_integration import FakeCypherGraphIntegration

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RAG_GRAPH_BOOTSTRAP = _REPO_ROOT / "intergrax" / "rag" / "graph" / "bootstrap"
_RAG_STACK_BOOTSTRAP = _REPO_ROOT / "intergrax" / "rag" / "bootstrap" / "rag_stack_bootstrap.py"
_RAG_RUNTIME_BRIDGE = _REPO_ROOT / "intergrax" / "applications" / "_shared" / "rag_runtime_bridge.py"
_CYPHER_ADAPTER = (
    _REPO_ROOT / "intergrax" / "rag" / "graph" / "providers" / "cypher_rag_graph_store.py"
)
_RAG_PROFILE = _REPO_ROOT / "intergrax" / "rag" / "profiles" / "rag_profile.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path))


@pytest.fixture(autouse=True)
def _register_integrations() -> None:
    register_default_integrations(override=True)


def test_rag_profile_has_no_graph_store_backend_field() -> None:
    names = {field.name for field in fields(RagProfile)}
    assert "graph_store_backend" not in names


def test_intergrax_rag_graph_store_not_in_production_rag_composition() -> None:
    for rel in (
        "rag/bootstrap/rag_stack_bootstrap.py",
        "rag/graph/bootstrap/graph_store_bootstrap.py",
        "rag/profiles/rag_profile.py",
        "applications/_shared/rag_runtime_bridge.py",
    ):
        source = _read(_REPO_ROOT / "intergrax" / rel)
        assert "INTERGRAX_RAG_GRAPH_STORE" not in source


def test_backend_registry_module_removed() -> None:
    assert not (_RAG_GRAPH_BOOTSTRAP / "backend_registry.py").is_file()


def test_rag_graph_bootstrap_has_no_provider_bundle_imports() -> None:
    tree = _parse(_RAG_GRAPH_BOOTSTRAP / "graph_store_bootstrap.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "integrations.providers.graph_store" not in node.module


def test_rag_stack_bootstrap_uses_resolve_from_profile() -> None:
    source = _read(_RAG_STACK_BOOTSTRAP)
    assert "resolve_from_profile" in source
    assert "IntegrationCategory.GRAPH_STORE" in source


def test_cypher_rag_graph_store_constructor_not_any() -> None:
    tree = _parse(_CYPHER_ADAPTER)
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CypherRagGraphStore"
    )
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    for arg in init.args.args:
        if arg.arg == "integration_store" and arg.annotation is not None:
            assert ast.unparse(arg.annotation) != "Any"
            return
    pytest.fail("CypherRagGraphStore.__init__ missing typed integration_store")


def test_rag_runtime_bridge_does_not_copy_graph_store_backend() -> None:
    source = _read(_RAG_RUNTIME_BRIDGE)
    assert "graph_store_backend" not in source


def test_case_a_neo4j_slug_uses_integration_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _spy(profile: IntegrationProfile, category: IntegrationCategory) -> FakeCypherGraphIntegration:
        calls.append(category.value)
        return FakeCypherGraphIntegration()

    with patch(
        "intergrax.rag.bootstrap.rag_stack_bootstrap.resolve_from_profile",
        side_effect=_spy,
    ):
        stack = create_default_rag_stack(
            integration_profile=IntegrationProfile(
                graph_store=IntegrationBinding.from_slug("neo4j"),
            ),
            profile=RagProfile(graph_rag_enabled=True),
            tenant_id="gate-a",
        )
    assert calls == [IntegrationCategory.GRAPH_STORE.value]
    assert isinstance(stack.graph_store, CypherRagGraphStore)


def test_case_b_resolution_error_no_inmemory_fallback() -> None:
    with patch(
        "intergrax.rag.bootstrap.rag_stack_bootstrap.resolve_from_profile",
        side_effect=RuntimeError("resolution_failed"),
    ):
        with pytest.raises(RuntimeError, match="resolution_failed"):
            create_default_rag_stack(
                integration_profile=IntegrationProfile(
                    graph_store=IntegrationBinding.from_slug("neo4j"),
                ),
                profile=RagProfile(graph_rag_enabled=True),
                tenant_id="gate-b",
            )


def test_case_c_prebuilt_integration_instance_adapted() -> None:
    integration = Neo4jGraphStoreIntegration.from_client(FakeCypherGraphIntegration())
    stack = create_default_rag_stack(
        integration_profile=IntegrationProfile(
            graph_store=IntegrationBinding.from_instance(integration),
        ),
        profile=RagProfile(graph_rag_enabled=True),
        tenant_id="gate-c",
    )
    assert isinstance(stack.graph_store, CypherRagGraphStore)


def test_case_d_harness_without_integration_uses_inmemory() -> None:
    stack = create_default_rag_stack(
        profile=RagProfile(graph_rag_enabled=True),
        tenant_id="gate-d",
    )
    assert isinstance(stack.graph_store, InMemoryGraphStore)


def test_case_e_injected_graph_store_skips_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    injected = InMemoryGraphStore(tenant_id="di")

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("resolve_from_profile must not run when graph_store is injected")

    with patch(
        "intergrax.rag.bootstrap.rag_stack_bootstrap.resolve_from_profile",
        side_effect=_fail,
    ):
        stack = create_default_rag_stack(
            integration_profile=IntegrationProfile(
                graph_store=IntegrationBinding.from_slug("neo4j"),
            ),
            profile=RagProfile(graph_rag_enabled=True),
            graph_store=injected,
            tenant_id="gate-e",
        )
    assert stack.graph_store is injected


def test_case_f_product_host_approved_slug_enables_graph_rag() -> None:
    env = ApplicationEnvironmentProfile.product_defaults().model_copy(
        update={"context_profile": ContextProfile(enable_rag=True)},
    )
    profile = resolve_rag_profile_for_environment(
        env,
        integration_profile=IntegrationProfile(
            graph_store=IntegrationBinding.from_slug("neo4j"),
        ),
    )
    assert profile is not None
    assert profile.graph_rag_enabled is True
    assert "graph_store_backend" not in _read(_RAG_PROFILE)


def test_case_g_product_host_without_graph_store_disables_graph_rag() -> None:
    env = ApplicationEnvironmentProfile.product_defaults().model_copy(
        update={"context_profile": ContextProfile(enable_rag=True)},
    )
    profile = resolve_rag_profile_for_environment(
        env,
        integration_profile=IntegrationProfile(),
    )
    assert profile is not None
    assert profile.graph_rag_enabled is False


def test_graph_store_bootstrap_configured_path_has_no_inmemory_branch() -> None:
    source = _read(_RAG_GRAPH_BOOTSTRAP / "graph_store_bootstrap.py")
    assert "integration_graph_store is None" in source
    assert "InMemoryGraphStore" in source
    assert "profile" not in source
    assert "RagProfile" not in source
