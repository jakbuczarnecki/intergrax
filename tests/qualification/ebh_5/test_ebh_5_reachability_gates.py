# © Artur Czarnecki. All rights reserved.

"""EBH-5 vendor selection reachability evidence gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.integrations._shared.config import read_integration_slug_from_env
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.tools.providers.observability.resolve import resolve_observability_backend
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_ebh5_reach_observability_fail_closed_without_role() -> None:
    ctx = ToolWiringContext(
        observability_backends={"custom_a": object(), "custom_b": object()},
    )
    with pytest.raises(RuntimeError, match="observability_role_backend_not_configured"):
        resolve_observability_backend(ctx, role="traces")


def test_ebh5_reach_observability_explicit_default_only() -> None:
    explicit = object()
    ctx = ToolWiringContext(observability_backend=explicit)
    assert resolve_observability_backend(ctx, role="default") is explicit


def test_ebh5_reach_integration_unknown_slug_fail_closed() -> None:
    profile = IntegrationProfile(relational_store="nonexistent_slug_ebh5_probe")
    with pytest.raises(Exception):
        resolve_from_profile(profile, IntegrationCategory.RELATIONAL_STORE)


def test_ebh5_reach_env_slug_is_canonical_merge_not_shadow_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_INTEGRATION_RELATIONAL_STORE", "sqlite")
    slug = read_integration_slug_from_env(IntegrationCategory.RELATIONAL_STORE)
    assert slug == "sqlite"


def test_ebh5_reach_factory_module_is_sanctioned_composition_owner() -> None:
    path = _REPO_ROOT / "intergrax/integrations/registry/factory.py"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "resolve_from_profile" in text
    assert "resolve_slug" in text


def test_ebh5_reach_no_second_global_provider_registry_in_rag_graph() -> None:
    rag_graph = _REPO_ROOT / "intergrax/rag/graph/integration_graph_store.py"
    if not rag_graph.is_file():
        pytest.skip("graph integration module absent")
    text = rag_graph.read_text(encoding="utf-8")
    assert "graph_store_backend" not in text
