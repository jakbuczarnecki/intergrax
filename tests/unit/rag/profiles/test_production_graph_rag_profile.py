# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.rag_runtime_bridge import resolve_rag_profile_for_environment
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
)
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.profiles.rag_profile import (
    APPROVED_PRODUCTION_GRAPH_STORE_SLUGS,
    HARNESS_GRAPH_STORE_BACKEND,
    PRODUCTION_GRAPH_STORE_BACKEND,
    is_harness_graph_rag_profile,
    production_graph_rag_profile,
    production_rag_profile,
    validate_graph_rag_production_wiring,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


@pytest.fixture(autouse=True)
def _register_integrations() -> None:
    register_default_integrations(override=True)


def test_production_rag_profile_is_harness_inmemory_graph() -> None:
    profile = production_rag_profile()
    assert is_harness_graph_rag_profile(profile) is True
    assert profile.graph_store_backend == HARNESS_GRAPH_STORE_BACKEND


def test_production_graph_rag_profile_requires_neo4j_backend() -> None:
    profile = production_graph_rag_profile()
    assert profile.graph_rag_enabled is True
    assert profile.graph_store_backend == PRODUCTION_GRAPH_STORE_BACKEND
    assert is_harness_graph_rag_profile(profile) is False


def test_validate_graph_rag_production_wiring_accepts_neo4j() -> None:
    profile = production_graph_rag_profile()
    assert validate_graph_rag_production_wiring(profile, graph_store_slug="neo4j") is None


def test_validate_graph_rag_production_wiring_accepts_memgraph() -> None:
    profile = production_graph_rag_profile()
    assert "memgraph" in APPROVED_PRODUCTION_GRAPH_STORE_SLUGS
    assert validate_graph_rag_production_wiring(profile, graph_store_slug="memgraph") is None


def test_validate_graph_rag_production_wiring_accepts_falkordb() -> None:
    profile = production_graph_rag_profile()
    assert "falkordb" in APPROVED_PRODUCTION_GRAPH_STORE_SLUGS
    assert validate_graph_rag_production_wiring(profile, graph_store_slug="falkordb") is None


def test_validate_graph_rag_production_wiring_rejects_inmemory_backend() -> None:
    profile = production_rag_profile()
    reason = validate_graph_rag_production_wiring(profile, graph_store_slug="neo4j")
    assert reason == "graph_store_backend_not_approved_for_production"


def test_product_environment_resolves_neo4j_graph_rag_profile() -> None:
    env = ApplicationEnvironmentProfile.product_defaults().model_copy(
        update={"context_profile": ContextProfile(enable_rag=True)},
    )
    assert env.application_profile is ApplicationProfile.PRODUCT

    profile = resolve_rag_profile_for_environment(
        env,
        integration_profile=IntegrationProfile(
            graph_store=IntegrationBinding.from_slug("neo4j"),
        ),
    )
    assert profile is not None
    assert profile.graph_store_backend == PRODUCTION_GRAPH_STORE_BACKEND


def test_validate_graph_rag_production_wiring_rejects_unapproved_integration_slug() -> None:
    profile = production_graph_rag_profile()
    reason = validate_graph_rag_production_wiring(profile, graph_store_slug="tigergraph")
    assert reason is not None
    assert reason.startswith("integration_graph_store_not_approved:")


def test_product_environment_without_graph_store_keeps_vector_only_rag() -> None:
    env = ApplicationEnvironmentProfile.product_defaults().model_copy(
        update={"context_profile": ContextProfile(enable_rag=True)},
    )
    profile = resolve_rag_profile_for_environment(
        env,
        integration_profile=IntegrationProfile(),
    )
    assert profile is not None
    assert profile.graph_rag_enabled is False


def test_product_environment_resolves_falkordb_graph_rag_profile() -> None:
    env = ApplicationEnvironmentProfile.product_defaults().model_copy(
        update={"context_profile": ContextProfile(enable_rag=True)},
    )
    profile = resolve_rag_profile_for_environment(
        env,
        integration_profile=IntegrationProfile(
            graph_store=IntegrationBinding.from_slug("falkordb"),
        ),
    )
    assert profile is not None
    assert profile.graph_store_backend == PRODUCTION_GRAPH_STORE_BACKEND
