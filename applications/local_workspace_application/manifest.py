# © Artur Czarnecki. All rights reserved.

"""Declarative agent roster for local_workspace_application (Tier-3 product host)."""

from __future__ import annotations

from intergrax.applications._shared.budget_wiring import product_agent_budget_slice
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from local_indexer.local_indexer_agent import LocalIndexerAgent
from local_search.local_search_agent import LocalSearchAgent
from local_synthesizer.local_synthesizer_agent import LocalSynthesizerAgent
from local_workspace_application.host.agent_factories import (
    build_local_workspace_local_indexer_from_context,
    build_local_workspace_local_search_from_context,
    build_local_workspace_local_synthesizer_from_context,
)


def _local_workspace_environment() -> ApplicationEnvironmentProfile:
    return (
        ApplicationEnvironmentProfile.product_defaults(
            profile_id="local_workspace.product",
            skill_bundles=["harness"],
        )
        .model_copy(
            update={
                "integration_profile": IntegrationProfile.legal_product(),
                "context_profile": ApplicationEnvironmentProfile.product_defaults()
                .context_profile.model_copy(update={"enable_rag": True, "enable_websearch": False}),
            }
        )
        .with_harness_memory()
    )


LOCAL_WORKSPACE_APPLICATION_MANIFEST = ApplicationManifest.product(
    app_id="local_workspace",
    name="Intergrax Local Knowledge Workspace API",
    route_prefix="/v1/local_workspace",
    env_prefix="LOCAL_WORKSPACE_",
    default_port=8020,
    default_capability="local.workspace.search",
    integration_profile=IntegrationProfile.legal_product(),
    environment=_local_workspace_environment(),
    agents=[
        AgentBinding.mount(
            LocalIndexerAgent,
            factory=build_local_workspace_local_indexer_from_context,
            capabilities=["local.workspace.index"],
            budget_slice=product_agent_budget_slice(),
        ),
        AgentBinding.mount(
            LocalSearchAgent,
            factory=build_local_workspace_local_search_from_context,
            capabilities=["local.workspace.search"],
            default=True,
            budget_slice=product_agent_budget_slice(),
        ),
        AgentBinding.mount(
            LocalSynthesizerAgent,
            factory=build_local_workspace_local_synthesizer_from_context,
            capabilities=["local.workspace.synthesize"],
            budget_slice=product_agent_budget_slice(),
        ),
    ],
    description=(
        "Local Knowledge Workspace (LKW) — multi-agent host for indexing, "
        "semantic search, and synthesis over user-local documents."
    ),
)
