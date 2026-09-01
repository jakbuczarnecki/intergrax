# © Artur Czarnecki. All rights reserved.

"""Declarative agent roster for local_workspace_application (Tier-3 product host)."""

from __future__ import annotations

from intergrax.applications._shared.agent_certification_wiring import apply_roster_agent_governance
from intergrax.applications._shared.budget_wiring import product_agent_budget_slice
from intergrax.applications._shared.ownership_wiring import standard_product_operational_ownership
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from local_indexer.local_indexer_agent import LocalIndexerAgent
from local_search.local_search_agent import LocalSearchAgent
from local_synthesizer.local_synthesizer_agent import LocalSynthesizerAgent
from tool_selection_qualifier.tool_selection_qualifier_agent import ToolSelectionQualifierAgent
from local_workspace_application.host.agent_factories import (
    build_local_workspace_local_indexer_from_context,
    build_local_workspace_local_search_from_context,
    build_local_workspace_local_synthesizer_from_context,
    build_local_workspace_tool_selection_qualifier_from_context,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
    build_local_workspace_integration_profile,
)


_LOCAL_WORKSPACE_AGENTS = [
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
    AgentBinding.mount(
        ToolSelectionQualifierAgent,
        factory=build_local_workspace_tool_selection_qualifier_from_context,
        capabilities=["local.workspace.tool_selection_qualification"],
        budget_slice=product_agent_budget_slice(),
    ),
]


def _local_workspace_environment() -> ApplicationEnvironmentProfile:
    return apply_roster_agent_governance(
        build_local_workspace_environment_profile(),
        agents=_LOCAL_WORKSPACE_AGENTS,
        app_id="local_workspace",
    )


LOCAL_WORKSPACE_APPLICATION_MANIFEST = ApplicationManifest.product(
    app_id="local_workspace",
    name="Intergrax Local Knowledge Workspace API",
    route_prefix="/v1/local_workspace",
    env_prefix="LOCAL_WORKSPACE_",
    default_port=8020,
    default_capability="local.workspace.search",
    integration_profile=build_local_workspace_integration_profile(),
    environment=_local_workspace_environment(),
    agents=list(_LOCAL_WORKSPACE_AGENTS),
    description=(
        "Local Knowledge Workspace (LKW) — multi-agent host for indexing, "
        "semantic search, and synthesis over user-local documents."
    ),
    ownership=standard_product_operational_ownership("local_workspace"),
)
