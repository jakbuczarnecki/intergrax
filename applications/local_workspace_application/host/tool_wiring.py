# © Artur Czarnecki. All rights reserved.

"""Tool catalog wiring for local_workspace_application (Local Knowledge Workspace)."""

from __future__ import annotations

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry.profile import ToolProfile
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

_LKW_BASE_TOOL_IDS: tuple[str, ...] = (
    "document.parse",
    "workspace.read_file",
    "workspace.write_file",
    "workspace.list_files",
    "workspace.snapshot",
    "workspace.delete_file",
    "workspace.search",
    "memory.read",
    "memory.write",
    "memory.list_keys",
    "cache.get",
    "cache.set",
    "rag.list_collections",
)


def wire_local_workspace_tools(
    *,
    settings: LocalWorkspaceBackendSettings | None = None,
    integration_profile: IntegrationProfile | None = None,
) -> ApplicationToolWiring:
    settings = settings or LocalWorkspaceBackendSettings.from_env()
    enabled = list(_LKW_BASE_TOOL_IDS)
    enabled.extend(settings.enabled_tool_ids)
    profile = ToolProfile(enabled=enabled)
    return build_application_tool_wiring(
        profile,
        integration_profile=integration_profile or IntegrationProfile.legal_product(),
    )
