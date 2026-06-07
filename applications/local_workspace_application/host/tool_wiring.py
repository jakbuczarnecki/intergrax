# © Artur Czarnecki. All rights reserved.

"""Tool catalog wiring for local_workspace_application (Local Knowledge Workspace)."""

from __future__ import annotations

from dataclasses import replace

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.providers.filesystem.allowlist import read_allowlist_roots_from_env
from intergrax.tools.providers.document.service import DOCUMENT_PARSE_PREVIEW_TOOL_ID, DOCUMENT_PARSE_TOOL_ID
from intergrax.tools.providers.filesystem.service import (
    FILESYSTEM_GLOB_TOOL_ID,
    FILESYSTEM_LIST_TOOL_ID,
    FILESYSTEM_READ_TEXT_TOOL_ID,
    FILESYSTEM_STAT_TOOL_ID,
)
from intergrax.tools.providers.rag.index_lifecycle_service import (
    RAG_CHECK_INDEX_STATUS_TOOL_ID,
    RAG_GET_DOCUMENT_TOOL_ID,
    RAG_LIST_DOCUMENTS_TOOL_ID,
)
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

_LKW_BASE_TOOL_IDS: tuple[str, ...] = (
    DOCUMENT_PARSE_TOOL_ID,
    DOCUMENT_PARSE_PREVIEW_TOOL_ID,
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
    "cache.delete",
    "rag.list_collections",
    RAG_LIST_DOCUMENTS_TOOL_ID,
    RAG_GET_DOCUMENT_TOOL_ID,
    RAG_CHECK_INDEX_STATUS_TOOL_ID,
    "rag.rerank",
)

_FILESYSTEM_TOOL_IDS: tuple[str, ...] = (
    FILESYSTEM_LIST_TOOL_ID,
    FILESYSTEM_GLOB_TOOL_ID,
    FILESYSTEM_READ_TEXT_TOOL_ID,
    FILESYSTEM_STAT_TOOL_ID,
)


def wire_local_workspace_tools(
    *,
    settings: LocalWorkspaceBackendSettings | None = None,
    integration_profile: IntegrationProfile | None = None,
) -> ApplicationToolWiring:
    settings = settings or LocalWorkspaceBackendSettings.from_env()
    enabled = list(_LKW_BASE_TOOL_IDS)
    enabled.extend(settings.enabled_tool_ids)

    resolved_profile = integration_profile or IntegrationProfile.legal_product()
    allowed_roots = frozenset(settings.allowed_read_roots) if settings.allowed_read_roots else read_allowlist_roots_from_env()
    if allowed_roots:
        for tool_id in _FILESYSTEM_TOOL_IDS:
            if tool_id not in enabled:
                enabled.append(tool_id)

    ctx = ToolWiringContext.from_integration_profile(resolved_profile)
    if allowed_roots:
        ctx = replace(ctx, read_allowlist_roots=allowed_roots)

    profile = ToolProfile(enabled=enabled)
    return build_application_tool_wiring(
        profile,
        integration_profile=resolved_profile,
        wiring_context=ctx,
    )
