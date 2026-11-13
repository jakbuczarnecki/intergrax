# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Supported public imports for Tier-3 MCP server authors (MCP-P6-01).

Ordinary application facades should import only from this module and
``intergrax.contracts.mcp_server_definition`` — not catalog bridge,
invocation bridge, or other engine internals.
"""

from __future__ import annotations

from intergrax.applications._shared.mcp_application_server import (
    McpIdentityMode,
    compose_application_mcp_server,
    execute_mcp_agent_task_with_resolved_identity,
    research_pipeline_binding,
)
from intergrax.applications._shared.mcp_host_mount import (
    compose_mcp_fastmcp_for_definition,
    couple_mcp_fastmcp_on_harness_host,
    normalize_mcp_mount_path,
    resolve_mcp_mount_path_for_host,
)
from intergrax.applications._shared.mcp_nexus_server import (
    build_nexus_mcp_server,
    execute_mcp_agent_task,
)
from intergrax.applications._shared.mcp_server_engine import (
    McpServerCompositionDeps,
    compose_mcp_server_from_definition,
    nexus_legacy_server_definition,
)
from intergrax.contracts.mcp_server_definition import (
    MCP_SERVER_DEFINITION_CONTRACT_VERSION,
    McpBindingVisibility,
    McpCapabilityBinding,
    McpCatalogToolBinding,
    McpExposurePolicy,
    McpHostBuiltinBinding,
    McpHostBuiltinId,
    McpHostCustomBinding,
    McpServerDefinition,
    McpServerDefinitionErrorCode,
    McpServerDefinitionValidationError,
    build_mcp_server_definition,
    compute_mcp_exposure_schema_fingerprint,
    resolve_effective_bindings,
    validate_mcp_server_definition,
)

__all__ = [
    "MCP_SERVER_DEFINITION_CONTRACT_VERSION",
    "McpBindingVisibility",
    "McpCapabilityBinding",
    "McpCatalogToolBinding",
    "McpExposurePolicy",
    "McpHostBuiltinBinding",
    "McpHostBuiltinId",
    "McpHostCustomBinding",
    "McpIdentityMode",
    "McpServerCompositionDeps",
    "McpServerDefinition",
    "McpServerDefinitionErrorCode",
    "McpServerDefinitionValidationError",
    "build_mcp_server_definition",
    "build_nexus_mcp_server",
    "compose_application_mcp_server",
    "compose_mcp_fastmcp_for_definition",
    "compose_mcp_server_from_definition",
    "compute_mcp_exposure_schema_fingerprint",
    "couple_mcp_fastmcp_on_harness_host",
    "execute_mcp_agent_task",
    "execute_mcp_agent_task_with_resolved_identity",
    "normalize_mcp_mount_path",
    "nexus_legacy_server_definition",
    "research_pipeline_binding",
    "resolve_effective_bindings",
    "resolve_mcp_mount_path_for_host",
    "validate_mcp_server_definition",
]
