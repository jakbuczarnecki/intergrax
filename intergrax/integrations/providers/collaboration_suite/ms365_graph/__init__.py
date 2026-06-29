# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft 365 Graph collaboration suite integration (Phase M.6)."""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    ENV_MS365_CLIENT_ID,
    ENV_MS365_CLIENT_SECRET,
    ENV_MS365_DEFAULT_USER,
    ENV_MS365_TENANT_ID,
    Ms365GraphIntegrationConfig,
)

__all__ = [
    "ENV_MS365_CLIENT_ID",
    "ENV_MS365_CLIENT_SECRET",
    "ENV_MS365_DEFAULT_USER",
    "ENV_MS365_TENANT_ID",
    "Ms365GraphCollaborationSuite",
    "Ms365GraphIntegrationBundle",
    "Ms365GraphIntegrationConfig",
    "create_ms365_graph_collaboration_suite",
    "create_ms365_graph_integration",
    "register_ms365_graph_integration",
    "resolve_ms365_graph_config",
    "create_ms365_graph_collaboration_suite_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "Ms365GraphIntegrationBundle",
        "Ms365GraphCollaborationSuite",
        "create_ms365_graph_integration",
        "create_ms365_graph_collaboration_suite",
        "register_ms365_graph_integration",
        "resolve_ms365_graph_config",
        "create_ms365_graph_collaboration_suite_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID",
        "Ms365GraphCollaborationSuiteIntegration",
        "Ms365GraphCollaborationSuiteIntegrationConfig",
        "Ms365GraphCollaborationSuiteClient",
    }
)

def __getattr__(name: str):
    if name == "register_ms365_graph_integration":
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.register import register_ms365_graph_integration

        return register_ms365_graph_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.collaboration_suite.ms365_graph import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "Ms365GraphCollaborationSuite":
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.adapter import _Ms365GraphCollaborationSuite

        return Ms365GraphCollaborationSuite
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.collaboration_suite.ms365_graph import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
