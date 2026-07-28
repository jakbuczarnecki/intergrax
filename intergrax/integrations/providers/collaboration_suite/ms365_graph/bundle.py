# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete MS365 Graph integration bundle — the single composition root for Graph in Intergrax.

HTTP clients are opened only in ``opens.py``. Tier-3 code MUST use
``create_ms365_graph_collaboration_suite()``, ``create_ms365_graph_integration()``, or
``profile.resolve(IntegrationCategory.COLLABORATION_SUITE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.client import GraphRestClient
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import Ms365GraphIntegrationConfig
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteClient,
    Ms365GraphCollaborationSuiteIntegration,
    Ms365GraphCollaborationSuiteIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.opens import (
    open_graph_rest_client,
    open_ms365_graph_collaboration_suite,
)


@dataclass(frozen=True)
class Ms365GraphIntegrationBundle:
    config: Ms365GraphIntegrationConfig
    collaboration_suite: Ms365GraphCollaborationSuiteIntegration
    rest_client: GraphRestClient


def resolve_ms365_graph_config(**overrides: object) -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig.from_env(**overrides)


def create_ms365_graph_integration(
    *,
    collaboration_suite: Optional[CollaborationSuite] = None,
    client: Optional[GraphRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[Ms365GraphIntegrationConfig], Any]] = None,
    access_token: Optional[str] = None,
    download_http_client: Optional[Any] = None,
    download_http_client_factory: Optional[Callable[[Ms365GraphIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> Ms365GraphIntegrationBundle:
    config = resolve_ms365_graph_config(**config_overrides)
    if collaboration_suite is not None:
        suite = open_ms365_graph_collaboration_suite(
            config,
            implementation=collaboration_suite,
        )
        rest_client = suite._require_client().rest_client
    else:
        rest_client = client or open_graph_rest_client(
            config,
            http_client=http_client,
            http_client_factory=http_client_factory,
            access_token=access_token,
            download_http_client=download_http_client,
            download_http_client_factory=download_http_client_factory,
        )
        suite = open_ms365_graph_collaboration_suite(
            config,
            client=rest_client,
        )
    assert isinstance(suite, Ms365GraphCollaborationSuiteIntegration)
    return Ms365GraphIntegrationBundle(
        config=config,
        collaboration_suite=suite,
        rest_client=rest_client,
    )


def create_ms365_graph_collaboration_suite(
    *,
    collaboration_suite: Optional[CollaborationSuite] = None,
    client: Optional[GraphRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[Ms365GraphIntegrationConfig], Any]] = None,
    access_token: Optional[str] = None,
    download_http_client: Optional[Any] = None,
    download_http_client_factory: Optional[Callable[[Ms365GraphIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> Ms365GraphCollaborationSuiteIntegration:
    """Catalog factory for ``"ms365_graph"`` / ``COLLABORATION_SUITE``."""
    return create_ms365_graph_integration(
        collaboration_suite=collaboration_suite,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        access_token=access_token,
        download_http_client=download_http_client,
        download_http_client_factory=download_http_client_factory,
        **config_overrides,
    ).collaboration_suite


def create_ms365_graph_collaboration_suite_integration(
    *,
    client: Ms365GraphCollaborationSuiteClient | None = None,
    enabled: bool = False,
) -> Ms365GraphCollaborationSuiteIntegration:
    """
    Build a contract-based Ms365 Graph collaboration suite integration.

    Compatibility shim — constructs Integration via from_store (create_ms365_graph_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Ms365 Graph collaboration suite integration requires an injected client when enabled=True",
        )
    if client is not None:
        return Ms365GraphCollaborationSuiteIntegration.from_client(client, enabled=enabled)
    return Ms365GraphCollaborationSuiteIntegration.for_provider(
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        display_name="Ms365 Graph",
        config=Ms365GraphCollaborationSuiteIntegrationConfig(enabled=enabled),
    )
