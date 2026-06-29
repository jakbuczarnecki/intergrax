# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level MS365 Graph openers — internal to the ms365_graph integration package.

Only this module may construct ``httpx.Client`` / ``GraphRestClient`` for Graph.
All composition roots use ``bundle.create_ms365_graph_*`` or
``profile.resolve(COLLABORATION_SUITE)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.adapter import _Ms365GraphCollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import Ms365GraphCollaborationSuiteIntegration
from intergrax.integrations.providers.collaboration_suite.ms365_graph.client import GraphRestClient
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_TIMEOUT_SECONDS,
    Ms365GraphIntegrationConfig,
)


def _fetch_access_token(config: Ms365GraphIntegrationConfig) -> str:
    import httpx

    timeout = float(config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    response = httpx.post(
        config.token_url,
        data={
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise IntegrationConfigurationError("Unexpected MS365 token response")
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise IntegrationConfigurationError("MS365 token response missing access_token")
    return token


def _create_http_client(config: Ms365GraphIntegrationConfig) -> Any:
    import httpx

    token = _fetch_access_token(config)
    timeout = float(config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    return httpx.Client(
        base_url=config.graph_base_url.rstrip("/"),
        timeout=timeout,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )


def open_graph_rest_client(
    config: Ms365GraphIntegrationConfig,
    *,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[Ms365GraphIntegrationConfig], Any]] = None,
    access_token: Optional[str] = None,
) -> GraphRestClient:
    if http_client is None:
        if access_token is not None:
            import httpx

            timeout = float(config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
            http_client = httpx.Client(
                base_url=config.graph_base_url.rstrip("/"),
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {access_token}",
                },
            )
        else:
            factory = http_client_factory or _create_http_client
            http_client = factory(config)
    return GraphRestClient(config, http_client=http_client)


def open_ms365_graph_collaboration_suite(
    config: Ms365GraphIntegrationConfig,
    *,
    implementation: Optional[CollaborationSuite] = None,
    client: Optional[GraphRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[Ms365GraphIntegrationConfig], Any]] = None,
    access_token: Optional[str] = None,
) -> Ms365GraphCollaborationSuiteIntegration:
    if implementation is not None:
        if isinstance(implementation, Ms365GraphCollaborationSuiteIntegration):
            return implementation
        return Ms365GraphCollaborationSuiteIntegration.from_client(implementation)
    rest_client = client or open_graph_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
        access_token=access_token,
    )
    return Ms365GraphCollaborationSuiteIntegration.from_client(_Ms365GraphCollaborationSuite(rest_client))