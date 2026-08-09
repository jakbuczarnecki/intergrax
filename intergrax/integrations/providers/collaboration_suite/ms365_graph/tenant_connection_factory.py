from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.bundle import (
    open_ms365_graph_collaboration_suite,
    resolve_ms365_graph_config,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.runtime.vendor_knowledge.models import JsonValue
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionIntegrationFactory,
)


Ms365GraphRuntimeBuilder = Callable[
    [object],
    Ms365GraphCollaborationSuiteIntegration,
]
_ALLOWED_SECRET_FREE_CONFIG_KEYS = frozenset(
    {"client_id", "default_user", "graph_base_url", "timeout_seconds"}
)


@dataclass(frozen=True, slots=True)
class Ms365GraphTenantConnectionIntegrationFactory(
    TenantConnectionIntegrationFactory
):
    """Build the existing Microsoft Graph runtime from a durable connection."""

    runtime_builder: Ms365GraphRuntimeBuilder | None = None

    def create_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        credential: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> Ms365GraphCollaborationSuiteIntegration:
        _require_nonblank(tenant_id, field_name="tenant_id")
        _require_nonblank(connection_ref, field_name="connection_ref")
        _require_nonblank(credential_ref, field_name="credential_ref")
        if provider_id != MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID:
            raise ValueError("provider_id does not match the Microsoft Graph factory")
        if integration_kind is not IntegrationCategory.COLLABORATION_SUITE:
            raise ValueError("integration_kind does not match the Microsoft Graph factory")
        if not isinstance(credential, str) or not credential.strip():
            raise ValueError("credential must be a non-empty string")

        config_overrides = _resolve_secret_free_config(secret_free_config)
        config_overrides.update(
            tenant_id=tenant_id,
            client_secret=credential,
        )
        try:
            config = resolve_ms365_graph_config(**config_overrides)
        except Exception:
            raise IntegrationConfigurationError(
                "Microsoft Graph runtime configuration failed (credentials redacted)",
            ) from None
        builder = self.runtime_builder or open_ms365_graph_collaboration_suite
        try:
            return builder(config)
        except IntegrationConfigurationError:
            raise
        except Exception:
            raise IntegrationConfigurationError(
                "Microsoft Graph runtime construction failed (credentials redacted)",
            ) from None


def _require_nonblank(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _resolve_secret_free_config(
    secret_free_config: Mapping[str, JsonValue],
) -> dict[str, object]:
    unexpected = set(secret_free_config) - _ALLOWED_SECRET_FREE_CONFIG_KEYS
    if unexpected:
        raise ValueError(
            "Microsoft Graph secret-free configuration contains unsupported fields"
        )

    client_id = secret_free_config.get("client_id")
    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError("Microsoft Graph secret-free configuration is missing client_id")

    default_user = secret_free_config.get("default_user", "")
    if not isinstance(default_user, str):
        raise ValueError("Microsoft Graph default_user must be a string")

    graph_base_url = secret_free_config.get("graph_base_url", DEFAULT_GRAPH_BASE_URL)
    if not isinstance(graph_base_url, str) or not graph_base_url.strip():
        raise ValueError("Microsoft Graph graph_base_url must be a non-empty string")

    timeout_seconds = secret_free_config.get(
        "timeout_seconds",
        DEFAULT_TIMEOUT_SECONDS,
    )
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("Microsoft Graph timeout_seconds must be a number")

    return {
        "client_id": client_id.strip(),
        "default_user": default_user.strip(),
        "graph_base_url": graph_base_url.strip(),
        "timeout_seconds": float(timeout_seconds),
    }


__all__ = [
    "Ms365GraphRuntimeBuilder",
    "Ms365GraphTenantConnectionIntegrationFactory",
]
