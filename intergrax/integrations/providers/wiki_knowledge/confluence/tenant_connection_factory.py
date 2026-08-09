# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluence tenant-connection factory for restart-safe Vendor Knowledge rehydration."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.wiki_knowledge.confluence.bundle import (
    create_confluence_integration,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.config import (
    ConfluenceIntegrationConfig,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
    ConfluenceWikiKnowledgeIntegration,
)
from intergrax.runtime.vendor_knowledge.models import JsonValue
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionIntegrationFactory,
)

_ALLOWED_SECRET_FREE_CONFIG_KEYS = frozenset({"base_url", "timeout_seconds"})


def _require_nonblank(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _parse_credential(credential: object) -> tuple[str, str]:
    if not isinstance(credential, str) or not credential.strip():
        raise ValueError("credential must be a non-empty string")
    try:
        payload = json.loads(credential)
    except json.JSONDecodeError:
        raise ValueError("credential payload is not valid JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("credential payload must be a JSON object")
    if set(payload) != {"email", "api_token"}:
        raise ValueError("credential payload must contain email and api_token")
    email = _require_nonblank(payload["email"], field_name="credential email")
    api_token = _require_nonblank(payload["api_token"], field_name="credential api_token")
    return email, api_token


def _parse_secret_free_config(
    value: Mapping[str, JsonValue],
) -> tuple[str, float | int | None]:
    if not isinstance(value, Mapping):
        raise ValueError("secret_free_config must be a mapping")
    unknown = set(value) - _ALLOWED_SECRET_FREE_CONFIG_KEYS
    if unknown:
        raise ValueError("secret_free_config contains unsupported keys")
    base_url = _require_nonblank(value.get("base_url"), field_name="base_url")
    timeout = value.get("timeout_seconds")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float))
    ):
        raise ValueError("timeout_seconds must be a number")
    return base_url, timeout


class ConfluenceTenantConnectionIntegrationFactory(TenantConnectionIntegrationFactory):
    """Compose one Confluence integration from a durable provider connection."""

    def __init__(
        self,
        *,
        http_client_factory: Callable[[ConfluenceIntegrationConfig], Any] | None = None,
    ) -> None:
        self._http_client_factory = http_client_factory

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
    ) -> ConfluenceWikiKnowledgeIntegration:
        _require_nonblank(tenant_id, field_name="tenant_id")
        _require_nonblank(connection_ref, field_name="connection_ref")
        _require_nonblank(credential_ref, field_name="credential_ref")
        if provider_id != CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID:
            raise ValueError("provider_id does not match confluence")
        if integration_kind is not IntegrationCategory.WIKI_KNOWLEDGE:
            raise ValueError("integration_kind does not match wiki_knowledge")

        email, api_token = _parse_credential(credential)
        base_url, timeout_seconds = _parse_secret_free_config(secret_free_config)
        overrides: dict[str, object] = {
            "base_url": base_url,
            "email": email,
            "api_token": api_token,
        }
        if timeout_seconds is not None:
            overrides["timeout_seconds"] = timeout_seconds
        return create_confluence_integration(
            http_client_factory=self._http_client_factory,
            **overrides,
        ).wiki_knowledge


__all__ = ["ConfluenceTenantConnectionIntegrationFactory"]
