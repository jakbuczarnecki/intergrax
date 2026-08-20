# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.change_approval.client import HttpxChangeApprovalReadClient
from intergrax.integrations.providers.change_approval.config import ChangeApprovalIntegrationConfig
from intergrax.integrations.providers.change_approval.integration import ChangeApprovalIntegration
from intergrax.integrations.providers.change_approval.knowledge_read import (
    CHANGE_APPROVAL_PROVIDER_ID,
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


def _parse_secret_free_config(
    value: Mapping[str, JsonValue],
) -> ChangeApprovalIntegrationConfig:
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
    return ChangeApprovalIntegrationConfig(
        base_url=base_url,
        timeout_seconds=float(timeout) if timeout is not None else 5.0,
    )


class ChangeApprovalTenantConnectionIntegrationFactory(TenantConnectionIntegrationFactory):
    """Compose one Change Approval integration from durable tenant connection config."""

    def __init__(
        self,
        *,
        http_client_factory: Callable[[ChangeApprovalIntegrationConfig], Any] | None = None,
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
    ) -> ChangeApprovalIntegration:
        _require_nonblank(tenant_id, field_name="tenant_id")
        _require_nonblank(connection_ref, field_name="connection_ref")
        _require_nonblank(credential_ref, field_name="credential_ref")
        if provider_id != CHANGE_APPROVAL_PROVIDER_ID:
            raise ValueError("provider_id does not match change_approval")
        if integration_kind is not IntegrationCategory.ISSUE_TRACKER:
            raise ValueError("integration_kind does not match issue_tracker")
        config = _parse_secret_free_config(secret_free_config)
        factory = self._http_client_factory or (
            lambda parsed: HttpxChangeApprovalReadClient(config=parsed)
        )
        client = factory(config)
        return ChangeApprovalIntegration.from_client(client, config=config)


__all__ = ["ChangeApprovalTenantConnectionIntegrationFactory"]
