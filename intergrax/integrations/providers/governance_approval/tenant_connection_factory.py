# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.governance_approval.client import (
    HttpxGovernanceApprovalReadClient,
)
from intergrax.integrations.providers.governance_approval.config import (
    GovernanceApprovalIntegrationConfig,
)
from intergrax.integrations.providers.governance_approval.integration import (
    GovernanceApprovalIntegration,
)
from intergrax.integrations.providers.governance_approval.knowledge_read import (
    GOVERNANCE_APPROVAL_PROVIDER_ID,
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
) -> GovernanceApprovalIntegrationConfig:
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
    return GovernanceApprovalIntegrationConfig(
        base_url=base_url,
        timeout_seconds=float(timeout) if timeout is not None else 5.0,
    )


class GovernanceApprovalTenantConnectionIntegrationFactory(
    TenantConnectionIntegrationFactory,
):
    """Compose one Governance Approval integration from durable tenant connection config."""

    def __init__(
        self,
        *,
        http_client_factory: Callable[[GovernanceApprovalIntegrationConfig], Any] | None = None,
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
    ) -> GovernanceApprovalIntegration:
        _require_nonblank(tenant_id, field_name="tenant_id")
        _require_nonblank(connection_ref, field_name="connection_ref")
        _require_nonblank(credential_ref, field_name="credential_ref")
        if provider_id != GOVERNANCE_APPROVAL_PROVIDER_ID:
            raise ValueError("provider_id does not match governance_approval")
        if integration_kind is not IntegrationCategory.WORKFLOW_ORCHESTRATOR:
            raise ValueError("integration_kind does not match workflow_orchestrator")
        config = _parse_secret_free_config(secret_free_config)
        factory = self._http_client_factory or (
            lambda parsed: HttpxGovernanceApprovalReadClient(config=parsed)
        )
        client = factory(config)
        return GovernanceApprovalIntegration.from_client(client, config=config)


__all__ = ["GovernanceApprovalTenantConnectionIntegrationFactory"]
