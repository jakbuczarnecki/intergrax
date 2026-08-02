# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace tenant connection integration factory for restart rehydration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.google_workspace.config import (
    GoogleWorkspaceCollaborationSuiteIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceClientFactory,
    GoogleWorkspaceCredentialResolver,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.runtime.vendor_knowledge.models import JsonValue


class _RuntimeCredentialResolver:
    """Runtime-only credential resolver bound to one durable credential reference."""

    def __init__(
        self,
        *,
        credential_ref: str,
        credential_material: Mapping[str, str],
    ) -> None:
        self._credential_ref = credential_ref
        self._credential_material = MappingProxyType(dict(credential_material))

    def resolve_credential(self, credential_ref: str) -> Mapping[str, str]:
        if credential_ref != self._credential_ref:
            raise ValueError("credential reference does not match bound reference")
        return self._credential_material


def _require_nonblank(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _parse_credential_material(credential: str) -> dict[str, str]:
    try:
        parsed = json.loads(credential)
    except json.JSONDecodeError:
        raise ValueError("credential payload is not valid JSON") from None
    if not isinstance(parsed, dict):
        raise ValueError("credential payload must be a JSON object")
    if not parsed:
        raise ValueError("credential payload must not be empty")
    material: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("credential payload keys must be nonblank strings")
        if not isinstance(value, str):
            raise ValueError("credential payload values must be strings")
        material[key] = value
    return material


class GoogleWorkspaceTenantConnectionIntegrationFactory:
    """Compose Google Workspace integrations from durable tenant connection material."""

    def __init__(self, client_factory: GoogleWorkspaceClientFactory) -> None:
        self._client_factory = client_factory

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
    ) -> object:
        _require_nonblank(tenant_id, field_name="tenant_id")
        _require_nonblank(connection_ref, field_name="connection_ref")
        cleaned_credential_ref = _require_nonblank(credential_ref, field_name="credential_ref")
        if not isinstance(credential, str) or not credential.strip():
            raise ValueError("credential must be a nonblank string")
        if provider_id != GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID:
            raise ValueError("provider_id does not match google_workspace")
        if integration_kind is not IntegrationCategory.COLLABORATION_SUITE:
            raise ValueError("integration_kind does not match collaboration_suite")
        if secret_free_config:
            raise ValueError("secret_free_config must be empty for google_workspace")
        credential_material = _parse_credential_material(credential)
        resolver: GoogleWorkspaceCredentialResolver = _RuntimeCredentialResolver(
            credential_ref=cleaned_credential_ref,
            credential_material=credential_material,
        )
        return GoogleWorkspaceCollaborationSuiteIntegration.compose(
            config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
                enabled=True,
                credential_ref=cleaned_credential_ref,
            ),
            credential_resolver=resolver,
            client_factory=self._client_factory,
        )
