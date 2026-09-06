# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace tenant connection integration factory for restart rehydration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.credential import (
    CredentialRef,
    CredentialResolutionContext,
    CredentialResolutionMode,
)
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.integrations.credentials.google_workspace import (
    GoogleWorkspaceSecretsStoreCredentialResolver,
)
from intergrax.integrations.credentials.secrets_store_resolver import (
    SecretsStoreCredentialResolver,
)
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
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionIntegrationFactory,
)


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


class GoogleWorkspaceTenantConnectionIntegrationFactory(TenantConnectionIntegrationFactory):
    """Compose Google Workspace integrations from durable tenant connection material."""

    def __init__(
        self,
        client_factory: GoogleWorkspaceClientFactory,
        *,
        secrets_store: SecretsStore | None = None,
    ) -> None:
        if not isinstance(client_factory, GoogleWorkspaceClientFactory):
            raise TypeError("client_factory must implement GoogleWorkspaceClientFactory")
        self._client_factory = client_factory
        self._secrets_store = secrets_store
        self._credential_resolution_mode = (
            CredentialResolutionMode.LATE_BOUND
            if secrets_store is not None
            else CredentialResolutionMode.RESOLVED_MATERIAL
        )

    @property
    def credential_resolution_mode(self) -> CredentialResolutionMode:
        return self._credential_resolution_mode

    def credential_resolution_mode_for(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> CredentialResolutionMode:
        return self.credential_resolution_mode

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
        if self.credential_resolution_mode is CredentialResolutionMode.LATE_BOUND:
            return self.create_late_bound_integration(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                provider_id=provider_id,
                integration_kind=integration_kind,
                credential_ref=credential_ref,
                secret_free_config=secret_free_config,
            )
        return self.create_integration_with_resolved_credential(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=provider_id,
            integration_kind=integration_kind,
            credential_ref=credential_ref,
            resolved_credential=credential,
            secret_free_config=secret_free_config,
        )

    def create_integration_with_resolved_credential(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        resolved_credential: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> object:
        if self.credential_resolution_mode is not CredentialResolutionMode.RESOLVED_MATERIAL:
            raise ValueError(
                "factory does not support resolved-material credential resolution",
            )
        _require_nonblank(tenant_id, field_name="tenant_id")
        _require_nonblank(connection_ref, field_name="connection_ref")
        cleaned_credential_ref = _require_nonblank(
            credential_ref,
            field_name="credential_ref",
        )
        if not isinstance(resolved_credential, str) or not resolved_credential.strip():
            raise ValueError("resolved_credential must be a nonblank string")
        if provider_id != GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID:
            raise ValueError("provider_id does not match google_workspace")
        if integration_kind is not IntegrationCategory.COLLABORATION_SUITE:
            raise ValueError("integration_kind does not match collaboration_suite")
        if not isinstance(secret_free_config, Mapping):
            raise ValueError("secret_free_config must be a mapping")
        if secret_free_config:
            raise ValueError("secret_free_config must be empty for google_workspace")
        credential_material = _parse_credential_material(resolved_credential)
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

    def create_late_bound_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> GoogleWorkspaceCollaborationSuiteIntegration:
        if self.credential_resolution_mode is not CredentialResolutionMode.LATE_BOUND:
            raise ValueError(
                "factory does not support late-bound credential resolution",
            )
        if provider_id != GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID:
            raise ValueError("provider_id does not match google_workspace")
        if integration_kind is not IntegrationCategory.COLLABORATION_SUITE:
            raise ValueError("integration_kind does not match collaboration_suite")
        if not isinstance(secret_free_config, Mapping):
            raise ValueError("secret_free_config must be a mapping")
        if secret_free_config:
            raise ValueError("secret_free_config must be empty for google_workspace")
        secrets_store = self._secrets_store
        if secrets_store is None:
            raise ValueError("secrets_store is required for late credential resolution")
        ref = CredentialRef.from_secret_path(
            provider_id=provider_id,
            secret_path=credential_ref,
            tenant_id=tenant_id,
        )
        resolver = GoogleWorkspaceSecretsStoreCredentialResolver(
            resolver=SecretsStoreCredentialResolver(secrets_store),
            credential_ref=ref,
            context=CredentialResolutionContext(tenant_id=tenant_id),
        )
        return GoogleWorkspaceCollaborationSuiteIntegration.compose(
            config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
                enabled=True,
                credential_ref=credential_ref,
            ),
            credential_resolver=resolver,
            client_factory=self._client_factory,
        )
