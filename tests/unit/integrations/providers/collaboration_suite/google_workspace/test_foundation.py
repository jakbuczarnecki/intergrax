# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers import collaboration_suite
from intergrax.integrations.providers.collaboration_suite import google_workspace
from intergrax.integrations.providers.collaboration_suite.google_workspace.config import (
    GoogleWorkspaceCollaborationSuiteCompositionMode,
    GoogleWorkspaceCollaborationSuiteIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.bundle import (
    create_google_workspace_collaboration_suite_integration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS,
    GoogleWorkspaceClientFamily,
    GoogleWorkspaceSourceKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.runtime.integrations.contracts import PlatformIntegrationKind, PlatformIntegrationStatus


class _SpyCredentialResolver:
    calls: list[str]

    def __init__(self) -> None:
        self.calls = []

    def resolve_credential(self, credential_ref: str) -> Mapping[str, str]:
        self.calls.append(credential_ref)
        return {"kind": "opaque"}


class _SpyClientFactory:
    calls: list[Mapping[str, str]]

    def __init__(self) -> None:
        self.calls = []

    def create_client_family(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> GoogleWorkspaceClientFamily:
        self.calls.append(dict(credential_material))
        return object()


def test_provider_identity_and_category() -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.for_provider(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        display_name="Google Workspace",
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(enabled=False),
    )
    assert integration.provider_id == "google_workspace"
    assert integration.integration_kind == PlatformIntegrationKind.COLLABORATION_SUITE.value
    assert GoogleWorkspaceCollaborationSuiteIntegration.__name__ == (
        "GoogleWorkspaceCollaborationSuiteIntegration"
    )


def test_public_exports_exclude_product_specific_integrations() -> None:
    public_names = set(google_workspace.__all__)
    forbidden = {
        "GoogleDriveIntegration",
        "GoogleDocsIntegration",
        "GoogleSheetsIntegration",
        "GoogleSlidesIntegration",
        "GoogleCalendarIntegration",
        "GmailIntegration",
        "GoogleChatIntegration",
    }
    assert forbidden.isdisjoint(public_names)
    assert "GoogleWorkspaceCollaborationSuiteIntegration" in public_names


def test_supported_source_kinds_are_exact_and_unique() -> None:
    expected = ("drive", "docs", "sheets", "slides", "calendar", "mail", "chat")
    values = tuple(kind.value for kind in GoogleWorkspaceSourceKind)
    assert values == expected
    assert len(values) == len(set(values))
    assert GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS == tuple(GoogleWorkspaceSourceKind)
    integration = GoogleWorkspaceCollaborationSuiteIntegration.for_provider(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        display_name="Google Workspace",
    )
    assert tuple(kind.value for kind in integration.supported_source_kinds) == expected


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        GoogleWorkspaceCollaborationSuiteIntegrationConfig.model_validate(
            {"enabled": False, "access_token": "secret"},
        )


def test_enabled_config_requires_credential_ref() -> None:
    with pytest.raises(ValidationError):
        GoogleWorkspaceCollaborationSuiteIntegrationConfig(enabled=True, credential_ref="")


def test_config_serialization_is_secret_safe() -> None:
    config = GoogleWorkspaceCollaborationSuiteIntegrationConfig(
        enabled=True,
        credential_ref="tenant/google-workspace/main",
    )
    payload = config.model_dump()
    assert payload["credential_ref"] == "tenant/google-workspace/main"
    for forbidden in (
        "access_token",
        "refresh_token",
        "client_secret",
        "private_key",
        "credentials_json",
    ):
        assert forbidden not in payload
    public_view = config.public_view()
    assert public_view["credential_ref"] == "tenant/google-workspace/main"
    assert "access_token" not in public_view


def test_construction_does_not_invoke_dependency_ports() -> None:
    resolver = _SpyCredentialResolver()
    factory = _SpyClientFactory()
    GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=resolver,
        client_factory=factory,
    )
    assert resolver.calls == []
    assert factory.calls == []


def test_disabled_health_is_deterministic() -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.for_provider(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        display_name="Google Workspace",
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(enabled=False),
    )
    health = integration.check_health()
    assert health.status is PlatformIntegrationStatus.DISABLED
    assert health.message == "integration is disabled"
    integration.validate_runtime()


def test_enabled_runtime_validation_fails_closed_without_dependencies() -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.for_provider(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        display_name="Google Workspace",
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match="credential resolver"):
        integration.validate_runtime()

    health = integration.check_health()
    assert health.status is PlatformIntegrationStatus.UNAVAILABLE


def test_compose_retains_injected_shared_dependencies() -> None:
    resolver = _SpyCredentialResolver()
    factory = _SpyClientFactory()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=resolver,
        client_factory=factory,
    )
    assert integration.credential_resolver is resolver
    assert integration.client_factory is factory
    integration.validate_runtime()


class _MinimalCollaborationSuiteClient(CollaborationSuite):
    pass


def _assert_injected_client_compatibility(
    integration: GoogleWorkspaceCollaborationSuiteIntegration,
    *,
    client: CollaborationSuite,
) -> None:
    assert isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration)
    assert integration.config.enabled is True
    assert integration.config.composition_mode is GoogleWorkspaceCollaborationSuiteCompositionMode.INJECTED_CLIENT
    assert integration.client is client
    assert integration.credential_resolver is None
    assert integration.client_factory is None
    integration.validate_runtime()
    health = integration.check_health()
    assert health.status is not PlatformIntegrationStatus.UNAVAILABLE


def test_bundle_factory_injected_client_composition() -> None:
    client = _MinimalCollaborationSuiteClient()
    resolver = _SpyCredentialResolver()
    factory = _SpyClientFactory()
    integration = create_google_workspace_collaboration_suite_integration(
        client=client,
        enabled=True,
    )
    _assert_injected_client_compatibility(integration, client=client)
    assert resolver.calls == []
    assert factory.calls == []


def test_from_client_injected_client_composition() -> None:
    client = _MinimalCollaborationSuiteClient()
    resolver = _SpyCredentialResolver()
    factory = _SpyClientFactory()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        client,
        enabled=True,
    )
    _assert_injected_client_compatibility(integration, client=client)
    assert resolver.calls == []
    assert factory.calls == []


def test_enabled_credential_ref_mode_missing_factory_fails_closed() -> None:
    resolver = _SpyCredentialResolver()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.for_provider(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        display_name="Google Workspace",
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
    )
    integration._credential_resolver = resolver
    with pytest.raises(IntegrationConfigurationError, match="client factory"):
        integration.validate_runtime()


def test_enabled_uncomposed_credential_ref_mode_fails_closed() -> None:
    with pytest.raises(ValidationError):
        GoogleWorkspaceCollaborationSuiteIntegrationConfig(enabled=True, credential_ref="")


def test_enabled_uncomposed_injected_client_mode_fails_closed() -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.for_provider(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        display_name="Google Workspace",
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            composition_mode=GoogleWorkspaceCollaborationSuiteCompositionMode.INJECTED_CLIENT,
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match="injected client"):
        integration.validate_runtime()
    health = integration.check_health()
    assert health.status is PlatformIntegrationStatus.UNAVAILABLE


def test_lazy_package_imports_foundation_symbols() -> None:
    assert collaboration_suite.google_workspace.GoogleWorkspaceSourceKind is GoogleWorkspaceSourceKind
    assert (
        collaboration_suite.google_workspace.GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
        == "google_workspace"
    )
