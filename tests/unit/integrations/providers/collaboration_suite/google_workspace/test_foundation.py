# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
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


@dataclass(frozen=True, slots=True)
class _FakeTransport:
    pass


@dataclass(frozen=True, slots=True)
class _FakeClientFamily:
    _transport: _FakeTransport

    @property
    def transport(self) -> _FakeTransport:
        return self._transport


def _make_family() -> _FakeClientFamily:
    return _FakeClientFamily(_transport=_FakeTransport())


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
        return _make_family()


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


def test_health_does_not_materialize_client_family() -> None:
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
    health = integration.check_health()
    assert health.status is not PlatformIntegrationStatus.UNAVAILABLE
    assert resolver.calls == []
    assert factory.calls == []


def test_require_client_family_resolves_and_creates_once() -> None:
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
    first = integration.require_client_family()
    second = integration.require_client_family()
    assert resolver.calls == ["refs/google-workspace"]
    assert factory.calls == [{"kind": "opaque"}]
    assert first is second


def test_require_client_family_concurrent_first_calls_share_one_family() -> None:
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

    def _require() -> GoogleWorkspaceClientFamily:
        return integration.require_client_family()

    with ThreadPoolExecutor(max_workers=8) as pool:
        families = list(pool.map(lambda _: _require(), range(8)))

    assert resolver.calls == ["refs/google-workspace"]
    assert len(factory.calls) == 1
    assert len({id(family) for family in families}) == 1


class _FailingCredentialResolver:
    def __init__(self) -> None:
        self.calls = 0

    def resolve_credential(self, credential_ref: str) -> Mapping[str, str]:
        self.calls += 1
        raise RuntimeError("secret-token-value must not leak")


class _FailingClientFactory:
    def __init__(self) -> None:
        self.calls = 0

    def create_client_family(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> GoogleWorkspaceClientFamily:
        self.calls += 1
        raise RuntimeError(f"boom {credential_material['kind']}")


def test_resolver_failure_is_not_cached() -> None:
    resolver = _FailingCredentialResolver()
    factory = _SpyClientFactory()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=resolver,
        client_factory=factory,
    )
    with pytest.raises(IntegrationConfigurationError, match="could not resolve") as exc_info:
        integration.require_client_family()
    assert "secret-token-value" not in str(exc_info.value)
    with pytest.raises(IntegrationConfigurationError):
        integration.require_client_family()
    assert resolver.calls == 2
    assert factory.calls == []


def test_factory_failure_is_not_cached() -> None:
    resolver = _SpyCredentialResolver()
    factory = _FailingClientFactory()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=resolver,
        client_factory=factory,
    )
    with pytest.raises(IntegrationDependencyError, match="could not create") as exc_info:
        integration.require_client_family()
    assert "opaque" not in str(exc_info.value)
    assert "boom" not in str(exc_info.value)
    with pytest.raises(IntegrationDependencyError):
        integration.require_client_family()
    assert factory.calls == 2


class _InvalidFamilyFactory:
    def create_client_family(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> GoogleWorkspaceClientFamily:
        return object()  # type: ignore[return-value]


def test_invalid_factory_result_fails_closed() -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=_SpyCredentialResolver(),
        client_factory=_InvalidFamilyFactory(),
    )
    with pytest.raises(IntegrationConfigurationError, match="invalid client family"):
        integration.require_client_family()


def test_injected_client_family_returned_directly() -> None:
    family = _make_family()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        family,  # type: ignore[arg-type]
        enabled=True,
    )
    assert integration.require_client_family() is family


def test_legacy_injected_client_remains_healthy_but_fails_on_require_family() -> None:
    client = _MinimalCollaborationSuiteClient()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        client,
        enabled=True,
    )
    integration.validate_runtime()
    health = integration.check_health()
    assert health.status is not PlatformIntegrationStatus.UNAVAILABLE
    with pytest.raises(IntegrationConfigurationError, match="does not expose"):
        integration.require_client_family()
