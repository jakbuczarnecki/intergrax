# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import hashlib
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
    GoogleWorkspaceBinaryPayload,
    GoogleWorkspaceClientFamily,
    GoogleWorkspaceSourceKind,
    copy_google_workspace_credential_material,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.runtime.integrations.contracts import PlatformIntegrationKind, PlatformIntegrationStatus


@dataclass(frozen=True, slots=True)
class _FakeTransport:
    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        return {}


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
    with pytest.raises(IntegrationConfigurationError, match="credential resolution failed") as exc_info:
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
    with pytest.raises(IntegrationDependencyError, match="could not be created") as exc_info:
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
    with pytest.raises(IntegrationConfigurationError, match="returned an invalid client family"):
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
    with pytest.raises(IntegrationConfigurationError, match="valid client family"):
        integration.require_client_family()


_SECRET_FRAGMENT = "private_key=super-secret-value"


def test_disabled_credential_ref_rejects_require_client_family() -> None:
    resolver = _SpyCredentialResolver()
    factory = _SpyClientFactory()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=False,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=resolver,
        client_factory=factory,
    )
    with pytest.raises(IntegrationConfigurationError, match="disabled"):
        integration.require_client_family()


def test_disabled_credential_ref_makes_zero_resolver_factory_calls() -> None:
    resolver = _SpyCredentialResolver()
    factory = _SpyClientFactory()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=False,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=resolver,
        client_factory=factory,
    )
    with pytest.raises(IntegrationConfigurationError):
        integration.require_client_family()
    assert resolver.calls == []
    assert factory.calls == []


def test_disabled_injected_client_does_not_return_injected_family() -> None:
    family = _make_family()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        family,  # type: ignore[arg-type]
        enabled=False,
    )
    with pytest.raises(IntegrationConfigurationError, match="disabled"):
        integration.require_client_family()


class _InvalidTransportFamily:
    @property
    def transport(self) -> object:
        return object()


class _RaisingTransportFamily:
    @property
    def transport(self) -> _FakeTransport:
        raise RuntimeError("transport property secret failure")


def test_invalid_transport_family_rejected() -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        _InvalidTransportFamily(),  # type: ignore[arg-type]
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match="valid client family"):
        integration.require_client_family()


def test_raising_transport_property_rejected_safely() -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        _RaisingTransportFamily(),  # type: ignore[arg-type]
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match="valid client family") as exc_info:
        integration.require_client_family()
    assert "transport property secret failure" not in str(exc_info.value)


class _InvalidExecutorFactory:
    def create_request_executor(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> object:
        return object()


def test_invalid_executor_from_factory_rejected() -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.client_family import (
        DefaultGoogleWorkspaceClientFactory,
    )
    from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
        GoogleWorkspaceRetryPolicy,
    )

    factory = DefaultGoogleWorkspaceClientFactory(
        executor_factory=_InvalidExecutorFactory(),
        retry_policy=GoogleWorkspaceRetryPolicy(),
    )
    with pytest.raises(IntegrationDependencyError, match="invalid"):
        factory.create_client_family(credential_material={"kind": "opaque"})


class _SecretRaisingResolver:
    def resolve_credential(self, credential_ref: str) -> Mapping[str, str]:
        raise IntegrationConfigurationError(_SECRET_FRAGMENT)


class _SecretDependencyResolver:
    def resolve_credential(self, credential_ref: str) -> Mapping[str, str]:
        raise IntegrationDependencyError(f"refresh_token={_SECRET_FRAGMENT}")


class _SecretValueResolver:
    def resolve_credential(self, credential_ref: str) -> Mapping[str, str]:
        raise ValueError(f"client_secret={_SECRET_FRAGMENT}")


@pytest.mark.parametrize(
    "resolver",
    [
        _SecretRaisingResolver(),
        _SecretDependencyResolver(),
        _SecretValueResolver(),
    ],
)
def test_resolver_exceptions_are_sanitized(resolver: object) -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=resolver,  # type: ignore[arg-type]
        client_factory=_SpyClientFactory(),
    )
    with pytest.raises(IntegrationConfigurationError, match="credential resolution failed") as exc_info:
        integration.require_client_family()
    assert _SECRET_FRAGMENT not in str(exc_info.value)
    assert "private_key" not in str(exc_info.value)
    assert "refresh_token" not in str(exc_info.value)
    assert "client_secret" not in str(exc_info.value)


class _SecretConfigFactory:
    def create_client_family(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> GoogleWorkspaceClientFamily:
        raise IntegrationConfigurationError(f"access_token={_SECRET_FRAGMENT}")


class _SecretDependencyFactory:
    def create_client_family(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> GoogleWorkspaceClientFamily:
        raise IntegrationDependencyError(f"private_key={_SECRET_FRAGMENT}")


@pytest.mark.parametrize(
    "factory",
    [
        _SecretConfigFactory(),
        _SecretDependencyFactory(),
    ],
)
def test_client_factory_exceptions_are_sanitized(factory: object) -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=_SpyCredentialResolver(),
        client_factory=factory,  # type: ignore[arg-type]
    )
    with pytest.raises(IntegrationDependencyError, match="could not be created") as exc_info:
        integration.require_client_family()
    assert _SECRET_FRAGMENT not in str(exc_info.value)
    assert "access_token" not in str(exc_info.value)
    assert "private_key" not in str(exc_info.value)


class _SecretExecutorFactory:
    def create_request_executor(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> object:
        raise ValueError(f"credential payload {_SECRET_FRAGMENT}")


def test_executor_factory_exceptions_are_sanitized() -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.client_family import (
        DefaultGoogleWorkspaceClientFactory,
    )
    from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
        GoogleWorkspaceRetryPolicy,
    )

    factory = DefaultGoogleWorkspaceClientFactory(
        executor_factory=_SecretExecutorFactory(),
        retry_policy=GoogleWorkspaceRetryPolicy(),
    )
    with pytest.raises(IntegrationDependencyError, match="could not be created") as exc_info:
        factory.create_client_family(credential_material={"kind": "opaque"})
    assert _SECRET_FRAGMENT not in str(exc_info.value)
    assert "credential payload" not in str(exc_info.value)


class _InvalidMaterialResolver:
    def resolve_credential(self, credential_ref: str) -> Mapping[str, str]:
        return {"kind": 123}  # type: ignore[dict-item]


def test_invalid_resolved_credential_material_rejected_before_factory() -> None:
    resolver = _InvalidMaterialResolver()
    factory = _SpyClientFactory()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=resolver,
        client_factory=factory,
    )
    with pytest.raises(IntegrationConfigurationError, match="credential material is invalid"):
        integration.require_client_family()
    assert factory.calls == []


def test_failed_family_validation_is_not_cached() -> None:
    resolver = _SpyCredentialResolver()
    factory = _InvalidFamilyFactory()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=resolver,
        client_factory=factory,
    )
    with pytest.raises(IntegrationConfigurationError):
        integration.require_client_family()
    assert integration._client_family is None
    with pytest.raises(IntegrationConfigurationError):
        integration.require_client_family()
    assert resolver.calls == ["refs/google-workspace", "refs/google-workspace"]


def test_later_valid_call_materializes_and_caches_family() -> None:
    resolver = _SpyCredentialResolver()
    failing_factory = _InvalidFamilyFactory()
    working_factory = _SpyClientFactory()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=resolver,
        client_factory=failing_factory,
    )
    with pytest.raises(IntegrationConfigurationError):
        integration.require_client_family()
    integration._client_factory = working_factory
    first = integration.require_client_family()
    second = integration.require_client_family()
    assert first is second
    assert len(working_factory.calls) == 1


_INVALID_CREDENTIAL_MESSAGE = "Google Workspace credential material is invalid"
_SECRET_CREDENTIAL_FRAGMENT = "super-secret-credential-value"


class _RaisingItemsMapping(Mapping[str, str]):
    def __getitem__(self, key: str) -> str:
        return "opaque"

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 1

    def items(self):
        raise RuntimeError(f"secret mapping message {_SECRET_CREDENTIAL_FRAGMENT}")


class _ValidExecutor:
    def get(
        self,
        *,
        url: str,
        params: Mapping[str, object] | None,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> object:
        raise NotImplementedError


class _CountingExecutorFactory:
    def __init__(self) -> None:
        self.calls = 0

    def create_request_executor(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> _ValidExecutor:
        self.calls += 1
        return _ValidExecutor()


def _default_client_factory(
    executor_factory: object | None = None,
) -> object:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.client_family import (
        DefaultGoogleWorkspaceClientFactory,
    )
    from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
        GoogleWorkspaceRetryPolicy,
    )

    return DefaultGoogleWorkspaceClientFactory(
        executor_factory=executor_factory or _CountingExecutorFactory(),
        retry_policy=GoogleWorkspaceRetryPolicy(),
    )


@pytest.mark.parametrize(
    "credential_material",
    [
        "not-a-mapping",
        {},
        {1: "value"},  # type: ignore[dict-item]
        {"": "value"},
        {"kind": 123},  # type: ignore[dict-item]
        _RaisingItemsMapping(),
    ],
)
def test_default_factory_rejects_invalid_credential_material(
    credential_material: object,
) -> None:
    executor_factory = _CountingExecutorFactory()
    factory = _default_client_factory(executor_factory=executor_factory)
    with pytest.raises(IntegrationConfigurationError, match=_INVALID_CREDENTIAL_MESSAGE) as exc_info:
        factory.create_client_family(credential_material=credential_material)  # type: ignore[arg-type]
    assert executor_factory.calls == 0
    assert _SECRET_CREDENTIAL_FRAGMENT not in str(exc_info.value)
    assert "secret mapping message" not in str(exc_info.value)


def test_default_factory_passes_defensive_copy_to_executor_factory() -> None:
    class _MutatingExecutorFactory:
        def __init__(self) -> None:
            self.received: Mapping[str, str] | None = None

        def create_request_executor(
            self,
            *,
            credential_material: Mapping[str, str],
        ) -> _ValidExecutor:
            self.received = credential_material
            credential_material["mutated"] = "changed"  # type: ignore[index]
            return _ValidExecutor()

    executor_factory = _MutatingExecutorFactory()
    factory = _default_client_factory(executor_factory=executor_factory)
    caller_material = {"kind": "opaque", "scope": "drive"}
    factory.create_client_family(credential_material=caller_material)
    assert executor_factory.received is not caller_material
    assert caller_material == {"kind": "opaque", "scope": "drive"}
    assert executor_factory.received is not None
    assert executor_factory.received["kind"] == "opaque"
    assert executor_factory.received["scope"] == "drive"
    assert executor_factory.received["mutated"] == "changed"


class _TypedErrorRaisingItemsMapping(Mapping[str, str]):
    def __getitem__(self, key: str) -> str:
        return "opaque"

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 1

    def items(self):
        raise IntegrationConfigurationError(
            f"private_key={_SECRET_CREDENTIAL_FRAGMENT}"
        )


class _LazyTypedErrorIterator:
    def __iter__(self):
        return self

    def __next__(self) -> tuple[str, str]:
        raise IntegrationConfigurationError(
            f"private_key={_SECRET_CREDENTIAL_FRAGMENT}"
        )


class _LazyTypedErrorItemsMapping(Mapping[str, str]):
    def __getitem__(self, key: str) -> str:
        return "opaque"

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 1

    def items(self):
        return _LazyTypedErrorIterator()


@pytest.mark.parametrize(
    "credential_material",
    [
        _TypedErrorRaisingItemsMapping(),
        _LazyTypedErrorItemsMapping(),
    ],
)
def test_typed_credential_mapping_exception_is_canonicalized(
    credential_material: object,
) -> None:
    executor_factory = _CountingExecutorFactory()
    factory = _default_client_factory(executor_factory=executor_factory)
    with pytest.raises(
        IntegrationConfigurationError,
        match=_INVALID_CREDENTIAL_MESSAGE,
    ) as exc_info:
        factory.create_client_family(credential_material=credential_material)  # type: ignore[arg-type]
    assert executor_factory.calls == 0
    assert "private_key" not in str(exc_info.value)
    assert "private_key" not in repr(exc_info.value)
    assert _SECRET_CREDENTIAL_FRAGMENT not in str(exc_info.value)
    assert _SECRET_CREDENTIAL_FRAGMENT not in repr(exc_info.value)


@pytest.mark.parametrize(
    "credential_material",
    [
        _TypedErrorRaisingItemsMapping(),
        _LazyTypedErrorItemsMapping(),
    ],
)
def test_copy_credential_material_canonicalizes_typed_mapping_exception(
    credential_material: object,
) -> None:
    with pytest.raises(
        IntegrationConfigurationError,
        match=_INVALID_CREDENTIAL_MESSAGE,
    ) as exc_info:
        copy_google_workspace_credential_material(credential_material)
    assert "private_key" not in str(exc_info.value)
    assert "private_key" not in repr(exc_info.value)
    assert _SECRET_CREDENTIAL_FRAGMENT not in str(exc_info.value)
    assert _SECRET_CREDENTIAL_FRAGMENT not in repr(exc_info.value)


# --- Drive foundation delegation ---


class _DriveRecordingTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": dict(params or {}),
            }
        )
        if relative_path == "/drives":
            return {
                "drives": [
                    {
                        "id": "drive-1",
                        "name": "Team",
                        "createdTime": "2024-01-01T00:00:00Z",
                        "hidden": False,
                    },
                ],
            }
        if relative_path == "/files":
            return {"files": []}
        if relative_path == "/changes/startPageToken":
            return {"startPageToken": "start-token"}
        if relative_path == "/changes":
            return {"changes": [], "newStartPageToken": "checkpoint"}
        return {}


@dataclass(frozen=True, slots=True)
class _DriveFakeClientFamily:
    _transport: _DriveRecordingTransport

    @property
    def transport(self) -> _DriveRecordingTransport:
        return self._transport


def test_disabled_integration_blocks_drive_methods() -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=False,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=_SpyCredentialResolver(),
        client_factory=_SpyClientFactory(),
    )
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
        GoogleDriveScope,
        GoogleDriveScopeKind,
    )

    scope = GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
    with pytest.raises(IntegrationConfigurationError, match="disabled"):
        integration.list_drive_shared_drives_page()
    with pytest.raises(IntegrationConfigurationError, match="disabled"):
        integration.read_drive_items_page(scope=scope)


def test_first_drive_call_materializes_client_family_once() -> None:
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
    transport = _DriveRecordingTransport()
    integration._client_family = _DriveFakeClientFamily(_transport=transport)  # type: ignore[assignment]
    integration.list_drive_shared_drives_page()
    assert resolver.calls == []
    assert factory.calls == []


def test_multiple_drive_operations_reuse_same_family_and_transport() -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
        GoogleDriveScope,
        GoogleDriveScopeKind,
    )
    from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
        GoogleWorkspacePageToken,
    )

    transport = _DriveRecordingTransport()
    family = _DriveFakeClientFamily(_transport=transport)
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=_SpyCredentialResolver(),
        client_factory=_SpyClientFactory(),
    )
    integration._client_family = family  # type: ignore[assignment]
    scope = GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
    integration.list_drive_shared_drives_page()
    integration.read_drive_items_page(scope=scope)
    integration.read_drive_start_page_token(scope=scope)
    integration.read_drive_changes_page(
        scope=scope,
        page_token=GoogleWorkspacePageToken(value="page-1"),
    )
    assert len(transport.calls) == 4
    assert integration.require_client_family() is family


def test_health_and_construction_do_not_materialize_drive() -> None:
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
    assert integration._client_family is None


def test_injected_valid_client_family_serves_drive_operations() -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
        GoogleDriveScope,
        GoogleDriveScopeKind,
    )

    transport = _DriveRecordingTransport()
    family = _DriveFakeClientFamily(_transport=transport)
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        family,  # type: ignore[arg-type]
        enabled=True,
    )
    scope = GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
    page = integration.list_drive_shared_drives_page()
    assert len(page.items) == 1
    integration.read_drive_items_page(scope=scope)
    assert len(transport.calls) == 2


def test_lazy_drive_foundation_exports_resolve() -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
        GoogleDriveKnowledgeReader,
    )

    assert google_workspace.GOOGLE_DRIVE_SOURCE_KIND == "drive"
    assert google_workspace.GoogleDriveKnowledgeReader is GoogleDriveKnowledgeReader


def test_existing_public_exports_remain_present_with_drive_symbols() -> None:
    public_names = set(google_workspace.__all__)
    assert "GoogleWorkspaceCollaborationSuiteIntegration" in public_names
    assert "GoogleWorkspacePageToken" in public_names
    assert "GoogleDriveKnowledgeReader" in public_names
    assert "GoogleDriveContentReader" in public_names
    assert "GoogleWorkspaceBinaryTransport" in public_names


# --- Drive content foundation delegation ---


class _DualDriveTransport:
    def __init__(self) -> None:
        self.json_calls: list[dict[str, object]] = []
        self.binary_calls: list[dict[str, object]] = []

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self.json_calls.append(
            {"relative_path": relative_path, "params": dict(params or {})}
        )
        if relative_path == "/drives":
            return {"drives": []}
        if relative_path.startswith("/files/") and not relative_path.endswith("/export"):
            return {
                "id": "file-blob-1",
                "name": "report.pdf",
                "mimeType": "application/pdf",
                "parents": ["parent-1"],
                "createdTime": "2024-01-01T12:00:00Z",
                "modifiedTime": "2024-01-02T12:00:00Z",
                "size": "4",
                "md5Checksum": hashlib.md5(b"test", usedforsecurity=False).hexdigest(),
                "version": "5",
                "headRevisionId": "head-rev-1",
                "trashed": False,
                "capabilities": {"canDownload": True},
            }
        return {}

    def get_bytes(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None,
        expected_content_type: str,
        max_bytes: int,
        range_limited: bool,
    ) -> GoogleWorkspaceBinaryPayload:
        self.binary_calls.append({"relative_path": relative_path})
        return GoogleWorkspaceBinaryPayload(data=b"test", content_type=expected_content_type)


@dataclass(frozen=True, slots=True)
class _DualDriveClientFamily:
    _transport: _DualDriveTransport

    @property
    def transport(self) -> _DualDriveTransport:
        return self._transport


def test_disabled_integration_blocks_drive_content() -> None:
    from datetime import datetime, timezone

    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
        GoogleDriveItem,
        GoogleDriveItemKind,
        GoogleDriveScope,
        GoogleDriveScopeKind,
    )

    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=False,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=_SpyCredentialResolver(),
        client_factory=_SpyClientFactory(),
    )
    item = GoogleDriveItem(
        remote_id="file-blob-1",
        scope=GoogleDriveScope(kind=GoogleDriveScopeKind.USER),
        kind=GoogleDriveItemKind.BLOB,
        name="report.pdf",
        mime_type="application/pdf",
        parent_ids=("parent-1",),
        created_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
        size_bytes=4,
        md5_checksum="abcd",
        version=5,
        head_revision_id="head-rev-1",
        can_download=True,
    )
    with pytest.raises(IntegrationConfigurationError, match="disabled"):
        integration.read_drive_file_content(item=item)


def test_first_content_request_materializes_family_once() -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
        GoogleDriveKnowledgeReader,
        GoogleDriveScope,
        GoogleDriveScopeKind,
    )

    transport = _DualDriveTransport()
    family = _DualDriveClientFamily(_transport=transport)
    integration = GoogleWorkspaceCollaborationSuiteIntegration.compose(
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(
            enabled=True,
            credential_ref="refs/google-workspace",
        ),
        credential_resolver=_SpyCredentialResolver(),
        client_factory=_SpyClientFactory(),
    )
    integration._client_family = family  # type: ignore[assignment]
    scope = GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
    item = GoogleDriveKnowledgeReader(transport=transport).read_item(
        scope=scope,
        file_id="file-blob-1",
    )
    integration.read_drive_file_content(item=item)
    integration.read_drive_file_content(item=item)
    assert len(transport.binary_calls) == 2
    assert len([c for c in transport.json_calls if c["relative_path"].startswith("/files/")]) == 5


def test_json_only_injected_family_still_serves_metadata_but_not_content() -> None:
    from datetime import datetime, timezone

    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
        GoogleDriveItem,
        GoogleDriveItemKind,
        GoogleDriveScope,
        GoogleDriveScopeKind,
    )

    transport = _DriveRecordingTransport()
    family = _DriveFakeClientFamily(_transport=transport)
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        family,  # type: ignore[arg-type]
        enabled=True,
    )
    scope = GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
    integration.list_drive_shared_drives_page()
    item = GoogleDriveItem(
        remote_id="file-blob-1",
        scope=scope,
        kind=GoogleDriveItemKind.BLOB,
        name="report.pdf",
        mime_type="application/pdf",
        parent_ids=("parent-1",),
        created_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
        size_bytes=4,
        md5_checksum="abcd",
        version=5,
        head_revision_id="head-rev-1",
        can_download=True,
    )
    with pytest.raises(IntegrationConfigurationError, match="binary content"):
        integration.read_drive_file_content(item=item)


def test_lazy_drive_content_exports_resolve() -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive_content import (
        GoogleDriveContentReader,
    )

    assert google_workspace.GoogleDriveContentReader is GoogleDriveContentReader
    assert google_workspace.GoogleWorkspaceBinaryPayload is not None
