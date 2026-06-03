# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Azure cloud platform integration provider (Phase M.6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_cloud_platform
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.cloud_platform.azure.adapter import AzureCloudPlatform
from intergrax.integrations.providers.cloud_platform.azure.bundle import (
    AzureIntegrationBundle,
    create_azure_cloud_platform,
    create_azure_integration,
)
from intergrax.integrations.providers.cloud_platform.azure.config import (
    ENV_AZURE_LOCATION,
    ENV_AZURE_TENANT_ID,
    AzureIntegrationConfig,
)
from intergrax.integrations.providers.cloud_platform.azure.register import register_azure_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_AZURE_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "azure"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "AzureCloudPlatform(",
    "integrations.providers.azure.adapter",
    "integrations.providers.azure.opens",
    "import azure",
    "from azure",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


@dataclass
class _FakeAccessToken:
    token: str = "fake-token"
    expires_on: int = 9999999999


class _FakeCredential:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.scopes: list[str] = []

    def get_token(self, *scopes: str) -> _FakeAccessToken:
        if self.fail:
            raise RuntimeError("auth denied")
        self.scopes.extend(scopes)
        return _FakeAccessToken()


def _azure_config() -> AzureIntegrationConfig:
    return AzureIntegrationConfig(
        tenant_id="tenant-1",
        client_id="client-1",
        client_secret="secret",
        location="westeurope",
        subscription_id="sub-123",
    )


def _credential_factory(credential: _FakeCredential | None = None):
    fake = credential or _FakeCredential()

    def _factory() -> _FakeCredential:
        return fake

    return _factory, fake


def _iter_python_files(*roots: str):
    for root_name in roots:
        root = _PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            yield path


def test_azure_identity_only_imported_in_opens_module() -> None:
    violations: list[str] = []
    for path in _AZURE_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "azure" in text:
            violations.append(path.name)
    assert violations == []


def test_azure_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _AZURE_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_azure_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_AZURE_TENANT_ID, "tenant-abc")
    monkeypatch.setenv(ENV_AZURE_LOCATION, "northeurope")
    config = AzureIntegrationConfig.from_env()
    assert config.tenant_id == "tenant-abc"
    assert config.location == "northeurope"
    assert config.uses_service_principal is False


def test_resolve_returns_azure_native_slugs() -> None:
    factory, _ = _credential_factory()
    platform = create_azure_cloud_platform(**_azure_config().model_dump(), credential_factory=factory)

    assert platform.resolve("object_storage") == "azure_blob"
    assert platform.resolve("message_bus") == "service_bus"
    assert platform.resolve("relational_store") == "azure_sql"
    assert platform.resolve("document_store") is None
    assert_cloud_platform(platform)


def test_health_reports_token_acquisition() -> None:
    factory, credential = _credential_factory()
    platform = create_azure_cloud_platform(**_azure_config().model_dump(), credential_factory=factory)

    status = platform.health()

    assert status.healthy is True
    assert "subscription=sub-123" in status.detail
    assert credential.scopes


def test_health_reports_failure_without_raising() -> None:
    factory, _ = _credential_factory(_FakeCredential(fail=True))
    platform = create_azure_cloud_platform(**_azure_config().model_dump(), credential_factory=factory)

    status = platform.health()

    assert status.healthy is False
    assert "auth denied" in status.detail


def test_create_azure_integration_bundle() -> None:
    factory, _ = _credential_factory()
    bundle = create_azure_integration(**_azure_config().model_dump(), credential_factory=factory)

    assert isinstance(bundle, AzureIntegrationBundle)
    assert isinstance(bundle.cloud_platform, AzureCloudPlatform)
    assert bundle.cloud_platform.default_region == "westeurope"


def test_register_and_resolve_via_profile() -> None:
    register_azure_integration()
    profile = IntegrationProfile(cloud_platform="azure")
    factory, _ = _credential_factory()

    platform = resolve(
        IntegrationCategory.CLOUD_PLATFORM,
        profile=profile,
        config={**_azure_config().model_dump(), "credential_factory": factory},
    )

    assert_cloud_platform(platform)
    assert isinstance(platform, AzureCloudPlatform)


def test_register_default_integrations_includes_azure() -> None:
    register_default_integrations()
    profile = IntegrationProfile(cloud_platform="azure")
    factory, _ = _credential_factory()

    platform = resolve(
        IntegrationCategory.CLOUD_PLATFORM,
        profile=profile,
        config={**_azure_config().model_dump(), "credential_factory": factory},
    )

    assert isinstance(platform, AzureCloudPlatform)


def test_opens_uses_service_principal_when_configured() -> None:
    config = _azure_config()
    mock_credential = MagicMock()

    with patch(
        "intergrax.integrations.providers.cloud_platform.azure.opens._import_azure_identity",
    ) as import_mock:
        import_mock.return_value = (MagicMock(return_value=mock_credential), MagicMock())
        ClientSecretCredential = import_mock.return_value[0]
        from intergrax.integrations.providers.cloud_platform.azure.opens import open_azure_credential

        credential = open_azure_credential(config)

    ClientSecretCredential.assert_called_once_with(
        tenant_id="tenant-1",
        client_id="client-1",
        client_secret="secret",
    )
    assert credential is mock_credential


def test_opens_uses_default_credential_without_service_principal() -> None:
    config = AzureIntegrationConfig()
    mock_credential = MagicMock()

    with patch(
        "intergrax.integrations.providers.cloud_platform.azure.opens._import_azure_identity",
    ) as import_mock:
        DefaultAzureCredential = MagicMock(return_value=mock_credential)
        import_mock.return_value = (MagicMock(), DefaultAzureCredential)
        from intergrax.integrations.providers.cloud_platform.azure.opens import open_azure_credential

        credential = open_azure_credential(config)

    DefaultAzureCredential.assert_called_once_with()
    assert credential is mock_credential


def test_opens_uses_credential_factory_when_injected() -> None:
    config = _azure_config()
    mock_credential = _FakeCredential()

    with patch(
        "intergrax.integrations.providers.cloud_platform.azure.opens._import_azure_identity",
    ) as import_mock:
        from intergrax.integrations.providers.cloud_platform.azure.opens import open_azure_credential

        credential = open_azure_credential(
            config,
            credential_factory=lambda: mock_credential,
        )

    import_mock.assert_not_called()
    assert credential is mock_credential
