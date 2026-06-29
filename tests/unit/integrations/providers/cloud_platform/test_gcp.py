# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for GCP cloud platform integration provider (Phase M.6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_cloud_platform
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.cloud_platform.gcp.adapter import _GcpCloudPlatform
from intergrax.integrations.providers.cloud_platform.gcp.bundle import (
    GcpIntegrationBundle,
    create_gcp_cloud_platform,
    create_gcp_integration,
)
from intergrax.integrations.providers.cloud_platform.gcp.config import (
    ENV_GCP_PROJECT_ID,
    ENV_GCP_REGION,
    GcpIntegrationConfig,
)
from intergrax.integrations.providers.cloud_platform.gcp.integration import GcpCloudPlatformIntegration
from intergrax.integrations.providers.cloud_platform.gcp.register import register_gcp_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_GCP_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "gcp"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "GcpCloudPlatform(",
    "integrations.providers.gcp.adapter",
    "integrations.providers.gcp.opens",
    "import google",
    "from google",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


@dataclass
class _FakeCredentials:
    valid: bool = True
    project_id: str = "resolved-project"


def _gcp_config() -> GcpIntegrationConfig:
    return GcpIntegrationConfig(
        project_id="my-project",
        region="europe-west1",
    )


def _credential_factory(credentials: _FakeCredentials | None = None):
    fake = credentials or _FakeCredentials()

    def _factory() -> tuple[_FakeCredentials, str]:
        return fake, fake.project_id

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


def test_google_auth_only_imported_in_opens_module() -> None:
    violations: list[str] = []
    for path in _GCP_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "google" in text:
            violations.append(path.name)
    assert violations == []


def test_gcp_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _GCP_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_gcp_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_GCP_PROJECT_ID, "prod-project")
    monkeypatch.setenv(ENV_GCP_REGION, "us-central1")
    config = GcpIntegrationConfig.from_env()
    assert config.project_id == "prod-project"
    assert config.region == "us-central1"


def test_resolve_returns_gcp_native_slugs() -> None:
    factory, _ = _credential_factory()
    platform = create_gcp_cloud_platform(**_gcp_config().model_dump(), credential_factory=factory)

    assert platform.resolve("object_storage") == "gcs"
    assert platform.resolve("message_bus") == "pubsub"
    assert platform.resolve("relational_store") == "cloud_sql"
    assert platform.resolve("document_store") is None
    assert_cloud_platform(platform)


def test_health_reports_valid_credentials() -> None:
    factory, _ = _credential_factory()
    platform = create_gcp_cloud_platform(**_gcp_config().model_dump(), credential_factory=factory)

    status = platform.health()

    assert status.healthy is True
    assert "project=my-project" in status.detail


def test_health_reports_failure_without_raising() -> None:
    factory, _ = _credential_factory(_FakeCredentials(valid=False))
    platform = create_gcp_cloud_platform(**_gcp_config().model_dump(), credential_factory=factory)

    status = platform.health()

    assert status.healthy is False
    assert "not valid" in status.detail


def test_create_gcp_integration_bundle() -> None:
    factory, _ = _credential_factory()
    bundle = create_gcp_integration(**_gcp_config().model_dump(), credential_factory=factory)

    assert isinstance(bundle, GcpIntegrationBundle)
    assert isinstance(bundle.cloud_platform, GcpCloudPlatformIntegration)
    assert bundle.cloud_platform.default_region == "europe-west1"
    assert bundle.cloud_platform.project_id == "my-project"


def test_register_and_resolve_via_profile() -> None:
    register_gcp_integration()
    profile = IntegrationProfile(cloud_platform="gcp")
    factory, _ = _credential_factory()

    platform = resolve(
        IntegrationCategory.CLOUD_PLATFORM,
        profile=profile,
        config={**_gcp_config().model_dump(), "credential_factory": factory},
    )

    assert_cloud_platform(platform)
    assert isinstance(platform, GcpCloudPlatformIntegration)


def test_register_default_integrations_includes_gcp() -> None:
    register_default_integrations()
    profile = IntegrationProfile(cloud_platform="gcp")
    factory, _ = _credential_factory()

    platform = resolve(
        IntegrationCategory.CLOUD_PLATFORM,
        profile=profile,
        config={**_gcp_config().model_dump(), "credential_factory": factory},
    )

    assert isinstance(platform, GcpCloudPlatformIntegration)


def test_opens_uses_service_account_file_when_configured() -> None:
    config = GcpIntegrationConfig(
        credentials_file="/tmp/sa.json",
        project_id="file-project",
    )
    mock_credentials = MagicMock()
    mock_credentials.valid = True
    mock_credentials.project_id = "file-project"

    with patch(
        "intergrax.integrations.providers.cloud_platform.gcp.opens._import_google_auth",
    ) as import_mock:
        service_account = MagicMock()
        service_account.Credentials.from_service_account_file.return_value = mock_credentials
        import_mock.return_value = (MagicMock(), MagicMock(), service_account)
        from intergrax.integrations.providers.cloud_platform.gcp.opens import open_gcp_credentials

        credentials, project_id = open_gcp_credentials(config)

    service_account.Credentials.from_service_account_file.assert_called_once()
    assert credentials is mock_credentials
    assert project_id == "file-project"


def test_opens_uses_adc_when_no_credentials_file() -> None:
    config = GcpIntegrationConfig(project_id="")
    mock_credentials = MagicMock()
    mock_credentials.valid = True

    with patch(
        "intergrax.integrations.providers.cloud_platform.gcp.opens._import_google_auth",
    ) as import_mock:
        google_auth = MagicMock()
        google_auth.default.return_value = (mock_credentials, "adc-project")
        import_mock.return_value = (google_auth, MagicMock(), MagicMock())
        from intergrax.integrations.providers.cloud_platform.gcp.opens import open_gcp_credentials

        credentials, project_id = open_gcp_credentials(config)

    google_auth.default.assert_called_once()
    assert credentials is mock_credentials
    assert project_id == "adc-project"


def test_opens_uses_credential_factory_when_injected() -> None:
    config = _gcp_config()
    mock_credentials = _FakeCredentials()

    with patch(
        "intergrax.integrations.providers.cloud_platform.gcp.opens._import_google_auth",
    ) as import_mock:
        from intergrax.integrations.providers.cloud_platform.gcp.opens import open_gcp_credentials

        credentials, project_id = open_gcp_credentials(
            config,
            credential_factory=lambda: (mock_credentials, "factory-project"),
        )

    import_mock.assert_not_called()
    assert credentials is mock_credentials
    assert project_id == "factory-project"
