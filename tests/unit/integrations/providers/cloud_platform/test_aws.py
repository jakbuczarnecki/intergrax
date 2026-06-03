# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for AWS cloud platform integration provider (Phase M.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_cloud_platform
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.cloud_platform.aws.adapter import AwsCloudPlatform
from intergrax.integrations.providers.cloud_platform.aws.bundle import (
    AwsIntegrationBundle,
    create_aws_cloud_platform,
    create_aws_integration,
)
from intergrax.integrations.providers.cloud_platform.aws.config import (
    ENV_AWS_REGION,
    ENV_AWS_ROLE_ARN,
    AwsIntegrationConfig,
)
from intergrax.integrations.providers.cloud_platform.aws.register import register_aws_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_AWS_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "aws"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "AwsCloudPlatform(",
    "integrations.providers.aws.adapter",
    "integrations.providers.aws.opens",
    "import boto3",
    "from boto3",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeStsClient:
    def __init__(self, *, arn: str = "arn:aws:sts::123456789012:assumed-role/demo") -> None:
        self._arn = arn
        self.assume_role_calls: list[dict[str, str]] = []

    def get_caller_identity(self) -> dict[str, str]:
        return {"Arn": self._arn}

    def assume_role(self, **kwargs: str) -> dict[str, object]:
        self.assume_role_calls.append(kwargs)
        return {
            "Credentials": {
                "AccessKeyId": "ASIAFAKE",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }


class _FakeSession:
    def __init__(self, *, region_name: str = "eu-central-1") -> None:
        self.region_name = region_name
        self.sts = _FakeStsClient()

    def client(self, service_name: str, region_name: str | None = None) -> _FakeStsClient:
        assert service_name == "sts"
        return self.sts


def _aws_config() -> AwsIntegrationConfig:
    return AwsIntegrationConfig(region="eu-central-1", profile="dev")


def _session_factory(session: _FakeSession | None = None):
    fake = session or _FakeSession()

    def _factory() -> _FakeSession:
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


def test_boto3_only_imported_in_opens_module() -> None:
    violations: list[str] = []
    for path in _AWS_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "boto3" in text:
            violations.append(path.name)
    assert violations == []


def test_aws_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _AWS_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_aws_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_AWS_REGION, "us-east-1")
    monkeypatch.setenv(ENV_AWS_ROLE_ARN, "arn:aws:iam::123:role/app")
    config = AwsIntegrationConfig.from_env()
    assert config.region == "us-east-1"
    assert config.role_arn.endswith(":role/app")


def test_resolve_returns_aws_native_slugs() -> None:
    factory, _ = _session_factory()
    platform = create_aws_cloud_platform(**_aws_config().model_dump(), session_factory=factory)

    assert platform.resolve("object_storage") == "s3"
    assert platform.resolve("message_bus") == "sqs"
    assert platform.resolve("document_store") == "dynamodb"
    assert platform.resolve("key_value_cache") == "elasticache"
    assert platform.resolve("relational_store") is None
    assert_cloud_platform(platform)


def test_health_reports_caller_identity() -> None:
    factory, session = _session_factory()
    platform = create_aws_cloud_platform(**_aws_config().model_dump(), session_factory=factory)

    status = platform.health()

    assert status.healthy is True
    assert "assumed-role/demo" in status.detail
    session.client("sts").get_caller_identity()


def test_health_reports_failure_without_raising() -> None:
    factory, session = _session_factory()
    session.sts.get_caller_identity = MagicMock(side_effect=RuntimeError("denied"))  # type: ignore[method-assign]
    platform = create_aws_cloud_platform(**_aws_config().model_dump(), session_factory=factory)

    status = platform.health()

    assert status.healthy is False
    assert "denied" in status.detail


def test_create_aws_integration_bundle() -> None:
    factory, _ = _session_factory()
    bundle = create_aws_integration(**_aws_config().model_dump(), session_factory=factory)

    assert isinstance(bundle, AwsIntegrationBundle)
    assert isinstance(bundle.cloud_platform, AwsCloudPlatform)
    assert bundle.cloud_platform.default_region == "eu-central-1"


def test_register_and_resolve_via_profile() -> None:
    register_aws_integration()
    profile = IntegrationProfile(cloud_platform="aws")
    factory, _ = _session_factory()

    platform = resolve(
        IntegrationCategory.CLOUD_PLATFORM,
        profile=profile,
        config={**_aws_config().model_dump(), "session_factory": factory},
    )

    assert_cloud_platform(platform)
    assert isinstance(platform, AwsCloudPlatform)


def test_register_default_integrations_includes_aws() -> None:
    register_default_integrations()
    profile = IntegrationProfile(cloud_platform="aws")
    factory, _ = _session_factory()

    platform = resolve(
        IntegrationCategory.CLOUD_PLATFORM,
        profile=profile,
        config={**_aws_config().model_dump(), "session_factory": factory},
    )

    assert isinstance(platform, AwsCloudPlatform)


def test_opens_assumes_role_when_configured() -> None:
    config = AwsIntegrationConfig(
        region="eu-central-1",
        role_arn="arn:aws:iam::123:role/app",
        role_session_name="intergrax-test",
    )
    base_session = _FakeSession()

    with patch(
        "intergrax.integrations.providers.cloud_platform.aws.opens._build_base_session",
        return_value=base_session,
    ):
        with patch("intergrax.integrations.providers.cloud_platform.aws.opens._import_boto3") as import_mock:
            assumed_session = MagicMock()
            import_mock.return_value.Session.return_value = assumed_session
            from intergrax.integrations.providers.cloud_platform.aws.opens import open_aws_boto_session

            session = open_aws_boto_session(config)

    assert base_session.sts.assume_role_calls
    assert base_session.sts.assume_role_calls[0]["RoleArn"] == config.role_arn
    assert session is assumed_session


def test_opens_uses_session_factory_when_injected() -> None:
    config = _aws_config()
    mock_session = _FakeSession()

    with patch(
        "intergrax.integrations.providers.cloud_platform.aws.opens._build_base_session",
    ) as build_mock:
        from intergrax.integrations.providers.cloud_platform.aws.opens import open_aws_boto_session

        session = open_aws_boto_session(config, session_factory=lambda: mock_session)

    build_mock.assert_not_called()
    assert session is mock_session
