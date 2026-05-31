# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for S3 integration provider (Phase M.6 P2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_object_storage
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.providers.object_storage.s3.adapter import S3ObjectStorage
from intergrax.integrations.providers.object_storage.s3.bundle import (
    S3IntegrationBundle,
    create_s3_integration,
    create_s3_object_storage,
)
from intergrax.integrations.providers.object_storage.s3.config import ENV_S3_BUCKET, ENV_S3_PREFIX, S3IntegrationConfig
from intergrax.integrations.providers.object_storage.s3.register import register_s3_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_S3_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "s3"


class _NoSuchKeyError(Exception):
    def __init__(self) -> None:
        super().__init__("NoSuchKey")
        self.response = {"Error": {"Code": "NoSuchKey"}}


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.operations: list[tuple[str, Any]] = []

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str = "application/octet-stream",
        Metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.operations.append(("put_object", Bucket, Key, Body, ContentType, Metadata))
        self.objects[Key] = {
            "Body": Body,
            "ContentType": ContentType,
            "Metadata": dict(Metadata or {}),
        }

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.operations.append(("get_object", Bucket, Key))
        if Key not in self.objects:
            raise _NoSuchKeyError()
        stored = self.objects[Key]
        return {
            "Body": _FakeBody(stored["Body"]),
            "ContentType": stored["ContentType"],
            "Metadata": stored["Metadata"],
        }

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.operations.append(("delete_object", Bucket, Key))
        self.objects.pop(Key, None)

    def generate_presigned_url(
        self,
        *,
        ClientMethod: str,
        Params: dict[str, str],
        ExpiresIn: int,
    ) -> str:
        self.operations.append(("generate_presigned_url", ClientMethod, Params, ExpiresIn))
        return (
            f"https://fake-s3/{Params['Bucket']}/{Params['Key']}"
            f"?method={ClientMethod}&expires={ExpiresIn}"
        )


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def _s3_config(**overrides: object) -> S3IntegrationConfig:
    base = {"bucket": "intergrax-artifacts", "region": "eu-central-1"}
    base.update(overrides)
    return S3IntegrationConfig.model_validate(base)


def _client_factory(client: _FakeS3Client | None = None):
    fake = client or _FakeS3Client()

    def _factory() -> _FakeS3Client:
        return fake

    return _factory, fake


def test_boto3_only_imported_in_opens_module() -> None:
    violations = [
        path.name
        for path in _S3_PKG.glob("*.py")
        if path.name != "opens.py" and "boto3" in path.read_text(encoding="utf-8")
    ]
    assert violations == []


def test_s3_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_S3_BUCKET, "prod-bucket")
    monkeypatch.setenv(ENV_S3_PREFIX, "tenant-a")
    config = S3IntegrationConfig.from_env()
    assert config.bucket == "prod-bucket"
    assert config.object_key("exports/run-1.zip") == "tenant-a/exports/run-1.zip"


def test_s3_config_requires_bucket() -> None:
    factory, _ = _client_factory()
    with pytest.raises(IntegrationConfigurationError, match="bucket"):
        create_s3_object_storage(bucket="", s3_client_factory=factory)


def test_put_and_get_round_trip() -> None:
    factory, client = _client_factory()
    store = create_s3_object_storage(**_s3_config().model_dump(), s3_client_factory=factory)

    store.put("artifacts/run-1/output.txt", b"hello", content_type="text/plain", metadata={"run": "1"})
    obj = store.get("artifacts/run-1/output.txt")

    assert obj is not None
    assert obj.body == b"hello"
    assert obj.content_type == "text/plain"
    assert obj.metadata["run"] == "1"
    assert any(op[0] == "put_object" for op in client.operations)
    assert_object_storage(store)


def test_get_returns_none_for_missing_key() -> None:
    factory, _ = _client_factory()
    store = create_s3_object_storage(**_s3_config().model_dump(), s3_client_factory=factory)
    assert store.get("missing.bin") is None


def test_delete_removes_object() -> None:
    factory, client = _client_factory()
    store = create_s3_object_storage(**_s3_config().model_dump(), s3_client_factory=factory)
    store.put("tmp.bin", b"x")

    store.delete("tmp.bin")

    assert store.get("tmp.bin") is None
    assert any(op[0] == "delete_object" for op in client.operations)


def test_presigned_url_get_and_put() -> None:
    factory, client = _client_factory()
    store = create_s3_object_storage(**_s3_config().model_dump(), s3_client_factory=factory)

    get_url = store.presigned_url("uploads/doc.pdf", expires_in_seconds=900)
    put_url = store.presigned_url("uploads/doc.pdf", expires_in_seconds=600, method="PUT")

    assert "get_object" in get_url
    assert "put_object" in put_url
    assert len([op for op in client.operations if op[0] == "generate_presigned_url"]) == 2


def test_prefix_is_applied_to_object_keys() -> None:
    factory, client = _client_factory()
    store = create_s3_object_storage(
        **_s3_config(prefix="tenant-a").model_dump(),
        s3_client_factory=factory,
    )

    store.put("file.bin", b"data")

    assert "tenant-a/file.bin" in client.objects


def test_close_blocks_further_operations() -> None:
    factory, _ = _client_factory()
    store = create_s3_object_storage(**_s3_config().model_dump(), s3_client_factory=factory)
    store.close()

    with pytest.raises(IntegrationConfigurationError, match="closed"):
        store.put("x", b"y")


def test_create_s3_integration_bundle() -> None:
    factory, _ = _client_factory()
    bundle = create_s3_integration(**_s3_config().model_dump(), s3_client_factory=factory)

    assert isinstance(bundle, S3IntegrationBundle)
    assert isinstance(bundle.object_storage, S3ObjectStorage)
    assert bundle.config.bucket == "intergrax-artifacts"


def test_register_and_resolve_via_profile() -> None:
    register_s3_integration()
    profile = IntegrationProfile(object_storage=IntegrationSlug.S3)
    factory, _ = _client_factory()

    store = resolve(
        IntegrationCategory.OBJECT_STORAGE,
        profile=profile,
        config={**_s3_config().model_dump(), "s3_client_factory": factory},
    )

    assert_object_storage(store)
    assert isinstance(store, S3ObjectStorage)


def test_register_default_integrations_includes_s3() -> None:
    register_default_integrations()
    profile = IntegrationProfile(object_storage=IntegrationSlug.S3)
    factory, _ = _client_factory()

    store = resolve(
        IntegrationCategory.OBJECT_STORAGE,
        profile=profile,
        config={**_s3_config().model_dump(), "s3_client_factory": factory},
    )

    assert isinstance(store, S3ObjectStorage)


def test_cloud_platform_profile_resolves_s3_by_default() -> None:
    register_default_integrations()
    profile = IntegrationProfile.with_cloud_platform(IntegrationSlug.AWS)
    factory, _ = _client_factory()

    store = resolve(
        IntegrationCategory.OBJECT_STORAGE,
        profile=profile,
        config={**_s3_config().model_dump(), "s3_client_factory": factory},
    )

    assert isinstance(store, S3ObjectStorage)


def test_opens_builds_boto_client_when_not_injected() -> None:
    config = _s3_config()
    mock_client = MagicMock()

    with patch(
        "intergrax.integrations.providers.object_storage.s3.opens.open_s3_boto_session",
        return_value=MagicMock(region_name="eu-central-1"),
    ) as session_mock:
        session_mock.return_value.client.return_value = mock_client
        from intergrax.integrations.providers.object_storage.s3.opens import open_s3_bucket_client

        bucket_client = open_s3_bucket_client(config)

    assert bucket_client.s3_client is mock_client
    assert bucket_client.bucket == "intergrax-artifacts"


def test_missing_boto3_raises_configuration_error() -> None:
    config = _s3_config()
    with patch(
        "intergrax.integrations.providers.object_storage.s3.opens._import_boto3",
        side_effect=IntegrationConfigurationError("missing boto3"),
    ):
        from intergrax.integrations.providers.object_storage.s3.opens import open_s3_object_storage

        with pytest.raises(IntegrationConfigurationError, match="missing boto3"):
            open_s3_object_storage(config)
