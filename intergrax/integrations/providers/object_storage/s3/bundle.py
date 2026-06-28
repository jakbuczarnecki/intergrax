# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete S3 integration bundle — the single composition root for S3 in Intergrax.

boto3 S3 clients are opened only in ``opens.py``. Tier-3 code MUST use
``create_s3_object_storage()``, ``create_s3_integration()``, or
``profile.resolve(IntegrationCategory.OBJECT_STORAGE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.providers.object_storage.s3.adapter import S3ObjectStorage
from intergrax.integrations.providers.object_storage.s3.client import S3BucketClient
from intergrax.integrations.providers.object_storage.s3.config import S3IntegrationConfig
from intergrax.integrations.providers.object_storage.s3.opens import open_s3_object_storage


@dataclass(frozen=True)
class S3IntegrationBundle:
    config: S3IntegrationConfig
    object_storage: S3ObjectStorage
    bucket_client: S3BucketClient


def resolve_s3_config(**overrides: object) -> S3IntegrationConfig:
    return S3IntegrationConfig.from_env(**overrides)


def create_s3_integration(
    *,
    object_storage: Optional[ObjectStorage] = None,
    s3_client: Optional[object] = None,
    session: Optional[object] = None,
    session_factory: Optional[Callable[[], object]] = None,
    s3_client_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> S3IntegrationBundle:
    config = resolve_s3_config(**config_overrides)
    store = open_s3_object_storage(
        config,
        implementation=object_storage,
        s3_client=s3_client,
        session=session,
        session_factory=session_factory,
        s3_client_factory=s3_client_factory,
    )
    assert isinstance(store, S3ObjectStorage)
    return S3IntegrationBundle(
        config=config,
        object_storage=store,
        bucket_client=store.bucket_client,
    )


def create_s3_object_storage(
    *,
    object_storage: Optional[ObjectStorage] = None,
    s3_client: Optional[object] = None,
    session: Optional[object] = None,
    session_factory: Optional[Callable[[], object]] = None,
    s3_client_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> S3ObjectStorage:
    """Catalog factory for ``"s3"`` / ``OBJECT_STORAGE``."""
    return create_s3_integration(
        object_storage=object_storage,
        s3_client=s3_client,
        session=session,
        session_factory=session_factory,
        s3_client_factory=s3_client_factory,
        **config_overrides,
    ).object_storage

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.object_storage.s3.integration import (
    S3_OBJECT_STORAGE_PROVIDER_ID,
    S3ObjectStorageIntegration,
    S3ObjectStorageIntegrationConfig,
    S3ObjectStorageClient,
)


def create_s3_object_storage_integration(
    *,
    client: S3ObjectStorageClient | None = None,
    enabled: bool = False,
) -> S3ObjectStorageIntegration:
    """
    Build a contract-based S3 object storage integration.

    The legacy facade (create_s3_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "S3 object storage integration requires an injected client when enabled=True",
        )
    if client is not None:
        return S3ObjectStorageIntegration.from_client(client, enabled=enabled)
    return S3ObjectStorageIntegration.for_provider(
        provider_id=S3_OBJECT_STORAGE_PROVIDER_ID,
        display_name="S3",
        config=S3ObjectStorageIntegrationConfig(enabled=enabled),
    )
