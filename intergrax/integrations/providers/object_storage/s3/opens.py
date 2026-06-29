# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level S3 client openers — internal to the s3 integration package.

Only this module may construct boto3 S3 clients. All composition roots use
``bundle.create_s3_*`` or ``profile.resolve(OBJECT_STORAGE)``.
"""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.providers.object_storage.s3.adapter import _S3ObjectStorage
from intergrax.integrations.providers.object_storage.s3.integration import S3ObjectStorageIntegration
from intergrax.integrations.providers.object_storage.s3.client import S3BucketClient
from intergrax.integrations.providers.object_storage.s3.config import S3IntegrationConfig


def _import_boto3() -> Any:
    try:
        import boto3
    except ImportError as exc:
        from intergrax.integrations.contracts.base import IntegrationConfigurationError

        raise IntegrationConfigurationError(
            "S3 integration requires boto3. Install with: uv sync  (includes boto3)"
        ) from exc
    return boto3


def _build_base_session(config: S3IntegrationConfig) -> Any:
    boto3 = _import_boto3()
    session_kwargs: dict[str, str] = {}
    if config.profile:
        session_kwargs["profile_name"] = config.profile
    if config.region:
        session_kwargs["region_name"] = config.region
    if config.access_key_id and config.secret_access_key:
        session_kwargs["aws_access_key_id"] = config.access_key_id
        session_kwargs["aws_secret_access_key"] = config.secret_access_key
    return boto3.Session(**session_kwargs)


def _assume_role_session(config: S3IntegrationConfig, base_session: Any) -> Any:
    boto3 = _import_boto3()
    sts = base_session.client("sts", region_name=config.region or base_session.region_name)
    response = sts.assume_role(
        RoleArn=config.role_arn,
        RoleSessionName=config.role_session_name,
    )
    credentials = response["Credentials"]
    return boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=config.region or base_session.region_name,
    )


def open_s3_boto_session(
    config: S3IntegrationConfig,
    *,
    session: Optional[Any] = None,
    session_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    if session is not None:
        return session
    if session_factory is not None:
        return session_factory()
    base = _build_base_session(config)
    if config.role_arn:
        return _assume_role_session(config, base)
    return base


def open_s3_client(
    config: S3IntegrationConfig,
    *,
    s3_client: Optional[Any] = None,
    session: Optional[Any] = None,
    session_factory: Optional[Callable[[], Any]] = None,
    s3_client_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    if s3_client is not None:
        return s3_client
    if s3_client_factory is not None:
        return s3_client_factory()
    boto_session = open_s3_boto_session(config, session=session, session_factory=session_factory)
    client_kwargs: dict[str, str] = {}
    if config.endpoint_url:
        client_kwargs["endpoint_url"] = config.endpoint_url
    region = config.region or attribute_access.optional(boto_session, "region_name", None)
    return boto_session.client("s3", region_name=region, **client_kwargs)


def open_s3_bucket_client(
    config: S3IntegrationConfig,
    *,
    s3_client: Optional[Any] = None,
    session: Optional[Any] = None,
    session_factory: Optional[Callable[[], Any]] = None,
    s3_client_factory: Optional[Callable[[], Any]] = None,
) -> S3BucketClient:
    client = open_s3_client(
        config,
        s3_client=s3_client,
        session=session,
        session_factory=session_factory,
        s3_client_factory=s3_client_factory,
    )
    return S3BucketClient(config, client)


def open_s3_object_storage(
    config: S3IntegrationConfig,
    *,
    implementation: Optional[ObjectStorage] = None,
    s3_client: Optional[Any] = None,
    session: Optional[Any] = None,
    session_factory: Optional[Callable[[], Any]] = None,
    s3_client_factory: Optional[Callable[[], Any]] = None,
) -> ObjectStorage:
    if implementation is not None:
        return implementation
    bucket_client = open_s3_bucket_client(
        config,
        s3_client=s3_client,
        session=session,
        session_factory=session_factory,
        s3_client_factory=s3_client_factory,
    )
    return S3ObjectStorageIntegration.from_runtime(_S3ObjectStorage(bucket_client))