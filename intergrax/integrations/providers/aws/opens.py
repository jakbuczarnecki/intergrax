# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level AWS session openers — internal to the aws integration package.

Only this module may construct boto3 ``Session`` instances for the AWS cloud facade.
All composition roots use ``bundle.create_aws_*`` or ``profile.resolve(CLOUD_PLATFORM)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.providers.aws.adapter import AwsCloudPlatform
from intergrax.integrations.providers.aws.config import AwsIntegrationConfig


def _import_boto3() -> Any:
    import boto3

    return boto3


def _build_base_session(config: AwsIntegrationConfig) -> Any:
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


def _assume_role_session(config: AwsIntegrationConfig, base_session: Any) -> Any:
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


def open_aws_boto_session(
    config: AwsIntegrationConfig,
    *,
    session_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    if session_factory is not None:
        return session_factory()
    session = _build_base_session(config)
    if config.role_arn:
        return _assume_role_session(config, session)
    return session


def open_aws_cloud_platform(
    config: AwsIntegrationConfig,
    *,
    implementation: Optional[CloudPlatform] = None,
    session: Optional[Any] = None,
    session_factory: Optional[Callable[[], Any]] = None,
) -> CloudPlatform:
    if implementation is not None:
        return implementation
    boto_session = session or open_aws_boto_session(config, session_factory=session_factory)
    return AwsCloudPlatform(config, boto_session)
