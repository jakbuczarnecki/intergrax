# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""S3 object storage integration configuration (Phase M.6 P2)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError

ENV_S3_BUCKET = "INTERGRAX_S3_BUCKET"
ENV_S3_REGION = "INTERGRAX_S3_REGION"
ENV_S3_PREFIX = "INTERGRAX_S3_PREFIX"
ENV_S3_ENDPOINT_URL = "INTERGRAX_S3_ENDPOINT_URL"
ENV_S3_PROFILE = "INTERGRAX_S3_PROFILE"
ENV_S3_ACCESS_KEY_ID = "INTERGRAX_S3_ACCESS_KEY_ID"
ENV_S3_SECRET_ACCESS_KEY = "INTERGRAX_S3_SECRET_ACCESS_KEY"
ENV_S3_ROLE_ARN = "INTERGRAX_S3_ROLE_ARN"
ENV_S3_ROLE_SESSION_NAME = "INTERGRAX_S3_ROLE_SESSION_NAME"

DEFAULT_ROLE_SESSION_NAME = "intergrax"


class S3IntegrationConfig(BaseIntegrationConfig):
    """
    AWS S3 settings for ``ObjectStorage``.

    Uses the standard boto3 credential chain when access keys are omitted.
    """

    bucket: str = ""
    region: str = ""
    prefix: str = ""
    endpoint_url: str = ""
    profile: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    role_arn: str = ""
    role_session_name: str = DEFAULT_ROLE_SESSION_NAME

    def require_bucket(self) -> str:
        bucket = self.bucket.strip()
        if not bucket:
            raise IntegrationConfigurationError(
                "S3 integration requires bucket (INTERGRAX_S3_BUCKET or bucket=...)"
            )
        return bucket

    def object_key(self, key: str) -> str:
        normalized = key.lstrip("/")
        prefix = self.prefix.strip("/")
        if prefix:
            return f"{prefix}/{normalized}"
        return normalized

    @classmethod
    def from_env(cls, **overrides: object) -> S3IntegrationConfig:
        payload: dict[str, object] = {
            "bucket": os.environ.get(ENV_S3_BUCKET, "").strip(),
            "region": os.environ.get(ENV_S3_REGION, "").strip(),
            "prefix": os.environ.get(ENV_S3_PREFIX, "").strip(),
            "endpoint_url": os.environ.get(ENV_S3_ENDPOINT_URL, "").strip(),
            "profile": os.environ.get(ENV_S3_PROFILE, "").strip(),
            "access_key_id": os.environ.get(ENV_S3_ACCESS_KEY_ID, "").strip(),
            "secret_access_key": os.environ.get(ENV_S3_SECRET_ACCESS_KEY, "").strip(),
            "role_arn": os.environ.get(ENV_S3_ROLE_ARN, "").strip(),
            "role_session_name": (
                os.environ.get(ENV_S3_ROLE_SESSION_NAME, "").strip()
                or DEFAULT_ROLE_SESSION_NAME
            ),
        }
        payload.update(overrides)
        return cls.model_validate(payload)
