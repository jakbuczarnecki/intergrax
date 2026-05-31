# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AWS cloud platform integration configuration (Phase M.6)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_AWS_REGION = "INTERGRAX_AWS_REGION"
ENV_AWS_PROFILE = "INTERGRAX_AWS_PROFILE"
ENV_AWS_ROLE_ARN = "INTERGRAX_AWS_ROLE_ARN"
ENV_AWS_ACCESS_KEY_ID = "INTERGRAX_AWS_ACCESS_KEY_ID"
ENV_AWS_SECRET_ACCESS_KEY = "INTERGRAX_AWS_SECRET_ACCESS_KEY"
ENV_AWS_ROLE_SESSION_NAME = "INTERGRAX_AWS_ROLE_SESSION_NAME"

DEFAULT_ROLE_SESSION_NAME = "intergrax"


class AwsIntegrationConfig(BaseIntegrationConfig):
    """
    AWS credential and region settings.

    Uses the standard boto3 credential chain when access keys are omitted
    (env vars, shared config, IAM role, …).
    """

    region: str = ""
    profile: str = ""
    role_arn: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    role_session_name: str = DEFAULT_ROLE_SESSION_NAME

    @classmethod
    def from_env(cls, **overrides: object) -> AwsIntegrationConfig:
        payload: dict[str, object] = {
            "region": os.environ.get(ENV_AWS_REGION, "").strip(),
            "profile": os.environ.get(ENV_AWS_PROFILE, "").strip(),
            "role_arn": os.environ.get(ENV_AWS_ROLE_ARN, "").strip(),
            "access_key_id": os.environ.get(ENV_AWS_ACCESS_KEY_ID, "").strip(),
            "secret_access_key": os.environ.get(ENV_AWS_SECRET_ACCESS_KEY, "").strip(),
            "role_session_name": (
                os.environ.get(ENV_AWS_ROLE_SESSION_NAME, "").strip()
                or DEFAULT_ROLE_SESSION_NAME
            ),
        }
        payload.update(overrides)
        return cls.model_validate(payload)
