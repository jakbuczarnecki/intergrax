# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GCP cloud platform integration configuration (Phase M.6)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_GCP_PROJECT_ID = "INTERGRAX_GCP_PROJECT_ID"
ENV_GCP_REGION = "INTERGRAX_GCP_REGION"
ENV_GCP_CREDENTIALS_FILE = "INTERGRAX_GCP_CREDENTIALS_FILE"


class GcpIntegrationConfig(BaseIntegrationConfig):
    """
    GCP auth settings for the cloud platform facade.

    When ``credentials_file`` is set, a service-account file is used; otherwise
    Application Default Credentials (Workload Identity, gcloud, env).
    """

    project_id: str = ""
    region: str = ""
    credentials_file: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> GcpIntegrationConfig:
        payload: dict[str, object] = {
            "project_id": os.environ.get(ENV_GCP_PROJECT_ID, "").strip(),
            "region": os.environ.get(ENV_GCP_REGION, "").strip(),
            "credentials_file": os.environ.get(ENV_GCP_CREDENTIALS_FILE, "").strip(),
        }
        payload.update(overrides)
        return cls.model_validate(payload)
