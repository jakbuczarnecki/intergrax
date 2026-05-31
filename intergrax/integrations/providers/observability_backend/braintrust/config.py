# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Braintrust observability integration configuration."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig


class BraintrustIntegrationConfig(BaseIntegrationConfig):
    base_url: str = "https://api.braintrust.dev"
    api_key: str = ""
    project: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> BraintrustIntegrationConfig:
        payload = {
            "base_url": os.environ.get("INTERGRAX_BRAINTRUST_URL", "https://api.braintrust.dev").strip()
            or "https://api.braintrust.dev",
            "api_key": os.environ.get("INTERGRAX_BRAINTRUST_API_KEY", "").strip()
            or os.environ.get("INTERGRAX_BRAINTRUST_TOKEN", "").strip(),
            "project": os.environ.get("INTERGRAX_BRAINTRUST_PROJECT", "").strip()
            or os.environ.get("INTERGRAX_BRAINTRUST_ORG", "").strip(),
        }
        payload.update(overrides)
        return cls.model_validate(payload)
