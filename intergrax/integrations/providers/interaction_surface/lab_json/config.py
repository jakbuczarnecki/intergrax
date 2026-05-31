# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lab JSON interaction integration configuration (Phase M.4)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_LAB_JSON_DEFAULT_SOURCE = "INTERGRAX_LAB_JSON_DEFAULT_SOURCE"

DEFAULT_SOURCE = "lab_json"


class LabJsonIntegrationConfig(BaseIntegrationConfig):
    default_source: str = DEFAULT_SOURCE

    @classmethod
    def from_env(cls, **overrides: object) -> LabJsonIntegrationConfig:
        default_source = (
            os.environ.get(ENV_LAB_JSON_DEFAULT_SOURCE, DEFAULT_SOURCE).strip()
            or DEFAULT_SOURCE
        )
        payload: dict[str, object] = {"default_source": default_source}
        payload.update(overrides)
        return cls.model_validate(payload)
