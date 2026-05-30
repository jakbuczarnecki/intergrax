# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Log notification integration configuration (Phase M.8)."""

from __future__ import annotations

from intergrax.integrations._shared.config import BaseIntegrationConfig


class LogIntegrationConfig(BaseIntegrationConfig):
    """No external settings — logging uses the process logger."""

    @classmethod
    def from_env(cls, **overrides: object) -> LogIntegrationConfig:
        return cls.model_validate(dict(overrides))
