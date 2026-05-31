# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slash-command interaction surface configuration."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_SLASH_COMMAND_DEFAULT_SOURCE = "INTERGRAX_SLASH_COMMAND_DEFAULT_SOURCE"
DEFAULT_SOURCE = "slash_command"


class SlashCommandIntegrationConfig(BaseIntegrationConfig):
    default_source: str = DEFAULT_SOURCE

    @classmethod
    def from_env(cls, **overrides: object) -> SlashCommandIntegrationConfig:
        payload = {
            "default_source": os.environ.get(ENV_SLASH_COMMAND_DEFAULT_SOURCE, DEFAULT_SOURCE).strip()
            or DEFAULT_SOURCE,
        }
        payload.update(overrides)
        return cls.model_validate(payload)
