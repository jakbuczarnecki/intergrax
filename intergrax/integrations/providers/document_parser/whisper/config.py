# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from intergrax.globals.settings import GLOBAL_SETTINGS


class WhisperIntegrationConfig(BaseModel):
    model: str = "medium"
    language: str = Field(default_factory=lambda: GLOBAL_SETTINGS.default_language)
    translate: bool = True
    out_dir: Path = Field(default=Path("./audio_downloads"))
    audio_format: str = "mp3"

    @classmethod
    def from_env(cls, **overrides: object) -> WhisperIntegrationConfig:
        data = cls().model_dump()
        data.update(overrides)
        return cls(**data)
