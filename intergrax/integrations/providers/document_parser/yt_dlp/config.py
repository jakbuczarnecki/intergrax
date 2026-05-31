# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class YtDlpIntegrationConfig(BaseModel):
    out_dir: Path = Field(default=Path("./audio_downloads"))
    audio_format: str = "mp3"

    @classmethod
    def from_env(cls, **overrides: object) -> YtDlpIntegrationConfig:
        data = cls().model_dump()
        data.update(overrides)
        return cls(**data)
