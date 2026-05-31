# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EXTRACTION_STRATEGY = Literal["auto", "fulltext", "paragraphs", "headings"]


class PythonDocxIntegrationConfig(BaseModel):
    strategy: EXTRACTION_STRATEGY = "auto"

    @classmethod
    def from_env(cls, **overrides: object) -> PythonDocxIntegrationConfig:
        data = cls().model_dump()
        data.update(overrides)
        return cls(**data)
