# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class PymupdfIntegrationConfig(BaseModel):
    enable_ocr: bool = False
    ocr_lang: str = "eng"
    ocr_dpi: int = 200
    ocr_psm: int | None = None
    ocr_oem: int | None = None
    ocr_max_pages: int | None = None

    @classmethod
    def from_env(cls, **overrides: object) -> PymupdfIntegrationConfig:
        data = cls().model_dump()
        data.update(overrides)
        return cls(**data)
