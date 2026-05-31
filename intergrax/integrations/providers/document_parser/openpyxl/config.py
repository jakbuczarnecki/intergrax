# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

EXTRACTION_STRATEGY = Literal["rows", "sheets", "markdown"]


class OpenpyxlIntegrationConfig(BaseModel):
    mode: EXTRACTION_STRATEGY = "rows"
    header: int = 0
    sheet: str | int | None = None
    na_filter: bool = True
    max_rows_per_sheet: int | None = None
    encoding: str | None = None
    delimiter: str | None = None

    @classmethod
    def from_env(cls, **overrides: object) -> OpenpyxlIntegrationConfig:
        data = cls().model_dump()
        data.update(overrides)
        return cls(**data)
