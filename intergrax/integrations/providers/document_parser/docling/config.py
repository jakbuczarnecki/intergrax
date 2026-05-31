# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, Field


class DoclingMode(str, Enum):
    NONE = "none"
    LOCAL = "local"
    SERVER = "server"


class DoclingIntegrationConfig(BaseModel):
    mode: DoclingMode = Field(default_factory=lambda: DoclingIntegrationConfig._read_mode())
    simple_pdf_mode: bool = True
    server_url: str = Field(
        default_factory=lambda: os.getenv("INTERGRAX_DOCLING_SERVER_URL", "http://localhost:8000")
    )
    server_path: str = Field(
        default_factory=lambda: os.getenv("INTERGRAX_DOCLING_SERVER_PATH", "/parse")
    )
    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("INTERGRAX_DOCLING_TIMEOUT", "120"))
    )

    @staticmethod
    def _read_mode() -> DoclingMode:
        raw = os.getenv("INTERGRAX_DOCLING_MODE", "local").strip().lower()
        try:
            return DoclingMode(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid INTERGRAX_DOCLING_MODE='{raw}'. Allowed: none, local, server."
            ) from exc

    @classmethod
    def from_env(cls, **overrides: object) -> DoclingIntegrationConfig:
        data = cls().model_dump()
        data.update(overrides)
        if "mode" in overrides and isinstance(overrides["mode"], str):
            data["mode"] = DoclingMode(str(overrides["mode"]).lower())
        return cls(**data)
