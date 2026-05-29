# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared integration configuration helpers (Phase M.1)."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field


ENV_INTEGRATION_PREFIX = "INTERGRAX_INTEGRATION_"


class BaseIntegrationConfig(BaseModel):
    """Common provider settings — extended per slug in ``providers/<slug>/config.py``."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    timeout_seconds: Optional[float] = None
    key_prefix: str = "intergrax"


def env_key_for_category(category: str) -> str:
    normalized = category.strip().upper().replace("-", "_")
    return f"{ENV_INTEGRATION_PREFIX}{normalized}"


def read_integration_slug_from_env(category: str) -> Optional[str]:
    key = env_key_for_category(category)
    raw = os.environ.get(key, "").strip()
    return raw or None


def merge_config(
    base: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if base:
        merged.update(dict(base))
    if overrides:
        merged.update(dict(overrides))
    return merged


class ProviderConfig(BaseModel):
    """Optional per-slug config bag passed into catalog factories."""

    model_config = ConfigDict(extra="allow")

    slug: str
    options: dict[str, Any] = Field(default_factory=dict)
