# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from intergrax.applications.contracts.settings import EnvReader, IntergraxApplicationSettingsBase


@dataclass(frozen=True, kw_only=True)
class PocTemplateApplicationSettings(IntergraxApplicationSettingsBase):
    """Environment for poc_template_application (scaffolded lab profile)."""

    env_prefix: ClassVar[str] = "POC_TEMPLATE_"
    route_prefix: str = "/v1/poc_template"
    backend_port: int = 8095

    # ------------------------------------------------------------------
    # Application-specific settings
    # Add your own env-backed fields here.
    # ------------------------------------------------------------------

    @classmethod
    def _load_app_env(cls, env: EnvReader) -> dict[str, object]:
        return {}
