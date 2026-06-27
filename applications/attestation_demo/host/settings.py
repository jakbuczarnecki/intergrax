# © Artur Czarnecki. All rights reserved.

"""Environment for attestation_demo (partner PoC lab profile)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from intergrax.applications.contracts.settings import EnvReader, IntergraxApplicationSettingsBase


@dataclass(frozen=True, kw_only=True)
class AttestationDemoSettings(IntergraxApplicationSettingsBase):
    """Environment-backed settings for the attestation demo host."""

    env_prefix: ClassVar[str] = "ATTESTATION_DEMO_"
    route_prefix: str = "/v1/attestation_demo"
    backend_port: int = 8097
    include_interaction_routes: bool = False
    include_scheduler: bool = False
    include_queue_worker: bool = False

    # ------------------------------------------------------------------
    # Application-specific settings
    # Add your own env-backed fields here.
    # ------------------------------------------------------------------

    @classmethod
    def _load_app_env(cls, env: EnvReader) -> dict[str, object]:
        return {}
