# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from intergrax.applications.contracts.settings import EnvReader, IntergraxApplicationSettingsBase
from intergrax.fastapi_core.config import ApiEnvironment


@dataclass(frozen=True, kw_only=True)
class LabApplicationSettings(IntergraxApplicationSettingsBase):
    """Environment for the universal lab application (Tier-3)."""

    env_prefix: ClassVar[str] = "LAB_"
    route_prefix: str = "/v1/lab"
    include_mock_agents: bool = True
    include_echo: bool = True
    include_signoff_probe: bool = True
    include_research: bool = False
    include_problem_radar: bool = False
    harness: bool = False
    otel_enabled: bool = True
    strict_harness: bool = False
    adaptive_observe_enabled: bool = True
    observability_grafana_stack: bool = False
    adaptive_feature_flag_slug: str | None = None
    secrets_backend_slug: str | None = None
    enable_llm_guardrails: bool = False
    tool_invocation_mode: str = "single_pass"

    @property
    def requires_harness_api_key(self) -> bool:
        """Staging/production and strict harness profiles must configure API key (W-OPS.7)."""
        return self.strict_harness or self.environment != ApiEnvironment.DEV

    # ------------------------------------------------------------------
    # Application-specific settings
    # Add your own env-backed fields here.
    # ------------------------------------------------------------------

    @classmethod
    def _load_app_env(cls, env: EnvReader) -> dict[str, object]:
        feature_flag_raw = env.optional_str("ADAPTIVE_FEATURE_FLAG")
        secrets_backend = env.optional_str("SECRETS_BACKEND")
        return {
            "include_mock_agents": env.bool("INCLUDE_MOCK_AGENTS", default=True),
            "include_echo": env.bool("INCLUDE_ECHO", default=True),
            "include_signoff_probe": env.bool("INCLUDE_SIGNOFF_PROBE", default=True),
            "include_research": env.bool("INCLUDE_RESEARCH", default=False),
            "include_problem_radar": env.bool("INCLUDE_PROBLEM_RADAR", default=False),
            "harness": env.bool("HARNESS", default=False),
            "otel_enabled": env.bool("OTEL_ENABLED", default=True),
            "strict_harness": env.bool("STRICT_HARNESS", default=False),
            "adaptive_observe_enabled": env.bool("ADAPTIVE_OBSERVE", default=True),
            "observability_grafana_stack": env.bool("OBSERVABILITY_GRAFANA_STACK", default=False),
            "adaptive_feature_flag_slug": feature_flag_raw.lower() if feature_flag_raw else None,
            "secrets_backend_slug": secrets_backend.lower() if secrets_backend else None,
            "enable_llm_guardrails": env.bool("ENABLE_LLM_GUARDRAILS", default=False),
            "tool_invocation_mode": env.str("TOOL_INVOCATION_MODE", default="single_pass").lower()
            or "single_pass",
        }
