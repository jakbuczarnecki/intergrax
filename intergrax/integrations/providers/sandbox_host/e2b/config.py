# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""E2B sandbox host provider configuration."""

from __future__ import annotations


from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig, _env
from intergrax.integrations.contracts.base import IntegrationConfigurationError


class E2bSandboxHostConfig(HttpIntegrationConfig):
    """Typed configuration for the E2B sandbox host adapter."""

    template_id: str = "base"
    sandbox_timeout_seconds: int = 300

    def resolved_api_key(self) -> str:
        api_key = self.api_key.strip() or self.token.strip() or _env("E2B_API_KEY")
        if not api_key:
            raise IntegrationConfigurationError(
                "E2B sandbox host requires an API key (INTERGRAX_E2B_API_KEY or E2B_API_KEY)",
            )
        return api_key

    def resolved_base_url(self) -> str:
        return (self.base_url.strip() or "https://api.e2b.app").rstrip("/")

    def resolved_template_id(self) -> str:
        template_id = self.template_id.strip() or _env("INTERGRAX_E2B_TEMPLATE_ID") or _env("E2B_TEMPLATE_ID")
        if not template_id:
            raise IntegrationConfigurationError(
                "E2B sandbox host requires a template id (INTERGRAX_E2B_TEMPLATE_ID)",
            )
        return template_id

    @classmethod
    def from_env(cls, prefix: str = "INTERGRAX_E2B", **overrides: object) -> E2bSandboxHostConfig:
        payload = HttpIntegrationConfig.from_env(prefix, **overrides).model_dump()
        template_id = _env(f"{prefix}_TEMPLATE_ID") or _env("E2B_TEMPLATE_ID")
        if template_id:
            payload["template_id"] = template_id
        timeout_raw = _env(f"{prefix}_SANDBOX_TIMEOUT")
        if timeout_raw:
            payload["sandbox_timeout_seconds"] = int(timeout_raw)
        payload.update(overrides)
        return cls.model_validate(payload)
