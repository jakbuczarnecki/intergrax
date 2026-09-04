# © Artur Czarnecki. All rights reserved.

"""Shared typed Integration boundary over existing LlmGuardrailBackend runtimes."""

from __future__ import annotations

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailContext,
    GuardrailScanResult,
    LlmGuardrailBackend,
)
from intergrax.runtime.integrations.categories.ai import LlmGuardrailIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig


class GuardrailTypedIntegration(LlmGuardrailIntegrationContract):
    """Thin typed Integration delegating scan operations to a runtime backend."""

    config: CategoryIntegrationConfig = CategoryIntegrationConfig()
    _backend: LlmGuardrailBackend | None = PrivateAttr(default=None)

    @property
    def slug(self) -> str:
        return self.provider_id

    def scan_input(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailScanResult:
        return self._require_backend().scan_input(text, context=context)

    def scan_output(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
        prompt: str | None = None,
    ) -> GuardrailScanResult:
        return self._require_backend().scan_output(text, context=context, prompt=prompt)

    def scan_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, str],
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailScanResult:
        return self._require_backend().scan_tool_call(
            tool_name,
            arguments,
            context=context,
        )

    def health_check(self) -> bool:
        backend = self._backend
        if backend is None:
            return False
        return backend.health_check()

    @classmethod
    def from_backend(
        cls,
        backend: LlmGuardrailBackend,
        *,
        provider_id: str,
        display_name: str,
        enabled: bool = False,
        config: CategoryIntegrationConfig | None = None,
    ) -> GuardrailTypedIntegration:
        integration = cls.for_provider(
            provider_id=provider_id,
            display_name=display_name,
            config=config or CategoryIntegrationConfig(enabled=enabled),
        )
        integration._backend = backend
        return integration

    def _require_backend(self) -> LlmGuardrailBackend:
        if self._backend is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime backend for guardrail operations",
            )
        return self._backend


__all__ = ["GuardrailTypedIntegration"]
