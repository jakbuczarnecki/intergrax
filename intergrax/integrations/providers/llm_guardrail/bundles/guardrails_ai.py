# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.contracts.llm_guardrail import GuardrailContext, GuardrailScanResult, LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail._vendor_opens import guardrails_ai_validate
from intergrax.integrations.providers.llm_guardrail.bundles._base import BaseGuardrailAdapter


class GuardrailsAiAdapter(BaseGuardrailAdapter):
    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = guardrails_ai_validate(text)
        if vendor is not None:
            return vendor
        return self._pattern_fallback(text, mode="input")

    def scan_output(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
        prompt: str | None = None,
    ) -> GuardrailScanResult:
        vendor = guardrails_ai_validate(text)
        if vendor is not None:
            return vendor
        return self._pattern_fallback(text, mode="output")


def create_guardrails_ai_backend(*, allow_pattern_fallback: bool = False) -> LlmGuardrailBackend:
    return GuardrailsAiAdapter(slug="guardrails_ai", allow_pattern_fallback=allow_pattern_fallback)
