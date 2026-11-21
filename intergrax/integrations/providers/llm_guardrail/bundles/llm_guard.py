# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.contracts.llm_guardrail import GuardrailContext, GuardrailScanResult, LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail._vendor_opens import (
    llm_guard_scan_input,
    llm_guard_scan_output,
)
from intergrax.integrations.providers.llm_guardrail.bundles._base import BaseGuardrailAdapter


class LlmGuardAdapter(BaseGuardrailAdapter):
    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = llm_guard_scan_input(text)
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
        vendor = llm_guard_scan_output(text, prompt=prompt or "")
        if vendor is not None:
            return vendor
        return self._pattern_fallback(text, mode="output")


def create_llm_guard_backend(*, allow_pattern_fallback: bool = False) -> LlmGuardrailBackend:
    return LlmGuardAdapter(slug="llm_guard", allow_pattern_fallback=allow_pattern_fallback)
