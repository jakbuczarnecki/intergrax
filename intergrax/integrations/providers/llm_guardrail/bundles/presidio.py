# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.contracts.llm_guardrail import GuardrailBackendOptions
from intergrax.integrations.contracts.llm_guardrail import GuardrailContext, GuardrailScanResult, LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail._pattern_scanner import scan_patterns
from intergrax.integrations.providers.llm_guardrail._vendor_opens import presidio_scan_text
from intergrax.integrations.providers.llm_guardrail.bundles._base import BaseGuardrailAdapter


class PresidioAdapter(BaseGuardrailAdapter):
    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = presidio_scan_text(text)
        if vendor is not None:
            return vendor
        return scan_patterns(text, mode="input", slug=self._slug)

    def scan_output(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
        prompt: str | None = None,
    ) -> GuardrailScanResult:
        return self.scan_input(text, context=context)


def create_presidio_backend(*, options: GuardrailBackendOptions | None = None) -> LlmGuardrailBackend:
    return PresidioAdapter(slug="presidio", options=options)
