# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailBackendOptions,
    GuardrailContext,
    GuardrailScanResult,
    LlmGuardrailBackend,
)
from intergrax.integrations.providers.llm_guardrail._pattern_scanner import scan_patterns
from intergrax.integrations.providers.llm_guardrail._vendor_opens import http_guardrail_scan
from intergrax.integrations.providers.llm_guardrail.bundles._base import BaseGuardrailAdapter


def _inference_env_prefix(options: GuardrailBackendOptions) -> str:
    slug = options.inference_slug or "llama_guard"
    return f"INTERGRAX_{slug.upper().replace('-', '_')}"


class LlamaGuardAdapter(BaseGuardrailAdapter):
    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = self._classify(text, mode="input")
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
        vendor = self._classify(text, mode="output")
        if vendor is not None:
            return vendor
        return scan_patterns(text, mode="output", slug=self._slug)

    def _classify(self, text: str, *, mode: str) -> GuardrailScanResult | None:
        dedicated = os.environ.get("INTERGRAX_LLAMA_GUARD_INFERENCE_URL", "").strip()
        if dedicated:
            return http_guardrail_scan(
                slug=self._slug,
                text=text,
                mode=mode,
                env_prefix="INTERGRAX_LLAMA_GUARD",
                default_path="/v1/classify",
            )
        prefix = _inference_env_prefix(self._options)
        return http_guardrail_scan(
            slug=self._slug,
            text=text,
            mode=mode,
            env_prefix=prefix,
            default_path="/v1/classify",
        )


def create_llama_guard_backend(*, options: GuardrailBackendOptions | None = None) -> LlmGuardrailBackend:
    return LlamaGuardAdapter(slug="llama_guard", options=options)
