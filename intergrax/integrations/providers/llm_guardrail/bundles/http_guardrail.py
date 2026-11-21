# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.contracts.llm_guardrail import GuardrailContext, GuardrailScanResult, LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail._vendor_opens import http_guardrail_scan
from intergrax.integrations.providers.llm_guardrail.bundles._base import BaseGuardrailAdapter


class HttpGuardrailAdapter(BaseGuardrailAdapter):
    def __init__(
        self,
        *,
        slug: str,
        env_prefix: str,
        path: str,
        allow_pattern_fallback: bool = False,
    ) -> None:
        super().__init__(slug=slug, allow_pattern_fallback=allow_pattern_fallback)
        self._env_prefix = env_prefix
        self._path = path

    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = http_guardrail_scan(
            slug=self._slug,
            text=text,
            mode="input",
            env_prefix=self._env_prefix,
            default_path=self._path,
        )
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
        vendor = http_guardrail_scan(
            slug=self._slug,
            text=text,
            mode="output",
            env_prefix=self._env_prefix,
            default_path=self._path,
        )
        if vendor is not None:
            return vendor
        return self._pattern_fallback(text, mode="output")


def create_openguardrails_backend(*, allow_pattern_fallback: bool = False) -> LlmGuardrailBackend:
    return HttpGuardrailAdapter(
        slug="openguardrails",
        env_prefix="INTERGRAX_OPENGUARDRAILS",
        path="/v1/guardrails/check",
        allow_pattern_fallback=allow_pattern_fallback,
    )


def create_lakera_backend(*, allow_pattern_fallback: bool = False) -> LlmGuardrailBackend:
    return HttpGuardrailAdapter(
        slug="lakera",
        env_prefix="INTERGRAX_LAKERA",
        path="/v1/guard",
        allow_pattern_fallback=allow_pattern_fallback,
    )


def create_azure_content_safety_backend(*, allow_pattern_fallback: bool = False) -> LlmGuardrailBackend:
    return HttpGuardrailAdapter(
        slug="azure_content_safety",
        env_prefix="INTERGRAX_AZURE_CONTENT_SAFETY",
        path="/contentsafety/text:analyze",
        allow_pattern_fallback=allow_pattern_fallback,
    )
