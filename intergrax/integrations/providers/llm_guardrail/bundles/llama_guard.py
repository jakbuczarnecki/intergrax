# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailContext,
    GuardrailScanResult,
    LlmGuardrailBackend,
)
from intergrax.integrations.providers.llm_guardrail._vendor_opens import http_guardrail_scan
from intergrax.integrations.providers.llm_guardrail.bundles._base import BaseGuardrailAdapter
from pydantic import BaseModel, ConfigDict


class LlamaGuardrailOptions(BaseModel):
    """Llama Guard provider-owned configuration."""

    model_config = ConfigDict(extra="forbid")

    inference_slug: str | None = None


def _parse_llama_options(
    provider_options: Mapping[str, Any] | None,
) -> LlamaGuardrailOptions:
    if provider_options:
        return LlamaGuardrailOptions.model_validate(provider_options)
    return LlamaGuardrailOptions()


def _inference_env_prefix(options: LlamaGuardrailOptions) -> str:
    slug = options.inference_slug or "llama_guard"
    return f"INTERGRAX_{slug.upper().replace('-', '_')}"


class LlamaGuardAdapter(BaseGuardrailAdapter):
    def __init__(
        self,
        *,
        slug: str,
        llama_options: LlamaGuardrailOptions,
        allow_pattern_fallback: bool = False,
    ) -> None:
        super().__init__(slug=slug, allow_pattern_fallback=allow_pattern_fallback)
        self._llama_options = llama_options

    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        vendor = self._classify(text, mode="input")
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
        vendor = self._classify(text, mode="output")
        if vendor is not None:
            return vendor
        return self._pattern_fallback(text, mode="output")

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
        prefix = _inference_env_prefix(self._llama_options)
        return http_guardrail_scan(
            slug=self._slug,
            text=text,
            mode=mode,
            env_prefix=prefix,
            default_path="/v1/classify",
        )


def create_llama_guard_backend(
    *,
    provider_options: Mapping[str, Any] | None = None,
    allow_pattern_fallback: bool = False,
) -> LlmGuardrailBackend:
    return LlamaGuardAdapter(
        slug="llama_guard",
        llama_options=_parse_llama_options(provider_options),
        allow_pattern_fallback=allow_pattern_fallback,
    )
