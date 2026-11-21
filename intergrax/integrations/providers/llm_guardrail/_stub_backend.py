# © Artur Czarnecki. All rights reserved.

"""Harness stub LLM guardrail backends (M-P12.*)."""

from __future__ import annotations

from intergrax.integrations.contracts.llm_guardrail import GuardrailContext, GuardrailScanResult, LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail._pattern_policy import pattern_fallback_scan


def _stamp_stub_provenance(result: GuardrailScanResult, *, slug: str) -> GuardrailScanResult:
    audit = dict(result.audit_payload)
    audit["provenance"] = "stub"
    audit["requested_slug"] = slug
    return result.model_copy(update={"audit_payload": audit})


class StubLlmGuardrailBackend:
    """Pass-through guardrail for harness wiring and CI."""

    def __init__(self, *, slug: str) -> None:
        self._slug = slug

    @property
    def slug(self) -> str:
        return self._slug

    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        result = pattern_fallback_scan(text, mode="input", slug=self._slug, allow_pattern_fallback=True)
        return _stamp_stub_provenance(result, slug=self._slug)

    def scan_output(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
        prompt: str | None = None,
    ) -> GuardrailScanResult:
        _ = prompt
        result = pattern_fallback_scan(text, mode="output", slug=self._slug, allow_pattern_fallback=True)
        return _stamp_stub_provenance(result, slug=self._slug)

    def scan_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, str],
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailScanResult:
        _ = context
        joined = " ".join(arguments.values())
        if "BLOCK_TOOL" in joined:
            return GuardrailScanResult(
                allowed=False,
                categories=("test_block",),
                detail="stub tool block",
                audit_payload={"provenance": "stub", "requested_slug": self._slug, "tool_name": tool_name},
            )
        return GuardrailScanResult(
            allowed=True,
            sanitized_text=joined,
            audit_payload={"provenance": "stub", "requested_slug": self._slug, "tool_name": tool_name},
        )

    def health_check(self) -> bool:
        return True


def create_stub_guardrail(slug: str) -> LlmGuardrailBackend:
    return StubLlmGuardrailBackend(slug=slug)
