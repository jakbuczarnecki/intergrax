# © Artur Czarnecki. All rights reserved.

"""Harness stub LLM guardrail backends (M-P12.*)."""

from __future__ import annotations

from intergrax.integrations.contracts.llm_guardrail import GuardrailContext, GuardrailScanResult, LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail._pattern_scanner import scan_patterns


class StubLlmGuardrailBackend:
    """Pass-through guardrail for harness wiring and CI."""

    def __init__(self, *, slug: str) -> None:
        self._slug = slug

    @property
    def slug(self) -> str:
        return self._slug

    def scan_input(self, text: str, *, context: GuardrailContext | None = None) -> GuardrailScanResult:
        return scan_patterns(text, mode="input", slug=self._slug)

    def scan_output(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
        prompt: str | None = None,
    ) -> GuardrailScanResult:
        _ = prompt
        return scan_patterns(text, mode="output", slug=self._slug)

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
            )
        return GuardrailScanResult(allowed=True, sanitized_text=joined)

    def health_check(self) -> bool:
        return True


def create_stub_guardrail(slug: str) -> LlmGuardrailBackend:
    return StubLlmGuardrailBackend(slug=slug)
