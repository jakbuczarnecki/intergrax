# © Artur Czarnecki. All rights reserved.

"""Harness stub LLM guardrail backends (M-P12.*)."""

from __future__ import annotations

from intergrax.integrations.contracts.llm_guardrail import GuardrailScanResult, LlmGuardrailBackend


class StubLlmGuardrailBackend:
    """Pass-through guardrail for harness wiring and CI."""

    def __init__(self, *, slug: str) -> None:
        self._slug = slug

    @property
    def slug(self) -> str:
        return self._slug

    def scan_input(self, text: str) -> GuardrailScanResult:
        if "BLOCK_INPUT" in text:
            return GuardrailScanResult(
                allowed=False,
                blocked_categories=("test_block",),
                detail="stub input block",
            )
        return GuardrailScanResult(allowed=True, redacted_text=text)

    def scan_output(self, text: str) -> GuardrailScanResult:
        if "BLOCK_OUTPUT" in text:
            return GuardrailScanResult(
                allowed=False,
                blocked_categories=("test_block",),
                detail="stub output block",
            )
        return GuardrailScanResult(allowed=True, redacted_text=text)


def create_stub_guardrail(slug: str) -> LlmGuardrailBackend:
    return StubLlmGuardrailBackend(slug=slug)
