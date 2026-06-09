# © Artur Czarnecki. All rights reserved.

"""Shared guardrail adapter base (M.12 bundles)."""

from __future__ import annotations

from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailBackendOptions,
    GuardrailContext,
    GuardrailScanResult,
)


class BaseGuardrailAdapter:
    """Shared defaults for catalog guardrail backends."""

    def __init__(
        self,
        *,
        slug: str,
        options: GuardrailBackendOptions | None = None,
    ) -> None:
        self._slug = slug
        self._options = options or GuardrailBackendOptions()

    @property
    def slug(self) -> str:
        return self._slug

    def scan_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, str],
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailScanResult:
        joined = " ".join(arguments.values())
        if "ignore previous instructions" in joined.lower():
            return GuardrailScanResult(
                allowed=False,
                categories=("tool_injection",),
                matched_rules=("tool_injection:argument",),
                detail=f"{self._slug} tool argument blocked",
                audit_payload={"tool_name": tool_name, "engine": self._slug},
            )
        return GuardrailScanResult(allowed=True, sanitized_text=joined)

    def health_check(self) -> bool:
        return True
