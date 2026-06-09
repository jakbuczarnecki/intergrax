# © Artur Czarnecki. All rights reserved.

"""Compose multiple guardrail backends (primary + semantic scanners)."""

from __future__ import annotations

from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailContext,
    GuardrailRiskLevel,
    GuardrailScanResult,
    LlmGuardrailBackend,
)


class ChainedGuardrailBackend:
    """Run scan methods across backends in order; first deny wins."""

    def __init__(self, *backends: LlmGuardrailBackend) -> None:
        if not backends:
            raise ValueError("ChainedGuardrailBackend requires at least one backend")
        self._backends = backends

    @property
    def slug(self) -> str:
        return "+".join(backend.slug for backend in self._backends)

    def scan_input(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailScanResult:
        sanitized = text
        audit: list[dict[str, object]] = []
        highest = GuardrailRiskLevel.LOW
        for backend in self._backends:
            result = backend.scan_input(sanitized, context=context)
            audit.append({"slug": backend.slug, **result.audit_payload})
            if _risk_rank(result.risk_level) > _risk_rank(highest):
                highest = result.risk_level
            if not result.allowed:
                return result.model_copy(
                    update={
                        "audit_payload": {"chain": audit, "denied_by": backend.slug},
                    },
                )
            if result.sanitized_text:
                sanitized = result.sanitized_text
        return GuardrailScanResult(
            allowed=True,
            risk_level=highest,
            sanitized_text=sanitized if sanitized != text else None,
            detail=f"chained input pass ({self.slug})",
            audit_payload={"chain": audit},
        )

    def scan_output(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
        prompt: str | None = None,
    ) -> GuardrailScanResult:
        sanitized = text
        audit: list[dict[str, object]] = []
        highest = GuardrailRiskLevel.LOW
        for backend in self._backends:
            result = backend.scan_output(sanitized, context=context, prompt=prompt)
            audit.append({"slug": backend.slug, **result.audit_payload})
            if _risk_rank(result.risk_level) > _risk_rank(highest):
                highest = result.risk_level
            if not result.allowed:
                return result.model_copy(
                    update={
                        "audit_payload": {"chain": audit, "denied_by": backend.slug},
                    },
                )
            if result.sanitized_text:
                sanitized = result.sanitized_text
        return GuardrailScanResult(
            allowed=True,
            risk_level=highest,
            sanitized_text=sanitized if sanitized != text else None,
            detail=f"chained output pass ({self.slug})",
            audit_payload={"chain": audit},
        )

    def scan_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, str],
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailScanResult:
        for backend in self._backends:
            result = backend.scan_tool_call(tool_name, arguments, context=context)
            if not result.allowed:
                return result
        return GuardrailScanResult(allowed=True, detail=f"chained tool pass ({self.slug})")

    def health_check(self) -> bool:
        return all(backend.health_check() for backend in self._backends)


def _risk_rank(level: GuardrailRiskLevel) -> int:
    order = {
        GuardrailRiskLevel.LOW: 0,
        GuardrailRiskLevel.MEDIUM: 1,
        GuardrailRiskLevel.HIGH: 2,
        GuardrailRiskLevel.CRITICAL: 3,
    }
    return order[level]
