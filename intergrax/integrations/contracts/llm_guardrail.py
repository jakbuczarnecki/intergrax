# © Artur Czarnecki. All rights reserved.

"""LLM guardrail backend contract (M-P12-CAT.1)."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class GuardrailScanResult(BaseModel):
    """Normalized guardrail scan output."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool = True
    blocked_categories: tuple[str, ...] = ()
    redacted_text: str | None = None
    detail: str = ""


class LlmGuardrailBackend(Protocol):
    """Vendor guardrail scanner — no SDK in Tier-2 agents."""

    @property
    def slug(self) -> str: ...

    def scan_input(self, text: str) -> GuardrailScanResult: ...

    def scan_output(self, text: str) -> GuardrailScanResult: ...
