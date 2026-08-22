# © Artur Czarnecki. All rights reserved.

"""LLM guardrail backend contract (M-P12-CAT.1)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class GuardrailRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailBackendOptions(BaseModel):
    """Provider-neutral guardrail wiring metadata (chaining only).

    Provider-specific configuration lives in ``IntegrationProfile.options[slug]``
    and is validated by provider-owned option models at factory time.
    """

    model_config = ConfigDict(extra="forbid")


class GuardrailContext(BaseModel):
    """Runtime metadata passed to vendor scanners at hook time."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = ""
    run_id: str = ""
    agent_id: str = ""
    step_id: str = ""
    hook: str = ""


class GuardrailScanResult(BaseModel):
    """Normalized guardrail scan output."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool = True
    risk_level: GuardrailRiskLevel = GuardrailRiskLevel.LOW
    categories: tuple[str, ...] = ()
    matched_rules: tuple[str, ...] = ()
    sanitized_text: str | None = None
    detail: str = ""
    audit_payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def redacted_text(self) -> str | None:
        return self.sanitized_text

    @property
    def blocked_categories(self) -> tuple[str, ...]:
        return self.categories


@runtime_checkable
class LlmGuardrailBackend(Protocol):
    """Vendor guardrail scanner — no SDK in Tier-2 agents."""

    @property
    def slug(self) -> str: ...

    def scan_input(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailScanResult: ...

    def scan_output(
        self,
        text: str,
        *,
        context: GuardrailContext | None = None,
        prompt: str | None = None,
    ) -> GuardrailScanResult: ...

    def scan_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, str],
        *,
        context: GuardrailContext | None = None,
    ) -> GuardrailScanResult: ...

    def health_check(self) -> bool: ...
