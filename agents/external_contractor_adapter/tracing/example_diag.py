# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload
from intergrax.runtime.observability.extension_sdk import agent_diagnostic_schema_id


@dataclass(frozen=True)
class CustomCheckDiagV1(DiagnosticPayload):
    """Example agent diagnostic payload — replace with domain semantics."""

    check_name: str
    passed: bool
    detail: str = ""

    @classmethod
    def schema_id(cls) -> str:
        return agent_diagnostic_schema_id("external_contractor_adapter", "custom_check")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "detail": self.detail,
        }

    def redact(self) -> CustomCheckDiagV1:
        return self
