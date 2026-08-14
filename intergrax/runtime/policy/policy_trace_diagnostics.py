# © Artur Czarnecki. All rights reserved.

"""Declarative policy trace diagnostics (BLOCK B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class DeclarativePolicyEvaluationDiagV1(DiagnosticPayload):
    tool_id: str
    action: str
    matched_rule_ids: tuple[str, ...]
    enforcement_mode: str
    enforced: bool
    would_deny: bool
    reasons: tuple[str, ...]
    unknown_handler_ids: tuple[str, ...]
    provenance_digest: str | None = None

    def redact(self) -> DeclarativePolicyEvaluationDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.policy.declarative_evaluation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "action": self.action,
            "matched_rule_ids": list(self.matched_rule_ids),
            "enforcement_mode": self.enforcement_mode,
            "enforced": self.enforced,
            "would_deny": self.would_deny,
            "reasons": list(self.reasons),
            "unknown_handler_ids": list(self.unknown_handler_ids),
            "provenance_digest": self.provenance_digest,
        }
