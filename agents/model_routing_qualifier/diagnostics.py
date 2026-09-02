# © Artur Czarnecki. All rights reserved.

"""Typed model-routing evidence payloads for AgentStepRecord.diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

_MODEL_ROUTING_SUMMARY_V1 = "lkw.model_routing_summary.v1"


@dataclass(frozen=True)
class ModelRoutingSummaryDiagnostic(DiagnosticPayload):
    used: bool
    reason: str
    routing_context_summary: str | None = None
    candidate_profile_refs: tuple[str, ...] = ()
    expected_profile_ref: str | None = None
    selected_profile_ref: str | None = None
    matched_rule_id: str | None = None
    routing_reason: str | None = None
    actual_adapter_provider: str | None = None
    actual_adapter_model: str | None = None
    invocation_status: str | None = None
    raw_model_output: str | None = None

    @classmethod
    def schema_id(cls) -> str:
        return _MODEL_ROUTING_SUMMARY_V1

    def redact(self) -> ModelRoutingSummaryDiagnostic:
        return ModelRoutingSummaryDiagnostic(
            used=self.used,
            reason=self.reason,
            routing_context_summary=self.routing_context_summary,
            candidate_profile_refs=self.candidate_profile_refs,
            expected_profile_ref=self.expected_profile_ref,
            selected_profile_ref=self.selected_profile_ref,
            matched_rule_id=self.matched_rule_id,
            routing_reason=self.routing_reason,
            actual_adapter_provider=self.actual_adapter_provider,
            actual_adapter_model=self.actual_adapter_model,
            invocation_status=self.invocation_status,
            raw_model_output=self.raw_model_output,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "used": self.used,
            "reason": self.reason,
            "routing_context_summary": self.routing_context_summary,
            "candidate_profile_refs": list(self.candidate_profile_refs),
            "expected_profile_ref": self.expected_profile_ref,
            "selected_profile_ref": self.selected_profile_ref,
            "matched_rule_id": self.matched_rule_id,
            "routing_reason": self.routing_reason,
            "actual_adapter_provider": self.actual_adapter_provider,
            "actual_adapter_model": self.actual_adapter_model,
            "invocation_status": self.invocation_status,
            "raw_model_output": self.raw_model_output,
            "ops": "model_routing_summary",
        }


def model_routing_diagnostic_from_output(output: dict[str, object]) -> ModelRoutingSummaryDiagnostic:
    summary = output.get("model_routing_summary")
    if not isinstance(summary, dict):
        return ModelRoutingSummaryDiagnostic(used=False, reason="summary_missing")
    raw_candidates = summary.get("candidate_profile_refs")
    candidates: tuple[str, ...] = ()
    if isinstance(raw_candidates, list):
        candidates = tuple(str(item) for item in raw_candidates if isinstance(item, str))
    return ModelRoutingSummaryDiagnostic(
        used=bool(summary.get("used")),
        reason=str(summary.get("reason") or "unknown"),
        routing_context_summary=(
            str(summary["routing_context_summary"])
            if isinstance(summary.get("routing_context_summary"), str)
            else None
        ),
        candidate_profile_refs=candidates,
        expected_profile_ref=(
            str(summary["expected_profile_ref"])
            if isinstance(summary.get("expected_profile_ref"), str)
            else None
        ),
        selected_profile_ref=(
            str(summary["selected_profile_ref"])
            if isinstance(summary.get("selected_profile_ref"), str)
            else None
        ),
        matched_rule_id=(
            str(summary["matched_rule_id"]) if isinstance(summary.get("matched_rule_id"), str) else None
        ),
        routing_reason=(
            str(summary["routing_reason"]) if isinstance(summary.get("routing_reason"), str) else None
        ),
        actual_adapter_provider=(
            str(summary["actual_adapter_provider"])
            if isinstance(summary.get("actual_adapter_provider"), str)
            else None
        ),
        actual_adapter_model=(
            str(summary["actual_adapter_model"])
            if isinstance(summary.get("actual_adapter_model"), str)
            else None
        ),
        invocation_status=(
            str(summary["invocation_status"])
            if isinstance(summary.get("invocation_status"), str)
            else None
        ),
        raw_model_output=(
            str(summary["raw_model_output"])
            if isinstance(summary.get("raw_model_output"), str)
            else None
        ),
    )


__all__ = ["ModelRoutingSummaryDiagnostic", "model_routing_diagnostic_from_output"]
