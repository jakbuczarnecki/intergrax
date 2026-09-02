# © Artur Czarnecki. All rights reserved.

"""Typed model-routing evidence payloads for AgentStepRecord.diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from intergrax.knowledge.contracts.validation import JsonObject
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
    production_evaluation_selected_profile: str | None = None
    matched_rule_id: str | None = None
    routing_reason: str | None = None
    policy_route_hint: str | None = None
    actual_adapter_provider: str | None = None
    actual_adapter_model: str | None = None
    actual_inner_adapter_provider: str | None = None
    actual_inner_adapter_model: str | None = None
    invocation_status: str | None = None
    invocation_failure_kind: str | None = None
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
            production_evaluation_selected_profile=self.production_evaluation_selected_profile,
            matched_rule_id=self.matched_rule_id,
            routing_reason=self.routing_reason,
            policy_route_hint=self.policy_route_hint,
            actual_adapter_provider=self.actual_adapter_provider,
            actual_adapter_model=self.actual_adapter_model,
            actual_inner_adapter_provider=self.actual_inner_adapter_provider,
            actual_inner_adapter_model=self.actual_inner_adapter_model,
            invocation_status=self.invocation_status,
            invocation_failure_kind=self.invocation_failure_kind,
            raw_model_output=self.raw_model_output,
        )

    def to_dict(self) -> JsonObject:
        return {
            "used": self.used,
            "reason": self.reason,
            "routing_context_summary": self.routing_context_summary,
            "candidate_profile_refs": list(self.candidate_profile_refs),
            "expected_profile_ref": self.expected_profile_ref,
            "selected_profile_ref": self.selected_profile_ref,
            "production_evaluation_selected_profile": self.production_evaluation_selected_profile,
            "matched_rule_id": self.matched_rule_id,
            "routing_reason": self.routing_reason,
            "policy_route_hint": self.policy_route_hint,
            "actual_adapter_provider": self.actual_adapter_provider,
            "actual_adapter_model": self.actual_adapter_model,
            "actual_inner_adapter_provider": self.actual_inner_adapter_provider,
            "actual_inner_adapter_model": self.actual_inner_adapter_model,
            "invocation_status": self.invocation_status,
            "invocation_failure_kind": self.invocation_failure_kind,
            "raw_model_output": self.raw_model_output,
            "ops": "model_routing_summary",
        }


def _optional_str(summary: Mapping[str, object], key: str) -> str | None:
    value = summary.get(key)
    return str(value) if isinstance(value, str) else None


def model_routing_diagnostic_from_output(output: Mapping[str, object]) -> ModelRoutingSummaryDiagnostic:
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
        routing_context_summary=_optional_str(summary, "routing_context_summary"),
        candidate_profile_refs=candidates,
        expected_profile_ref=_optional_str(summary, "expected_profile_ref"),
        selected_profile_ref=_optional_str(summary, "selected_profile_ref"),
        production_evaluation_selected_profile=_optional_str(
            summary,
            "production_evaluation_selected_profile",
        ),
        matched_rule_id=_optional_str(summary, "matched_rule_id"),
        routing_reason=_optional_str(summary, "routing_reason"),
        policy_route_hint=_optional_str(summary, "policy_route_hint"),
        actual_adapter_provider=_optional_str(summary, "actual_adapter_provider"),
        actual_adapter_model=_optional_str(summary, "actual_adapter_model"),
        actual_inner_adapter_provider=_optional_str(summary, "actual_inner_adapter_provider"),
        actual_inner_adapter_model=_optional_str(summary, "actual_inner_adapter_model"),
        invocation_status=_optional_str(summary, "invocation_status"),
        invocation_failure_kind=_optional_str(summary, "invocation_failure_kind"),
        raw_model_output=_optional_str(summary, "raw_model_output"),
    )


__all__ = ["ModelRoutingSummaryDiagnostic", "model_routing_diagnostic_from_output"]
