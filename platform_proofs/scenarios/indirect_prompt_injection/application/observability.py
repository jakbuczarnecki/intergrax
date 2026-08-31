"""Scenario application observability contract."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.nexus.tracing.trace_models import DEFAULT_REDACTED_TEXT, DiagnosticPayload


@dataclass(frozen=True, slots=True)
class OrderRetrievalDiagV1(DiagnosticPayload):
    order_id: str
    tool_id: str
    note_count: int

    @classmethod
    def schema_id(cls) -> str:
        return "order_assistant.retrieval.v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "tool_id": self.tool_id,
            "note_count": self.note_count,
        }

    def redact(self) -> OrderRetrievalDiagV1:
        return OrderRetrievalDiagV1(
            order_id=self.order_id,
            tool_id=self.tool_id,
            note_count=self.note_count,
        )


@dataclass(frozen=True, slots=True)
class OrderPlannerRoundDiagV1(DiagnosticPayload):
    round_index: int
    proposed_tool_ids: tuple[str, ...]
    assistant_excerpt: str

    @classmethod
    def schema_id(cls) -> str:
        return "order_assistant.planner_round.v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "proposed_tool_ids": list(self.proposed_tool_ids),
            "assistant_excerpt": self.assistant_excerpt,
        }

    def redact(self) -> OrderPlannerRoundDiagV1:
        return OrderPlannerRoundDiagV1(
            round_index=self.round_index,
            proposed_tool_ids=self.proposed_tool_ids,
            assistant_excerpt=DEFAULT_REDACTED_TEXT,
        )


@dataclass(frozen=True, slots=True)
class OrderPolicyDenialDiagV1(DiagnosticPayload):
    tool_id: str
    matched_rule_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    @classmethod
    def schema_id(cls) -> str:
        return "order_assistant.policy_denial.v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_id": self.tool_id,
            "matched_rule_ids": list(self.matched_rule_ids),
            "reasons": list(self.reasons),
        }

    def redact(self) -> OrderPolicyDenialDiagV1:
        return self


@dataclass(frozen=True, slots=True)
class OrderWorkflowCompletionDiagV1(DiagnosticPayload):
    workflow_kind: str
    outcome: str
    write_tool_proposed: bool
    write_tool_executed: bool
    policy_denied: bool

    @classmethod
    def schema_id(cls) -> str:
        return "order_assistant.workflow_completion.v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_kind": self.workflow_kind,
            "outcome": self.outcome,
            "write_tool_proposed": self.write_tool_proposed,
            "write_tool_executed": self.write_tool_executed,
            "policy_denied": self.policy_denied,
        }

    def redact(self) -> OrderWorkflowCompletionDiagV1:
        return self
