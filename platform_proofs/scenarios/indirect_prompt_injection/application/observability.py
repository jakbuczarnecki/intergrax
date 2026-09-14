"""Scenario application observability contract."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


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
class OrderWorkflowCompletionDiagV1(DiagnosticPayload):
    """Business workflow outcome only — policy authority is platform DeclarativePolicyEvaluationDiagV1."""

    workflow_kind: str
    outcome: str
    order_id: str
    retrieved_note_count: int

    @classmethod
    def schema_id(cls) -> str:
        return "order_assistant.workflow_completion.v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_kind": self.workflow_kind,
            "outcome": self.outcome,
            "order_id": self.order_id,
            "retrieved_note_count": self.retrieved_note_count,
        }

    def redact(self) -> OrderWorkflowCompletionDiagV1:
        return self
