# © Artur Czarnecki. All rights reserved.

"""Typed tool-selection evidence payloads for AgentStepRecord.diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

_TOOL_SELECTION_SUMMARY_V1 = "lkw.tool_selection_summary.v1"


@dataclass(frozen=True)
class ToolSelectionSummaryDiagnostic(DiagnosticPayload):
    used: bool
    reason: str
    selected_tool_id: str | None = None
    selected_artifact_ref: str | None = None
    invoke_status: str | None = None
    available_tool_ids: tuple[str, ...] = ()

    @classmethod
    def schema_id(cls) -> str:
        return _TOOL_SELECTION_SUMMARY_V1

    def redact(self) -> ToolSelectionSummaryDiagnostic:
        return ToolSelectionSummaryDiagnostic(
            used=self.used,
            reason=self.reason,
            selected_tool_id=self.selected_tool_id,
            selected_artifact_ref=self.selected_artifact_ref,
            invoke_status=self.invoke_status,
            available_tool_ids=self.available_tool_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "used": self.used,
            "reason": self.reason,
            "selected_tool_id": self.selected_tool_id,
            "selected_artifact_ref": self.selected_artifact_ref,
            "invoke_status": self.invoke_status,
            "available_tool_ids": list(self.available_tool_ids),
            "ops": "tool_selection_summary",
        }


def tool_selection_diagnostic_from_output(output: dict[str, object]) -> ToolSelectionSummaryDiagnostic:
    summary = output.get("tool_selection_summary")
    if not isinstance(summary, dict):
        return ToolSelectionSummaryDiagnostic(used=False, reason="summary_missing")
    raw_tools = summary.get("available_tool_ids")
    tools: tuple[str, ...] = ()
    if isinstance(raw_tools, list):
        tools = tuple(str(item) for item in raw_tools if isinstance(item, str))
    return ToolSelectionSummaryDiagnostic(
        used=bool(summary.get("used")),
        reason=str(summary.get("reason") or "unknown"),
        selected_tool_id=(
            str(summary["selected_tool_id"])
            if isinstance(summary.get("selected_tool_id"), str)
            else None
        ),
        selected_artifact_ref=(
            str(summary["selected_artifact_ref"])
            if isinstance(summary.get("selected_artifact_ref"), str)
            else None
        ),
        invoke_status=(
            str(summary["invoke_status"]) if isinstance(summary.get("invoke_status"), str) else None
        ),
        available_tool_ids=tools,
    )


__all__ = ["ToolSelectionSummaryDiagnostic", "tool_selection_diagnostic_from_output"]
