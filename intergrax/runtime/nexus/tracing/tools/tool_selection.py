# © Artur Czarnecki. All rights reserved.

"""Tool selection diagnostic payloads (TOOL-ENG-32)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class ToolSelectionCandidateDiagV1:
    tool_id: str
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"tool_id": self.tool_id}
        if self.score is not None:
            payload["score"] = self.score
        return payload


@dataclass(frozen=True)
class ToolSelectionDiagV1(DiagnosticPayload):
    strategy_id: str
    selection_mode: str
    candidate_tool_ids: List[str]
    candidates: List[ToolSelectionCandidateDiagV1]
    ops: str = "tool_selection"

    def redact(self) -> ToolSelectionDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.tools.selection"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "selection_mode": self.selection_mode,
            "candidate_tool_ids": list(self.candidate_tool_ids),
            "candidates": [item.to_dict() for item in self.candidates],
            "ops": self.ops,
        }
