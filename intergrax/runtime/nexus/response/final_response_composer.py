# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import json
from typing import Any, Dict, List

from intergrax.contracts.orchestration_enums import MergeStrategy
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus


class FinalResponseComposer:
    """Compose Nexus final response from one or more AgentExecutionResults (§10.10, Phase B.7)."""

    def __init__(self, *, merge_strategy: MergeStrategy = MergeStrategy.CONCAT) -> None:
        self._merge_strategy = merge_strategy

    def compose_summary(self, results: List[AgentExecutionResult]) -> str:
        if not results:
            return ""
        if len(results) == 1:
            return results[0].summary or ""
        if self._merge_strategy is MergeStrategy.LAST_WINS:
            for result in reversed(results):
                summary = (result.summary or "").strip()
                if summary:
                    return summary
            return ""
        if self._merge_strategy in (
            MergeStrategy.STRUCTURED_JSON,
            MergeStrategy.CITATION_PRESERVING,
        ):
            payload = []
            for result in results:
                entry: dict[str, Any] = {
                    "agent_id": result.agent_id,
                    "status": result.status.value,
                    "summary": result.summary or "",
                }
                if self._merge_strategy is MergeStrategy.CITATION_PRESERVING:
                    citations = result.structured_data.get("citations")
                    if citations is not None:
                        entry["citations"] = citations
                payload.append(entry)
            return json.dumps({"agents": payload}, ensure_ascii=False)
        parts: List[str] = []
        for result in results:
            label = result.agent_id or "agent"
            summary = (result.summary or "").strip()
            if summary:
                parts.append(f"[{label}] {summary}")
        return "\n\n".join(parts)

    @staticmethod
    def compose_metadata(
        results: List[AgentExecutionResult],
        *,
        classification: str = "",
        plan_id: str = "",
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        return {
            "classification": classification,
            "plan_id": plan_id,
            "agent_count": len(results),
            "agent_ids": [r.agent_id for r in results],
            "retry_count": retry_count,
            "all_completed": all(r.status == AgentExecutionStatus.COMPLETED for r in results),
        }
