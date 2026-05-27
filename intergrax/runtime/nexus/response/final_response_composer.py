# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Dict, List

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus


class FinalResponseComposer:
    """Compose Nexus final response from one or more AgentExecutionResults (§10.10, Phase B.7)."""

    @staticmethod
    def compose_summary(results: List[AgentExecutionResult]) -> str:
        if not results:
            return ""
        if len(results) == 1:
            return results[0].summary or ""
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
