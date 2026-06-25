# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.agent_step_context import AgentStepContext
from lkw_shared.runtime_helpers import exec_ctx_from_step, request_metadata

SEARCH_STEP_ID = "local_search_step"


def _failure_output(*, run_id: str, reason: str) -> dict[str, object]:
    answer = f"local_search: search failed — {reason}"
    return {
        "summary": answer,
        "answer": answer,
        "run_id": run_id,
        "search_summary": {
            "used": False,
            "reason": reason,
            "evidence": [],
        },
    }


async def run_search_job(step_ctx: AgentStepContext) -> dict[str, object]:
    """LKW.1.2 implementation point — wire rag.retrieve here. Pattern: agents/lkw_shared/PATTERN.md"""
    exec_ctx = exec_ctx_from_step(step_ctx)
    metadata = request_metadata(exec_ctx)
    run_input = step_ctx.metadata.get(AcpRunContextKey.RUN_INPUT, metadata.get("message", ""))
    if isinstance(run_input, dict):
        run_input = str(run_input.get("message") or run_input.get("summary") or "")
    query = str(metadata.get("query") or run_input or step_ctx.message or "").strip()
    if not query:
        return _failure_output(run_id=step_ctx.run_id, reason="query_missing")

    _ = exec_ctx, SEARCH_STEP_ID
    return _failure_output(run_id=step_ctx.run_id, reason="not_implemented")
