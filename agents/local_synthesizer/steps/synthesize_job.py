# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.contracts.agent_step_context import AgentStepContext
from lkw_shared.runtime_helpers import exec_ctx_from_step, request_metadata

SYNTHESIZE_STEP_ID = "local_synthesizer_step"


def _failure_output(*, run_id: str, reason: str) -> dict[str, object]:
    answer = f"local_synthesizer: synthesize failed — {reason}"
    return {
        "summary": answer,
        "answer": answer,
        "run_id": run_id,
        "synthesize_summary": {
            "used": False,
            "reason": reason,
            "artifact_path": None,
        },
    }


async def run_synthesize_job(step_ctx: AgentStepContext) -> dict[str, object]:
    """LKW.1.3 implementation point — wire shadow workspace.write_file here. Pattern: agents/lkw_shared/PATTERN.md"""
    exec_ctx = exec_ctx_from_step(step_ctx)
    metadata = request_metadata(exec_ctx)
    if not metadata.get("shadow_workspace") and not metadata.get("output_path"):
        return _failure_output(run_id=step_ctx.run_id, reason="output_target_missing")

    _ = exec_ctx, SYNTHESIZE_STEP_ID
    return _failure_output(run_id=step_ctx.run_id, reason="not_implemented")
