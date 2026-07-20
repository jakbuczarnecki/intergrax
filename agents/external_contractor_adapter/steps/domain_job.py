# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.agents.authoring.runtime_tool_helpers import exec_ctx_from_step, request_metadata

DOMAIN_STEP_ID = "external_contractor_adapter_step"


async def run_domain_job(step_ctx: AgentStepContext) -> dict[str, object]:
    """Cursor implementation point — see intergrax/agents/authoring/runtime_tool_helpers.py."""
    _ = exec_ctx_from_step(step_ctx), request_metadata(None), DOMAIN_STEP_ID
    # Capability id kept in smoke output until GEC-3 implements real mapping.
    answer = (
        "external_contractor_adapter: domain job not implemented "
        "(external_contractor.adapt)"
    )
    return {
        "summary": answer,
        "answer": answer,
        "run_id": step_ctx.run_id,
        "domain_summary": {
            "used": False,
            "reason": "not_implemented",
            "capability": "external_contractor.adapt",
        },
    }
