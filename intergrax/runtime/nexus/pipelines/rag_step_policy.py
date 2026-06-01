# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""When NoPlannerPipeline should include RagStep (Phase Q-R.4)."""

from __future__ import annotations

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.engine_plan_models import EnginePlan


def pipeline_should_include_rag_step(state: RuntimeState) -> bool:
    cfg = state.context.config
    if not cfg.enable_rag:
        return False
    plan = state.engine_plan
    if isinstance(plan, EnginePlan):
        tool_ids = plan.resolved_tool_ids()
        if tool_ids and "rag.retrieve" not in tool_ids and not plan.use_rag:
            return False
        if not plan.use_rag:
            return False
    return cfg.retrieval_service is not None or (
        cfg.embedding_manager is not None and cfg.vectorstore_manager is not None
    )
