# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""When NoPlannerPipeline should include RagStep (Phase Q-R.4)."""

from __future__ import annotations

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


def pipeline_should_include_rag_step(state: RuntimeState) -> bool:
    cfg = state.context.config
    if not cfg.enable_rag:
        return False
    plan = state.engine_plan
    if plan is not None:
        tool_ids = getattr(plan, "tool_ids", None) or []
        if tool_ids and "rag.retrieve" not in tool_ids and not getattr(plan, "use_rag", True):
            return False
        if hasattr(plan, "use_rag") and plan.use_rag is False:
            return False
    return cfg.retrieval_service is not None or (
        cfg.embedding_manager is not None and cfg.vectorstore_manager is not None
    )
