# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional, List

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_usage_track import LLMUsageTracker
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.planning.engine_planner import EnginePlanner
from intergrax.runtime.nexus.planning.engine_plan_models import EnginePlan


async def build_plan(
    *,
    config: RuntimeConfig,
    message: str,
    user_id: str,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    instructions: Optional[str] = None,
    attachments: Optional[list] = None,
    base_history: Optional[List[ChatMessage]] = None,
    profile_user_instructions: Optional[str] = None,
    profile_org_instructions: Optional[str] = None
) -> EnginePlan:
    """
    Minimal helper:
      EnginePlanner.plan() -> EnginePlan
    """

    inner_session_id = session_id or "plan_builder_helper_session"
    inner_run_id = run_id or "plan_builder_helper_run_id"

    planner = EnginePlanner(llm_adapter=config.llm_adapter)

    # 1. Request
    req = RuntimeRequest(
        user_id=user_id,
        session_id=inner_session_id,
        message=message,
        instructions=instructions,
        attachments=attachments or [],
    )

    # 2. State
    state = RuntimeState(
        request=req,
        run_id=inner_run_id,
        llm_usage_tracker=LLMUsageTracker(run_id=inner_run_id),
    )

    state.llm_usage_tracker.register_adapter(config.llm_adapter, label="core_adapter")

    state.base_history = base_history or []

    state.profile_user_instructions = profile_user_instructions
    state.profile_org_instructions = profile_org_instructions

    state.cap_rag_available = config.enable_rag
    state.cap_user_ltm_available = config.enable_user_longterm_memory
    state.cap_attachments_available = bool(attachments and len(attachments)>0)
    state.cap_websearch_available = config.enable_websearch
    state.cap_tools_available = config.tools_mode != "off"

    # 3. Engine plan
    engine_plan = await planner.plan(
        req=req,
        state=state,
        config=config,
        run_id=inner_run_id,
    )

    return engine_plan
