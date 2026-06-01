# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.engine_planner import EnginePlanner
from intergrax.runtime.nexus.planning.plan_sources import PlanSource, PlanSourceMeta, PlanSourceResult
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager


class _InvalidJsonPlanSource(PlanSource):
    async def generate_plan_raw(self, *, req):
        return PlanSourceResult(
            raw="not-json",
            meta=PlanSourceMeta(source_kind="test", source_detail="invalid"),
        )


@pytest.mark.asyncio
@pytest.mark.gate
async def test_engine_planner_emits_plan_failed_on_invalid_json() -> None:
    bus = RuntimeEventBus()
    adapter = MagicMock()
    planner = EnginePlanner(llm_adapter=adapter, plan_source=_InvalidJsonPlanSource())
    config = RuntimeConfig(
        llm_adapter=adapter,
        enable_rag=False,
        production_mode=False,
        runtime_event_bus=bus,
    )
    session_manager = SessionManager(storage=InMemorySessionStorage())
    context = RuntimeContext.build(config=config, session_manager=session_manager)
    state = RuntimeState(
        context=context,
        request=RuntimeRequest(
            agent_id="a",
            user_id="u1",
            session_id="s1",
            message="plan me",
            tenant_id="t1",
            metadata={"task_id": "task-1"},
        ),
        run_id="run-1",
    )

    with pytest.raises(ValueError):
        await planner.plan(
            req=state.request,
            state=state,
            config=config,
            run_id="run-1",
        )

    failed = [e for e in bus.history if e.event_type == RuntimeEventType.PLAN_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["failure_kind"] == "parse"
    assert failed[0].task_id == "task-1"
