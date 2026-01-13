# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.pipelines.no_planner_pipeline import NoPlannerPipeline
from intergrax.runtime.nexus.pipelines.pipeline_factory import PipelineFactory
from intergrax.runtime.nexus.planning.engine_plan_models import EngineNextStep, PlanIntent
from intergrax.runtime.nexus.planning.plan_sources import PlanSpec
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager

from tests._support.builder import build_runtime_config_deterministic


class _DummyPipeline(RuntimePipeline):
    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        answer = RuntimeAnswer(answer="ok-from-dummy")
        state.runtime_answer = answer
        return answer


def _make_state_with_config(*, cfg) -> RuntimeState:
    storage = InMemorySessionStorage()
    session_manager = SessionManager(storage=storage)

    ctx = RuntimeContext(
        config=cfg,
        session_manager=session_manager,
    )

    req = RuntimeRequest(
        user_id="u-001",
        session_id="s-001",
        message="hello",
    )

    return RuntimeState(
        context=ctx,
        request=req,
        run_id="run-test-001",
    )


def _build_cfg_minimal():
     # Even with OFF, deterministic builder wraps plan_specs into ScriptedPlanSource,
    # which requires a non-empty sequence. Provide a minimal benign plan.
    plan_specs = [
        PlanSpec(
            version="1",
            intent=PlanIntent.GENERIC,
            next_step=EngineNextStep.FINALIZE,
            reasoning_summary="test: minimal plan spec for deterministic harness",
            ask_clarifying_question=False,
            clarifying_question=None,
            use_websearch=False,
            use_user_longterm_memory=False,
            use_rag=False,
            use_tools=False,
            debug=None,
        )
    ]

    return build_runtime_config_deterministic(
        pipeline=None,
        plan_specs=plan_specs,
    )


def test_pipeline_factory_returns_injected_pipeline_instance() -> None:
    cfg = _build_cfg_minimal()
    injected = _DummyPipeline()
    cfg.pipeline = injected

    state = _make_state_with_config(cfg=cfg)

    pipeline = PipelineFactory.build_pipeline(state)
    assert pipeline is injected


def test_pipeline_factory_returns_default_when_pipeline_not_set() -> None:
    cfg = _build_cfg_minimal()
    cfg.pipeline = None

    state = _make_state_with_config(cfg=cfg)

    pipeline = PipelineFactory.build_pipeline(state)
    assert isinstance(pipeline, NoPlannerPipeline)


def test_pipeline_factory_rejects_none_state() -> None:
    with pytest.raises(ValueError) as exc:
        PipelineFactory.build_pipeline(None)  # type: ignore[arg-type]
    assert "State is None" in str(exc.value)
