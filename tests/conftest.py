# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import sys
from pathlib import Path

from intergrax.runtime.nexus.pipelines.planner_dynamic_pipeline import PlannerDynamicPipeline
from intergrax.runtime.nexus.pipelines.planner_static_pipeline import PlannerStaticPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest

from tests._support.builder import DeterministicRuntimeHarness, build_engine_harness, build_in_memory_session_manager, build_runtime_config_deterministic
from intergrax.runtime.nexus.planning.engine_plan_models import EngineNextStep, PlanIntent
from intergrax.runtime.nexus.planning.plan_sources import PlanSpec



@pytest.fixture
def session_manager_in_memory():
    return build_in_memory_session_manager()


@pytest.fixture
def harness_static(session_manager_in_memory) -> DeterministicRuntimeHarness:
    # Minimal deterministic plan for STATIC smoke: planner returns FINALIZE.
    plans = [
        PlanSpec(
            version="1",
            intent=PlanIntent.GENERIC,
            next_step=EngineNextStep.FINALIZE,
            reasoning_summary="Deterministic STATIC plan for integration tests.",
            ask_clarifying_question=False,
            clarifying_question=None,
            use_websearch=False,
            use_user_longterm_memory=False,
            use_rag=False,
            use_tools=False,
            debug=None,
        )
    ]

    cfg = build_runtime_config_deterministic(
        pipeline= PlannerStaticPipeline(),
        plan_specs=plans,
        llm_text="OK",
    )
    return build_engine_harness(cfg=cfg, session_manager=session_manager_in_memory)


@pytest.fixture
def harness_dynamic(session_manager_in_memory) -> DeterministicRuntimeHarness:
    # Deterministic two-iteration DYNAMIC: SYNTHESIZE then FINALIZE.
    plans = [
        PlanSpec(
            version="1",
            intent=PlanIntent.GENERIC,
            next_step=EngineNextStep.SYNTHESIZE,
            reasoning_summary="Iter1: synthesize draft.",
            ask_clarifying_question=False,
            clarifying_question=None,
            use_websearch=False,
            use_user_longterm_memory=False,
            use_rag=False,
            use_tools=False,
            debug=None,
        ),
        PlanSpec(
            version="1",
            intent=PlanIntent.GENERIC,
            next_step=EngineNextStep.FINALIZE,
            reasoning_summary="Iter2: finalize answer.",
            ask_clarifying_question=False,
            clarifying_question=None,
            use_websearch=False,
            use_user_longterm_memory=False,
            use_rag=False,
            use_tools=False,
            debug=None,
        ),
    ]

    cfg = build_runtime_config_deterministic(
        pipeline= PlannerDynamicPipeline(),
        plan_specs=plans,
        llm_text="OK",
    )
    return build_engine_harness(cfg=cfg, session_manager=session_manager_in_memory)