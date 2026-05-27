# © Artur Czarnecki. All rights reserved.

import pytest

from legal.config.legal_agent_config import LegalAgentConfig
from legal.legal_agent import LegalAgent
from legal.uaep.dynamic_steps import (
    FINAL_DYNAMIC_STEP_ID,
    LEGAL_DYNAMIC_STEP_DEFS,
    legal_dynamic_agent_steps,
)
from legal.uaep.thin_steps import (
    FINAL_SEQUENTIAL_STEP_ID,
    LEGAL_SEQUENTIAL_STEP_DEFS,
)
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_legal_sequential_get_steps_exposes_domain_steps():
    cfg = LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(fixed_text="ok"),
        production_mode=False,
        enable_sequential_legal_pipeline=True,
    )
    agent = LegalAgent(config=cfg)
    steps = agent.get_steps(context=None)  # type: ignore[arg-type]

    assert len(steps) == len(LEGAL_SEQUENTIAL_STEP_DEFS)
    assert steps[0].step_id == "legal_setup"
    assert steps[-1].step_id == FINAL_SEQUENTIAL_STEP_ID
    assert [step.step_index for step in steps] == list(range(len(steps)))


def test_legal_dynamic_get_steps_exposes_macro_steps():
    cfg = LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(fixed_text="ok"),
        production_mode=False,
        enable_sequential_legal_pipeline=False,
    )
    agent = LegalAgent(config=cfg)
    steps = agent.get_steps(context=None)  # type: ignore[arg-type]

    assert len(steps) == len(LEGAL_DYNAMIC_STEP_DEFS)
    assert steps[0].step_id == "legal_setup_dynamic"
    assert steps[-1].step_id == FINAL_DYNAMIC_STEP_ID
