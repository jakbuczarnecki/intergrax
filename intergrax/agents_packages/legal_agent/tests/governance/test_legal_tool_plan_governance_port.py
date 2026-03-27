# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for :class:`LegalToolPlanGovernancePort` wiring helpers (FakeLLM, dummy guard).

Full agent + Ollama coverage: ``test_legal_agent_governance_port_e2e.py``.
"""

from __future__ import annotations

import pytest

from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.domain.legal_tool_plan import LegalToolPlan
from intergrax.agents_packages.legal_agent.governance.legal_tool_plan_governance_impl import (
    CallableLegalToolPlanGovernance,
    PassthroughLegalToolPlanGovernance,
)
from intergrax.agents_packages.legal_agent.governance.legal_platform_policy_governance import (
    DualLegalGovernanceService,
    LegalNexusLayerCaps,
    ResolvingLegalToolPlanGovernance,
    StaticLegalExecutionPolicy,
    apply_legal_nexus_layer_caps_to_plan,
)
from intergrax.agents_packages.legal_agent.governance.legal_tool_plan_governance_port import (
    LegalToolPlanGovernancePort,
)
from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

from testing_support.builder import (
    DummyExecutionGuard,
    FakeLLMAdapter,
    build_in_memory_session_manager,
    build_runtime_state_for_tests,
)

pytestmark = pytest.mark.unit


def _cfg(**kwargs: object) -> LegalAgentConfig:
    base = dict(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(),
    )
    base.update(kwargs)
    return LegalAgentConfig(**base)  # type: ignore[arg-type]


def test_platform_can_use_one_class_for_post_run_and_dynamic_port() -> None:
    """Same object on governance_service + legal_tool_plan_governance when subclass adds adjust."""

    class CombinedGovernance(GovernanceService, LegalToolPlanGovernancePort):
        def adjust_legal_tool_plan(
            self,
            plan: LegalToolPlan,
            *,
            state: RuntimeState,
            legal_config: LegalAgentConfig,
        ) -> LegalToolPlan:
            return plan.model_copy(update={"use_tools": False})

    g = CombinedGovernance(guard=DummyExecutionGuard())
    cfg = _cfg(governance_service=g, legal_tool_plan_governance=g)
    plan = LegalToolPlan(
        intent="tools",
        confidence=1.0,
        use_rag=False,
        use_tools=True,
        use_websearch=False,
    )
    out = g.adjust_legal_tool_plan(plan, state=build_runtime_state_for_tests(run_id="x"), legal_config=cfg)
    assert out.use_tools is False


def test_passthrough_returns_same_plan_instance() -> None:
    state = build_runtime_state_for_tests(run_id="gov-pass")
    cfg = _cfg()
    plan = LegalToolPlan(
        intent="rag",
        confidence=1.0,
        use_rag=True,
        use_tools=False,
        use_websearch=False,
    )
    gov = PassthroughLegalToolPlanGovernance()
    out = gov.adjust_legal_tool_plan(plan, state=state, legal_config=cfg)
    assert out is plan


def test_callable_delegates_with_positional_callback() -> None:
    state = build_runtime_state_for_tests(run_id="gov-call")
    cfg = _cfg()

    def _fn(
        p: LegalToolPlan,
        st: RuntimeState,
        lc: LegalAgentConfig,
    ) -> LegalToolPlan:
        assert st is state
        assert lc is cfg
        return p.model_copy(update={"use_websearch": False, "intent": "rag"})

    plan = LegalToolPlan(
        intent="combination",
        confidence=0.5,
        use_rag=True,
        use_tools=False,
        use_websearch=True,
    )
    gov = CallableLegalToolPlanGovernance(_fn)
    out = gov.adjust_legal_tool_plan(plan, state=state, legal_config=cfg)
    assert out.use_websearch is False
    assert out.intent == "rag"


def test_resolving_governance_applies_policy_caps() -> None:
    state = build_runtime_state_for_tests(run_id="gov-resolve")
    cfg = _cfg()
    policy = StaticLegalExecutionPolicy(
        caps=LegalNexusLayerCaps(allow_rag=False, allow_websearch=True, allow_tools=True),
    )
    gov = ResolvingLegalToolPlanGovernance(policy=policy)
    plan = LegalToolPlan(
        intent="rag",
        confidence=1.0,
        use_rag=True,
        use_tools=False,
        use_websearch=False,
    )
    out = gov.adjust_legal_tool_plan(plan, state=state, legal_config=cfg)
    assert out.use_rag is False
    assert out.intent == "llm_only"
    assert "execution policy clamp" in (out.reasoning_summary or "")


def test_dual_governance_service_is_single_wiring_instance() -> None:
    state = build_runtime_state_for_tests(run_id="gov-dual")
    cfg = _cfg()
    policy = StaticLegalExecutionPolicy(
        caps=LegalNexusLayerCaps(allow_rag=True, allow_websearch=False, allow_tools=True),
    )
    g = DualLegalGovernanceService(guard=DummyExecutionGuard(), policy=policy)
    assert isinstance(g, GovernanceService)
    assert isinstance(g, LegalToolPlanGovernancePort)
    plan = LegalToolPlan(
        intent="websearch",
        confidence=1.0,
        use_rag=False,
        use_tools=False,
        use_websearch=True,
    )
    out = g.adjust_legal_tool_plan(plan, state=state, legal_config=cfg)
    assert out.use_websearch is False


def test_apply_caps_noop_when_nothing_clamped() -> None:
    state = build_runtime_state_for_tests(run_id="gov-cap-noop")
    plan = LegalToolPlan(
        intent="rag",
        confidence=1.0,
        use_rag=True,
        use_tools=False,
        use_websearch=False,
    )
    caps = LegalNexusLayerCaps()
    out = apply_legal_nexus_layer_caps_to_plan(plan=plan, state=state, caps=caps)
    assert out is plan
