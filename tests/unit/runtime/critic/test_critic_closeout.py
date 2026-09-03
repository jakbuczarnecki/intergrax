# © Artur Czarnecki. All rights reserved.

"""ToolRegistryCriticEvalClient and L1 wiring tests (Phase CRIT-V-FOLLOWUP)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.critic_tool_wiring import build_critic_eval_tool_client
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    CriticProfile,
    CriticVerificationScopes,
)
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.critic.critic_orchestrator import CriticOrchestrator
from intergrax.runtime.critic.l1_gateway import L1Gateway
from intergrax.runtime.critic.tool_registry_client import ToolRegistryCriticEvalClient
from intergrax.tools.providers.eval.contracts import EvalJudgeInput
from intergrax.tools.providers.eval.judge import _JudgeLLMResult
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_tool_registry_critic_client_judge_uses_wiring_context_llm() -> None:
    adapter = FakeLLMAdapter(
        fake_structured_data=_JudgeLLMResult(score=0.95, passed=True, reasons=[]),
    )
    ctx = ToolWiringContext(extras={"llm_adapter": adapter})
    client = ToolRegistryCriticEvalClient(ctx)
    result = client.judge(
        EvalJudgeInput(
            output_text="candidate answer",
            rubric_id="rubric.a",
            criteria=["complete"],
            min_score=0.75,
        )
    )
    assert result.passed is True
    assert result.score == 0.95


def test_build_critic_eval_tool_client_returns_none_when_layers_disabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="critic.client.off")
    from intergrax.applications._shared.tool_wiring import build_application_tool_wiring
    from intergrax.tools.registry.profile import ToolProfile

    tool_wiring = build_application_tool_wiring(ToolProfile(enabled_bundles=("eval",)))
    assert build_critic_eval_tool_client(env, tool_wiring) is None


def test_build_critic_eval_tool_client_materializes_when_semantic_enabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="critic.client.llm")
    env.critic_profile = CriticProfile(
        semantic_judge_enabled=True,
        default_rubric_ref="rubric.default",
        critic_llm_profile=LLMProfile.lab(),
        scopes=CriticVerificationScopes(graph_final=True),
    )
    from intergrax.applications._shared.tool_wiring import build_application_tool_wiring
    from intergrax.tools.registry.profile import ToolProfile

    tool_wiring = build_application_tool_wiring(
        ToolProfile(enabled_bundles=("eval",)),
        extras={"llm_adapter": FakeLLMAdapter()},
    )
    client = build_critic_eval_tool_client(env, tool_wiring)
    assert client is not None
    orchestrator = CriticOrchestrator(l1_gateway=L1Gateway(tool_client=client))
    assert orchestrator.l1_client_configured is True
