# © Artur Czarnecki. All rights reserved.

"""Typed run smoke tests — one harness probe per cognitive pattern (ACP-10)."""

from __future__ import annotations

import pytest

from intergrax.agents.authoring.patterns.reference import (
    PatternDecompositionProbe,
    PatternPlanExecuteProbe,
    PatternReActProbe,
    PatternReflectionProbe,
    PatternReflexProbe,
)
from intergrax.contracts.agent_run import AgentRunRequest
from intergrax.contracts.agent_run_enums import AgentRunStatus, CognitivePattern

_PROBE_CASES = (
    (PatternReflexProbe, CognitivePattern.REFLEX, 1),
    (PatternReActProbe, CognitivePattern.REACT, 1),
    (PatternPlanExecuteProbe, CognitivePattern.PLAN_EXECUTE, 3),
    (PatternDecompositionProbe, CognitivePattern.DECOMPOSITION, 1),
    (PatternReflectionProbe, CognitivePattern.REFLECTION, 3),
)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    ("probe_cls", "expected_pattern", "min_steps"),
    _PROBE_CASES,
    ids=[case[1].value for case in _PROBE_CASES],
)
async def test_pattern_probe_typed_run_succeeds(
    probe_cls: type,
    expected_pattern: CognitivePattern,
    min_steps: int,
    pattern_run_request: AgentRunRequest,
) -> None:
    agent = probe_cls()
    contract = agent.get_contract()
    assert contract.cognitive_pattern == expected_pattern

    result = await agent.run(pattern_run_request)

    assert result.status == AgentRunStatus.SUCCEEDED
    assert result.trace.total_steps >= min_steps
    assert result.output not in ("", None, {})
