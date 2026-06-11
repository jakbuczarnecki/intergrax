# © Artur Czarnecki. All rights reserved.

"""
ACP-12 — cognitive pattern acceptance via Agent OS (NexusLoop + typed ACP session).

One harness probe per pattern; mock LLM stub inside reference probes (no network).
"""

from __future__ import annotations

import pytest

from intergrax.agents.authoring.patterns.reference import (
    PatternDecompositionProbe,
    PatternPlanExecuteProbe,
    PatternReActProbe,
    PatternReflectionProbe,
    PatternReflexProbe,
)
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState

pytestmark = [pytest.mark.integration, pytest.mark.agent_os, pytest.mark.gate]

_PATTERN_ACCEPTANCE_CASES = (
    (PatternReflexProbe, CognitivePattern.REFLEX, "harness.pattern.reflex"),
    (PatternReActProbe, CognitivePattern.REACT, "harness.pattern.react"),
    (PatternPlanExecuteProbe, CognitivePattern.PLAN_EXECUTE, "harness.pattern.plan_execute"),
    (PatternDecompositionProbe, CognitivePattern.DECOMPOSITION, "harness.pattern.decomposition"),
    (PatternReflectionProbe, CognitivePattern.REFLECTION, "harness.pattern.reflection"),
)


@pytest.fixture
def pattern_registry() -> AgentRegistry:
    registry = AgentRegistry()
    for probe_cls, _pattern, _cap in _PATTERN_ACCEPTANCE_CASES:
        registry.register(probe_cls())
    return registry


@pytest.fixture
def pattern_nexus_loop(pattern_registry: AgentRegistry) -> NexusLoop:
    return NexusLoop(pattern_registry)


def _pattern_task(*, capability: str, message: str) -> Task:
    return Task(
        tenant_id="t-agent-os-pattern",
        user_id="u-pattern",
        message=message,
        context=TaskContext(capability=capability),
        metadata={
            AcpMetadataKey.SESSION_ENABLED: True,
            "user_id": "u-pattern",
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probe_cls", "expected_pattern", "capability"),
    _PATTERN_ACCEPTANCE_CASES,
    ids=[case[1].value for case in _PATTERN_ACCEPTANCE_CASES],
)
async def test_acceptance_acp_pattern_via_nexus_loop(
    pattern_nexus_loop: NexusLoop,
    probe_cls: type,
    expected_pattern: CognitivePattern,
    capability: str,
) -> None:
    """Task → NexusLoop → AgentEngine ACP session → pattern probe result."""
    agent = probe_cls()
    assert agent.get_contract().cognitive_pattern == expected_pattern

    result = await pattern_nexus_loop.handle_task(
        _pattern_task(capability=capability, message=f"acceptance-{expected_pattern.value}"),
    )

    assert result.state == TaskState.COMPLETED
    assert result.agent_id == agent.contract_id
    assert result.answer
