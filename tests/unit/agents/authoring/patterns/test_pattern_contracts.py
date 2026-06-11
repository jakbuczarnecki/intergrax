# © Artur Czarnecki. All rights reserved.

"""Contract and registry coverage for cognitive pattern library (ACP-10)."""

from __future__ import annotations

import pytest

from intergrax.agents.authoring.patterns import (
    PATTERN_AGENT_BY_ID,
    PATTERN_VERSION,
    DecompositionAgent,
    PlanExecuteAgent,
    ReActAgent,
    ReflexAgent,
    ReflectionAgent,
)
from intergrax.agents.authoring.patterns.reference import (
    PatternDecompositionProbe,
    PatternPlanExecuteProbe,
    PatternReActProbe,
    PatternReflectionProbe,
    PatternReflexProbe,
)
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.runtime.registry.agent_assembly_resolver import validate_cognitive_pattern_metadata

_PATTERN_CLASSES = (
    ReflexAgent,
    ReActAgent,
    PlanExecuteAgent,
    DecompositionAgent,
    ReflectionAgent,
)

_PROBE_CLASSES = (
    PatternReflexProbe,
    PatternReActProbe,
    PatternPlanExecuteProbe,
    PatternDecompositionProbe,
    PatternReflectionProbe,
)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("pattern_cls", _PATTERN_CLASSES)
def test_pattern_library_maps_cognitive_pattern_to_base_class(pattern_cls: type) -> None:
    assert PATTERN_AGENT_BY_ID[pattern_cls.cognitive_pattern] is pattern_cls


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("pattern_cls", _PATTERN_CLASSES)
def test_pattern_base_declares_version_and_main_step(pattern_cls: type) -> None:
    assert pattern_cls.pattern_version == PATTERN_VERSION
    assert pattern_cls.main_step_id
    assert pattern_cls.cognitive_pattern != CognitivePattern.CUSTOM


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("probe_cls", _PROBE_CLASSES)
def test_pattern_probe_contract_passes_assembly_validation(probe_cls: type) -> None:
    contract = probe_cls().get_contract()
    result = validate_cognitive_pattern_metadata(contract)
    assert result.valid, result.errors


@pytest.mark.unit
@pytest.mark.gate
def test_cognitive_pattern_contract_validation_requires_pattern_version() -> None:
    contract = AgentContract(
        id="demo",
        name="Demo",
        description="demo",
        capabilities=["demo.cap"],
        cognitive_pattern=CognitivePattern.REFLEX,
        risk_level=AgentRiskLevel.LOW,
        max_steps=1,
    )
    result = validate_cognitive_pattern_metadata(contract)
    assert not result.valid
