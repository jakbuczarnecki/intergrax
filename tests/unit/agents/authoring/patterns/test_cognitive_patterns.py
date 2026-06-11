# © Artur Czarnecki. All rights reserved.

"""Backward-compatible import path — pattern package entry (ACP-10)."""

from __future__ import annotations

import pytest

from intergrax.agents.authoring.patterns import PATTERN_AGENT_BY_ID
from intergrax.contracts.agent_run_enums import CognitivePattern


@pytest.mark.unit
@pytest.mark.gate
def test_pattern_package_exports_all_five_cognitive_patterns() -> None:
    expected = {
        CognitivePattern.REFLEX,
        CognitivePattern.REACT,
        CognitivePattern.PLAN_EXECUTE,
        CognitivePattern.DECOMPOSITION,
        CognitivePattern.REFLECTION,
    }
    assert set(PATTERN_AGENT_BY_ID) == expected
