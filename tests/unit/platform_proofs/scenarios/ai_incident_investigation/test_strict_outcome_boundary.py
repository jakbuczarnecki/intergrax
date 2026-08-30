# © Artur Czarnecki. All rights reserved.

from __future__ import annotations
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import build_runtime_bundle

import pytest

from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import ScenarioVariant
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
    TERMINAL_STATE_NOT_ACCEPTED,
    derive_terminal_outcome,
    execute_resolved_skeleton,
    execute_with_completion_gate_blocked,
    is_epistemic_unresolved_completion,
    is_resolved_completion,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_valid_unresolved_terminal_outcome() -> None:
    bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_UNRESOLVED
    assert result.critic_verdict_passed


@pytest.mark.asyncio
async def test_critic_failure_does_not_return_unresolved() -> None:
    bundle = build_runtime_bundle()
    with pytest.raises(RuntimeError, match=TERMINAL_STATE_NOT_ACCEPTED):
        await execute_with_completion_gate_blocked(bundle)


def test_wrong_completion_mode_not_unresolved() -> None:
    with pytest.raises(RuntimeError, match=TERMINAL_STATE_NOT_ACCEPTED):
        derive_terminal_outcome(
            critic_verdict_passed=True,
            has_supported_diagnosis=False,
            completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        )


def test_supported_diagnosis_with_unresolved_completion_mode_not_unresolved() -> None:
    assert not is_epistemic_unresolved_completion(
        critic_verdict_passed=True,
        has_supported_diagnosis=True,
        completion_mode=COMPLETION_UNRESOLVED,
    )
    with pytest.raises(RuntimeError, match=TERMINAL_STATE_NOT_ACCEPTED):
        derive_terminal_outcome(
            critic_verdict_passed=True,
            has_supported_diagnosis=True,
            completion_mode=COMPLETION_UNRESOLVED,
        )


def test_critic_fail_predicate_neither_resolved_nor_unresolved() -> None:
    assert not is_resolved_completion(
        critic_verdict_passed=False,
        has_supported_diagnosis=False,
        completion_mode=COMPLETION_UNRESOLVED,
    )
    assert not is_epistemic_unresolved_completion(
        critic_verdict_passed=False,
        has_supported_diagnosis=False,
        completion_mode=COMPLETION_UNRESOLVED,
    )


@pytest.mark.asyncio
async def test_valid_resolved_terminal_outcome() -> None:
    bundle = build_runtime_bundle(variant=ScenarioVariant.RESOLVED)
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED
    assert is_resolved_completion(
        critic_verdict_passed=result.critic_verdict_passed,
        has_supported_diagnosis=True,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
    )
