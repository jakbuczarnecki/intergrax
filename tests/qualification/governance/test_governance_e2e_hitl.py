# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-4 — HITL and Decision-flow E2E via MP-4R7 canonical qualification composition."""

from __future__ import annotations

import pytest

from intergrax.contracts.decision_authorization import DecisionGovernanceDisposition
from intergrax.contracts.decision_human_review import DecisionHumanReviewOutcome
from intergrax.contracts.execution_continuation import ExecutionContinuationLifecycleState

from testing_support.mp4r7_enterprise_integration.composition import (
    open_mp4r7_enterprise_integration_composition,
)
from testing_support.mp4r7_enterprise_integration.contracts import (
    Mp4R7QualificationDisposition,
    Mp4R7ScenarioId,
)
from testing_support.mp4r7_enterprise_integration.scenario import (
    Mp4R7EnterpriseIntegrationScenarioExecutor,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_scenario_n_through_o_mp4r7_require_human_then_resume() -> None:
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(
        open_mp4r7_enterprise_integration_composition(),
    )
    result = await executor.run_success()
    assert result.disposition is Mp4R7QualificationDisposition.QUALIFIED
    assert result.governance_required_human is True
    assert result.human_outcome is DecisionHumanReviewOutcome.APPROVED
    assert result.post_human_governance_disposition is DecisionGovernanceDisposition.ALLOW
    assert result.continuation_result_state.value == "resumed"
    assert result.protected_operation_completed is True


@pytest.mark.asyncio
async def test_scenario_p_human_approve_fresh_governance_deny_zero_effect() -> None:
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(
        open_mp4r7_enterprise_integration_composition(),
    )
    result = await executor.run_governance_deny_after_human_approve()
    assert result.scenario_id is Mp4R7ScenarioId.GOVERNANCE_DENY
    assert result.human_outcome is DecisionHumanReviewOutcome.APPROVED
    assert result.post_human_governance_disposition is DecisionGovernanceDisposition.DENY
    assert result.protected_operation_completed is False
    assert result.continuation_result_state is not ExecutionContinuationLifecycleState.RESUMED
    assert not result.evidence_records


@pytest.mark.asyncio
async def test_scenario_q_human_reject_no_resume() -> None:
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(
        open_mp4r7_enterprise_integration_composition(),
    )
    result = await executor.run_human_reject()
    assert result.protected_operation_completed is False
    assert result.continuation_result_state is not None
    assert result.continuation_result_state.value == "rejected"
