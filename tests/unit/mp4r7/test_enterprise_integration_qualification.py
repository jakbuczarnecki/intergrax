# © Artur Czarnecki. All rights reserved.

"""MP-4R7 enterprise integration qualification tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage

from testing_support.mp4r7_enterprise_integration.approver import (
    qualification_approver_factory_call_count,
    reset_qualification_approver_factory_call_count,
)
from testing_support.mp4r7_enterprise_integration.composition import (
    open_mp4r7_enterprise_integration_composition,
)
from testing_support.mp4r7_enterprise_integration.contracts import (
    Mp4R7ProtectedOperationError,
    Mp4R7QualificationDisposition,
    Mp4R7ScenarioId,
)
from testing_support.mp4r7_enterprise_integration.scenario import (
    Mp4R7EnterpriseIntegrationScenarioExecutor,
)

pytestmark = pytest.mark.unit


def _assert_identity_continuity(result) -> None:
    if len(result.execution_identities) < 2:
        return
    first = result.execution_identities[0]
    for snapshot in result.execution_identities[1:]:
        assert snapshot.run_id == first.run_id
        assert snapshot.attempt_id == first.attempt_id
        assert snapshot.execution_id == first.execution_id


@pytest.mark.asyncio
async def test_mp4r7_same_human_review_authorizes_continuation() -> None:
    reset_qualification_approver_factory_call_count()
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(
        open_mp4r7_enterprise_integration_composition(),
    )
    result = await executor.run_success()
    assert result.disposition is Mp4R7QualificationDisposition.QUALIFIED
    assert len(result.human_authority_continuity) == 2
    decision_phase, continuation_phase = result.human_authority_continuity
    assert decision_phase.human_request_id == continuation_phase.human_request_id
    assert decision_phase.approver_user_id == continuation_phase.approver_user_id
    assert decision_phase.approver_tenant_id == continuation_phase.approver_tenant_id
    assert result.human_request_id == decision_phase.human_request_id
    assert result.decision_final_stage is DecisionLifecycleStage.TERMINAL
    assert qualification_approver_factory_call_count() == 1
    assert "hr_mp4r7" not in (result.human_request_id or "")


@pytest.mark.asyncio
async def test_mp4r7_success_e2e() -> None:
    composition = open_mp4r7_enterprise_integration_composition()
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(composition)
    result = await executor.run_success()
    assert result.scenario_id is Mp4R7ScenarioId.SUCCESS
    assert result.disposition is Mp4R7QualificationDisposition.QUALIFIED
    assert result.governance_required_human is True
    assert result.continuation_result_state.value == "resumed"
    assert result.protected_operation_completed is True
    assert result.diagnostics is not None
    assert result.diagnostics.operation_outcome_check_status == "proven_pass"
    _assert_identity_continuity(result)


@pytest.mark.asyncio
async def test_mp4r7_human_reject_e2e() -> None:
    composition = open_mp4r7_enterprise_integration_composition()
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(composition)
    result = await executor.run_human_reject()
    assert result.disposition is Mp4R7QualificationDisposition.QUALIFIED
    assert result.protected_operation_completed is False
    assert result.continuation_result_state is not None
    assert result.continuation_result_state.value == "rejected"


@pytest.mark.asyncio
async def test_mp4r7_stale_proposal_fail_closed() -> None:
    composition = open_mp4r7_enterprise_integration_composition()
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(composition)
    result = await executor.run_stale_proposal()
    assert result.disposition is Mp4R7QualificationDisposition.FAIL_CLOSED


@pytest.mark.asyncio
async def test_mp4r7_cross_tenant_fail_closed() -> None:
    composition = open_mp4r7_enterprise_integration_composition()
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(composition)
    result = await executor.run_cross_tenant()
    assert result.disposition is Mp4R7QualificationDisposition.FAIL_CLOSED


@pytest.mark.asyncio
async def test_mp4r7_evidence_failure_preserves_primary() -> None:
    composition = open_mp4r7_enterprise_integration_composition()
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(composition)
    result = await executor.run_evidence_failure()
    assert result.primary_error_code == Mp4R7ProtectedOperationError.__name__
    assert result.secondary_evidence_error_code == "FunctionalEvidencePersistenceError"


@pytest.mark.asyncio
async def test_mp4r7_binding_idempotency() -> None:
    composition = open_mp4r7_enterprise_integration_composition()
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(composition)
    result = await executor.run_binding_idempotency()
    assert result.disposition is Mp4R7QualificationDisposition.QUALIFIED


@pytest.mark.asyncio
async def test_mp4r7_binding_idempotency_conflict() -> None:
    composition = open_mp4r7_enterprise_integration_composition()
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(composition)
    await executor.run_binding_conflict()


@pytest.mark.asyncio
async def test_mp4r7_process_restart_resume() -> None:
    composition = open_mp4r7_enterprise_integration_composition(durable_continuation=True)
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(composition)
    result = await executor.run_process_restart()
    assert result.disposition is Mp4R7QualificationDisposition.QUALIFIED
    _assert_identity_continuity(result)
