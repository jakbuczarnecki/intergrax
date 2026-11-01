# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.assess.deployment_safety import assess_rollback_eligibility
from testing_support.change_safety.assess.workflow_evolution import classify_inflight_definition_change
from testing_support.change_safety.contracts.enums import DeploySafetyVerdict, WorkflowEvolutionDecision

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_s7_dm04_durable_state_refuse() -> None:
    verdict, reason = assess_rollback_eligibility(durable_state_forward_only=True)
    assert verdict is DeploySafetyVerdict.REFUSE
    assert reason is not None


def test_s3_inflight_unsupported_without_migration() -> None:
    decision = classify_inflight_definition_change(has_inflight=True, migration_declared=False)
    assert decision is WorkflowEvolutionDecision.EXPLICITLY_UNSUPPORTED
