# © Artur Czarnecki. All rights reserved.

import pytest
from datetime import datetime, timedelta, timezone

from intergrax.contracts.agent_decision import (
    AgentDecisionType,
    HumanRequest,
    HumanRequestUrgency,
    human_request_fields_from_payload,
)
from intergrax.runtime.human.request_contract import (
    HumanTimeoutCoordinator,
    compute_expires_at_utc,
    human_request_event_payload,
)
from intergrax.runtime.task.task import Task


@pytest.mark.unit
@pytest.mark.gate
def test_human_request_normalizes_legacy_urgency_string():
    request = HumanRequest(
        request_id="hr_1",
        prompt="Review?",
        urgency="HIGH",
    )
    assert request.urgency == HumanRequestUrgency.HIGH
    assert request.schema_version == "human_request.v2"


@pytest.mark.unit
@pytest.mark.gate
def test_human_request_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout_seconds"):
        HumanRequest(
            request_id="hr_1",
            prompt="Review?",
            timeout_seconds=0,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_human_request_requires_timeout_for_default_on_timeout():
    with pytest.raises(ValueError, match="requires timeout_seconds"):
        HumanRequest(
            request_id="hr_1",
            prompt="Review?",
            default_on_timeout=AgentDecisionType.FAIL,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_human_request_rejects_invalid_default_on_timeout():
    with pytest.raises(ValueError, match="default_on_timeout"):
        HumanRequest(
            request_id="hr_1",
            prompt="Review?",
            timeout_seconds=60,
            default_on_timeout=AgentDecisionType.COMPLETE,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_human_request_event_payload_includes_expires_at():
    created = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    request = HumanRequest(
        request_id="hr_1",
        prompt="Review?",
        urgency=HumanRequestUrgency.CRITICAL,
        timeout_seconds=300,
        default_on_timeout=AgentDecisionType.ESCALATE,
    )
    payload = human_request_event_payload(
        request,
        created_at_utc=created.isoformat(),
    )
    assert payload["urgency"] == "critical"
    assert payload["timeout_seconds"] == 300
    assert payload["default_on_timeout"] == "escalate"
    assert payload["expires_at_utc"] == compute_expires_at_utc(
        created_at=created,
        timeout_seconds=300,
    )


@pytest.mark.unit
@pytest.mark.gate
def test_human_request_from_decision_payload_extracts_v2_fields():
    fields = human_request_fields_from_payload(
        {
            "urgency": "high",
            "timeout_seconds": 120,
            "default_on_timeout": "fail",
        }
    )
    assert fields["urgency"] == "high"
    assert fields["timeout_seconds"] == 120
    assert fields["default_on_timeout"] == AgentDecisionType.FAIL


@pytest.mark.unit
@pytest.mark.gate
def test_human_timeout_coordinator_attaches_deadline_to_task():
    task = Task(tenant_id="t1", user_id="u1", message="x")
    request = HumanRequest(
        request_id="hr_deadline",
        prompt="Approve vendor onboarding?",
        urgency=HumanRequestUrgency.HIGH,
        timeout_seconds=900,
        default_on_timeout=AgentDecisionType.FAIL,
    )
    HumanTimeoutCoordinator.attach_to_task(task, request)

    assert task.runtime.governance.human_request is not None
    assert task.runtime.governance.human_request_expires_at is not None
    assert task.metadata.get("human_request_expires_at") is not None
    assert HumanTimeoutCoordinator.planned_timeout_action(task) == AgentDecisionType.FAIL
    assert HumanTimeoutCoordinator.is_expired(task) is False

    expires = datetime.fromisoformat(task.runtime.governance.human_request_expires_at)
    assert expires > datetime.now(timezone.utc)
