# © Artur Czarnecki. All rights reserved.

"""LLM-EXTERNAL-OPERATION-ADMISSION R1 — contract validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.external_operations import (
    ExternalOperationAttempt,
    ExternalOperationAttemptLifecycle,
    ExternalOperationAttemptTransitionError,
    ExternalOperationIntent,
    ExternalOperationType,
)
from intergrax.contracts.external_operations.intent import mint_external_operation_intent_id

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _intent(**overrides: object) -> ExternalOperationIntent:
    base = {
        "intent_id": mint_external_operation_intent_id(),
        "tenant_id": "tenant_a",
        "task_id": mint_task_id(),
        "operation_type": ExternalOperationType.LLM_PROVIDER_CALL,
        "target_resource": "openai:gpt-4:sync",
        "requested_by": "operator",
        "justification": "diagnostic replay",
        "created_at": datetime.now(timezone.utc),
    }
    base.update(overrides)
    return ExternalOperationIntent(**base)


def test_external_operation_contract_validation() -> None:
    intent = _intent()
    assert intent.operation_type is ExternalOperationType.LLM_PROVIDER_CALL
    with pytest.raises(ValidationError):
        _intent(justification="   ")


def test_operation_requires_admission_before_execution() -> None:
    attempt = ExternalOperationAttempt(
        operation_attempt_id="ext_op_attempt_" + "a" * 32,
        intent=_intent(),
        tenant_id="tenant_a",
        task_id=_intent().task_id,
    )
    assert attempt.lifecycle is ExternalOperationAttemptLifecycle.CREATED
    with pytest.raises(ExternalOperationAttemptTransitionError):
        attempt.transition(ExternalOperationAttemptLifecycle.SUCCEEDED)
    admitted = attempt.transition(ExternalOperationAttemptLifecycle.ADMITTED)
    executing = admitted.transition(ExternalOperationAttemptLifecycle.EXECUTING)
    done = executing.transition(ExternalOperationAttemptLifecycle.SUCCEEDED)
    assert done.lifecycle is ExternalOperationAttemptLifecycle.SUCCEEDED


def test_created_to_succeeded_forbidden_at_model_level() -> None:
    with pytest.raises(ValidationError):
        ExternalOperationAttempt(
            operation_attempt_id="ext_op_attempt_" + "b" * 32,
            intent=_intent(),
            tenant_id="tenant_a",
            task_id=_intent().task_id,
            lifecycle=ExternalOperationAttemptLifecycle.SUCCEEDED,
            terminal_at=datetime.now(timezone.utc),
        )
