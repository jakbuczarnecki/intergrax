# © Artur Czarnecki. All rights reserved.

"""Tenant isolation across durable authorities."""

from __future__ import annotations

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptTransitionReason
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome

from tests.conformance.runtime.durability._helpers import admit, transport_ref
from tests.conformance.runtime.durability.provider_factories import (
    DurableAdmissionBacking,
    create_admission_dependencies,
)
from tests.conformance.runtime.durability.restart import fresh_admission_composition

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_tenant_isolation_for_identity_attempt_and_terminal(
    admission_backing: DurableAdmissionBacking,
) -> None:
    process_a = create_admission_dependencies(admission_backing)
    process_b = fresh_admission_composition(admission_backing)
    transport_a = transport_ref(tenant_id="tenant-a", task_id="shared-key")
    transport_b = transport_ref(tenant_id="tenant-b", task_id="shared-key")

    identity_a = admit(transport=transport_a, deps=process_a).identity
    identity_b = admit(transport=transport_b, deps=process_b).identity
    assert identity_a.task_id != identity_b.task_id
    assert identity_a.run_id != identity_b.run_id

    a2_a = process_a.attempt_lifecycle.transition_to_next_attempt(
        tenant_id=identity_a.tenant_id,
        run_id=identity_a.run_id,
        expected_attempt_id=identity_a.attempt_id,
        reason=AttemptTransitionReason.RETRY,
    ).active_attempt_id
    a2_b = process_b.attempt_lifecycle.transition_to_next_attempt(
        tenant_id=identity_b.tenant_id,
        run_id=identity_b.run_id,
        expected_attempt_id=identity_b.attempt_id,
        reason=AttemptTransitionReason.RETRY,
    ).active_attempt_id
    assert a2_a != a2_b

    process_a.execution_terminal.commit_terminal_outcome(
        tenant_id=identity_a.tenant_id,
        task_id=str(identity_a.task_id),
        run_id=identity_a.run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
    )
    process_b.execution_terminal.commit_terminal_outcome(
        tenant_id=identity_b.tenant_id,
        task_id=str(identity_b.task_id),
        run_id=identity_b.run_id,
        outcome=ExecutionTerminalOutcome.FAILED,
    )

    terminal_a = process_a.execution_terminal.get_terminal_record(
        tenant_id=identity_a.tenant_id,
        task_id=str(identity_a.task_id),
    )
    terminal_b = process_b.execution_terminal.get_terminal_record(
        tenant_id=identity_b.tenant_id,
        task_id=str(identity_b.task_id),
    )
    assert terminal_a is not None and terminal_a.outcome is ExecutionTerminalOutcome.COMPLETED
    assert terminal_b is not None and terminal_b.outcome is ExecutionTerminalOutcome.FAILED
