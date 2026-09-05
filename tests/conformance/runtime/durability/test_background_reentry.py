# © Artur Czarnecki. All rights reserved.

"""P0C-7 — background re-entry durability across process restart."""

from __future__ import annotations

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptTransitionReason
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.runtime.background_execution.reentry_admission import (
    BackgroundExecutionReentryDisposition,
)

from tests.conformance.runtime.durability._helpers import admit, transport_ref
from tests.conformance.runtime.durability.provider_factories import (
    DurableAdmissionBacking,
    create_admission_dependencies,
)
from tests.conformance.runtime.durability.restart import fresh_admission_composition

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize(
    "outcome",
    [
        ExecutionTerminalOutcome.COMPLETED,
        ExecutionTerminalOutcome.FAILED,
        ExecutionTerminalOutcome.CANCELLED,
    ],
)
def test_terminal_redelivery_skips_handler_after_restart(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
    outcome: ExecutionTerminalOutcome,
) -> None:
    process_a = create_admission_dependencies(admission_backing)
    transport = transport_ref(
        tenant_id=tenant_id,
        task_id=f"terminal-redelivery-{outcome.value}",
    )
    first = admit(transport=transport, deps=process_a)
    process_a.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=outcome,
    )

    process_b = fresh_admission_composition(admission_backing)
    redelivery = admit(transport=transport, deps=process_b)
    assert redelivery.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED


def test_a2_with_cancelled_terminal_blocks_without_a3_after_restart(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
) -> None:
    process_a = create_admission_dependencies(admission_backing)
    transport = transport_ref(tenant_id=tenant_id, task_id="a2-cancelled")
    first = admit(transport=transport, deps=process_a)
    a2 = process_a.attempt_lifecycle.transition_to_next_attempt(
        tenant_id=first.identity.tenant_id,
        run_id=first.identity.run_id,
        expected_attempt_id=first.identity.attempt_id,
        reason=AttemptTransitionReason.RETRY,
    ).active_attempt_id
    process_a.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=ExecutionTerminalOutcome.CANCELLED,
    )

    process_b = fresh_admission_composition(admission_backing)
    redelivery = admit(transport=transport, deps=process_b)
    assert redelivery.identity.attempt_id == a2
    assert redelivery.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED
