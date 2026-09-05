# © Artur Czarnecki. All rights reserved.

"""P0C-4 + P0C-7 — retry attempt continuity across process restart."""

from __future__ import annotations

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptTransitionReason
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


def test_retry_attempt_survives_restart(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
) -> None:
    process_a = create_admission_dependencies(admission_backing)
    transport = transport_ref(tenant_id=tenant_id, task_id="retry-a2")
    first = admit(transport=transport, deps=process_a)
    bootstrap_a1 = first.identity.attempt_id
    transition = process_a.attempt_lifecycle.transition_to_next_attempt(
        tenant_id=first.identity.tenant_id,
        run_id=first.identity.run_id,
        expected_attempt_id=bootstrap_a1,
        reason=AttemptTransitionReason.RETRY,
    )

    process_b = fresh_admission_composition(admission_backing)
    recovered = admit(transport=transport, deps=process_b)

    assert recovered.identity.attempt_id == transition.active_attempt_id
    assert recovered.identity.attempt_id != bootstrap_a1
    assert recovered.disposition is BackgroundExecutionReentryDisposition.EXECUTE


def test_multiple_redelivery_after_restart_returns_a2(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
) -> None:
    process_a = create_admission_dependencies(admission_backing)
    transport = transport_ref(tenant_id=tenant_id, task_id="retry-redelivery")
    first = admit(transport=transport, deps=process_a)
    a2 = process_a.attempt_lifecycle.transition_to_next_attempt(
        tenant_id=first.identity.tenant_id,
        run_id=first.identity.run_id,
        expected_attempt_id=first.identity.attempt_id,
        reason=AttemptTransitionReason.RETRY,
    ).active_attempt_id

    process_b = fresh_admission_composition(admission_backing)
    attempts = [
        admit(transport=transport, deps=process_b).identity.attempt_id
        for _ in range(3)
    ]

    assert attempts == [a2, a2, a2]
    assert a2 != first.identity.attempt_id
