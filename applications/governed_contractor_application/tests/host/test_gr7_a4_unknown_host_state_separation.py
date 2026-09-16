# © Artur Czarnecki. All rights reserved.

"""GR-7-A4 — durable UNKNOWN host/runtime state separation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from governed_contractor_application.host.lifecycle_states import (
    GovernedExternalWorkHostState,
    map_provider_invocation_outcome_status_to_host_state,
)
from governed_contractor_application.host.stores import InMemoryGovernedExecutionStore
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from applications.governed_contractor_application.tests.host.durable_provider_invocation_test_store import (
    DurableTestProviderInvocationStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)


def test_map_provider_invocation_outcome_status_to_host_state() -> None:
    assert (
        map_provider_invocation_outcome_status_to_host_state(
            ProviderInvocationStatus.SUCCEEDED,
        )
        is None
    )
    assert (
        map_provider_invocation_outcome_status_to_host_state(
            ProviderInvocationStatus.FAILED,
        )
        is GovernedExternalWorkHostState.EXECUTION_FAILED
    )
    assert (
        map_provider_invocation_outcome_status_to_host_state(
            ProviderInvocationStatus.UNKNOWN,
        )
        is GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN
    )


def test_execution_outcome_unknown_host_state_roundtrip() -> None:
    store = InMemoryGovernedExecutionStore()
    store.put_state(
        "exec-gr7a4",
        GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN,
    )
    assert (
        store.get_state("exec-gr7a4")
        is GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN
    )


def test_crash_ambiguity_intent_without_outcome_not_explicit_unknown() -> None:
    store = DurableTestProviderInvocationStore()
    inv = ProviderInvocation(
        invocation_id="inv-crash-ambiguity",
        provider_id="gec3_deterministic_fake",
        operation="create_work",
        task_id="t1",
        run_id="r1",
        request_digest="sha256:" + ("cd" * 32),
        started_at=_T0,
    )
    store.put_invocation(inv)
    assert store.get_outcome(inv.invocation_id) is None
    store.put_outcome(
        ProviderInvocationOutcome(
            invocation_id=inv.invocation_id,
            status=ProviderInvocationStatus.UNKNOWN,
            completed_at=_T0,
            response_digest="sha256:" + ("ef" * 32),
        ),
    )
    assert (
        store.get_outcome(inv.invocation_id).status is ProviderInvocationStatus.UNKNOWN
    )
