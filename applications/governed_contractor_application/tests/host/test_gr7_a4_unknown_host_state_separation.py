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


def test_unknown_host_state_from_persisted_outcome_single_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from governed_contractor_application.host import provider_invocation_lifecycle as pil

    classify_calls = 0
    original = pil.classify_provider_invocation_status

    def counting_classify(**kwargs: object) -> ProviderInvocationStatus | None:
        nonlocal classify_calls
        classify_calls += 1
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pil, "classify_provider_invocation_status", counting_classify)

    from applications.governed_contractor_application.tests.host.test_gr7_a3_durable_provider_invocation import (
        _runtime_with_store,
    )
    from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
        RecordingAdmissionPort,
        _create_step,
        _UNCERTAIN_IDEMP,
    )
    from external_contractor_adapter.tests.fakes.deterministic_external_work import (
        DeterministicExternalWorkFake,
    )
    from intergrax.contracts.external_work import ExternalWorkErrorCode
    from tests.unit.runtime.governance.gr3_test_support import default_gr3_identity_bundle

    fake = DeterministicExternalWorkFake(
        fail_create_with_code={
            _UNCERTAIN_IDEMP: ExternalWorkErrorCode.PROVIDER_OUTCOME_UNCERTAIN,
        },
    )
    store = DurableTestProviderInvocationStore()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime_with_store(fake, task_id, store)
    step, _ = _create_step(
        runtime,
        fake,
        task_id,
        run_id,
        attempt_id,
        execution_id,
        idempotency_key=_UNCERTAIN_IDEMP,
    )
    inv_id = next(iter(store._invocations))  # noqa: SLF001
    assert store.get_outcome(inv_id).status is ProviderInvocationStatus.UNKNOWN
    assert step.state is GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN
    assert classify_calls == 1


def test_failed_outcome_persisted_before_execution_failed_host_state() -> None:
    from applications.governed_contractor_application.tests.host.test_gr7_a3_durable_provider_invocation import (
        _runtime_with_store,
    )
    from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
        _create_step,
        _FAIL_IDEMP,
    )
    from external_contractor_adapter.tests.fakes.deterministic_external_work import (
        DeterministicExternalWorkFake,
    )
    from intergrax.contracts.external_work import ExternalWorkErrorCode
    from tests.unit.runtime.governance.gr3_test_support import default_gr3_identity_bundle

    fake = DeterministicExternalWorkFake(
        fail_create_with_code={_FAIL_IDEMP: ExternalWorkErrorCode.PERMANENT_PROVIDER_FAILURE},
    )
    store = DurableTestProviderInvocationStore()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime_with_store(fake, task_id, store)
    step, _ = _create_step(
        runtime,
        fake,
        task_id,
        run_id,
        attempt_id,
        execution_id,
        idempotency_key=_FAIL_IDEMP,
    )
    inv_id = next(iter(store._invocations))  # noqa: SLF001
    assert store.get_outcome(inv_id).status is ProviderInvocationStatus.FAILED
    assert step.state is GovernedExternalWorkHostState.EXECUTION_FAILED


def test_unknown_erl_admission_exactly_once() -> None:
    from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
        RecordingAdmissionPort,
        _build_runtime,
        _create_step,
        _UNCERTAIN_IDEMP,
    )
    from external_contractor_adapter.tests.fakes.deterministic_external_work import (
        DeterministicExternalWorkFake,
    )
    from intergrax.contracts.external_work import ExternalWorkErrorCode
    from tests.unit.runtime.governance.gr3_test_support import default_gr3_identity_bundle

    fake = DeterministicExternalWorkFake(
        fail_create_with_code={
            _UNCERTAIN_IDEMP: ExternalWorkErrorCode.PROVIDER_OUTCOME_UNCERTAIN,
        },
    )
    port = RecordingAdmissionPort()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, port = _build_runtime(fake, task_id, admission_port=port)
    step, _ = _create_step(
        runtime,
        fake,
        task_id,
        run_id,
        attempt_id,
        execution_id,
        idempotency_key=_UNCERTAIN_IDEMP,
    )
    assert step.state is GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN
    assert len(port.requests) == 1


def test_outcome_persistence_failure_not_execution_outcome_unknown() -> None:
    from applications.governed_contractor_application.tests.host.test_gr7_a3_durable_provider_invocation import (
        RecordingProviderInvocationStore,
        _runtime_with_store,
    )
    from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
        _create_step,
    )
    from external_contractor_adapter.tests.fakes.deterministic_external_work import (
        DeterministicExternalWorkFake,
    )
    from tests.unit.runtime.governance.gr3_test_support import default_gr3_identity_bundle

    fake = DeterministicExternalWorkFake()
    store = RecordingProviderInvocationStore(fail_outcome=True)
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime_with_store(fake, task_id, store)
    step, fake_ref = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert fake_ref.create_calls == 1
    assert step.reason == "provider_invocation_outcome_persistence_failed"
    assert step.state is GovernedExternalWorkHostState.EXECUTION_FAILED
    assert step.state is not GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN
    assert store.outcomes == {}
