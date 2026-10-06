# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P3-R1-R1 — canonical task/tenant identity gates on governed host."""

from __future__ import annotations

import pytest

from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
    _PRINCIPAL,
    _TENANT,
    _WORKSPACE,
    _build_runtime,
    _create_meta,
    _create_step,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.lifecycle_states import GovernedExternalWorkHostState
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from tests.unit.runtime.governance.gr3_test_support import (
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_OTHER_TENANT = "other-tenant-r1r1"


def _orchestrator_create(
    runtime,
    fake: DeterministicExternalWorkFake,
    *,
    task_id,
    run_id,
    attempt_id,
    execution_id,
    tenant_id: str | None,
    bind_task: bool = True,
    bind_governance: bool = True,
    active_governance_tenant: str = _TENANT,
    requested_task_id: object | None = None,
):
    calls_before = fake.create_calls
    kwargs = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "execution_id": execution_id,
    }
    if bind_task:
        kwargs["task_id"] = task_id
    if bind_governance:
        kwargs["governance_tenant_id"] = active_governance_tenant
        kwargs["governance_workspace_id"] = _WORKSPACE
        kwargs["governance_principal_id"] = _PRINCIPAL
    create_task_id = requested_task_id if requested_task_id is not None else task_id
    with bound_gr3_active_execution(**kwargs):
        step = runtime.orchestrator.create(
            task_id=str(create_task_id),
            run_id=str(run_id),
            principal_id=_PRINCIPAL,
            tenant_id=tenant_id,
            metadata=_create_meta(str(task_id), str(run_id)),
            execution_id=execution_id,
        )
    return step, calls_before


def test_tid1_missing_active_task_fails_before_provider() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, _ = _build_runtime(fake, task_id)
    step, calls_before = _orchestrator_create(
        runtime,
        fake,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id=_TENANT,
        bind_task=False,
    )
    assert step.reason == "canonical_task_id_required"
    assert fake.create_calls == calls_before


def test_tid2_task_mismatch_fails_closed() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, _ = _build_runtime(fake, task_id)
    other_task = mint_task_id()
    step, calls_before = _orchestrator_create(
        runtime,
        fake,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id=_TENANT,
        requested_task_id=other_task,
    )
    assert step.reason == "task_id_mismatch"
    assert fake.create_calls == calls_before


def test_tid3_matching_task_reaches_provider() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, _ = _build_runtime(fake, task_id)
    step, _ = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert fake.create_calls == 1
    assert step.governed_result is not None
    assert str(step.governed_result.task_id) == str(task_id)


def test_ten1_none_tenant_fails_before_provider() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, _ = _build_runtime(fake, task_id)
    step, calls_before = _orchestrator_create(
        runtime,
        fake,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id=None,
    )
    assert step.reason == "tenant_id_required"
    assert fake.create_calls == calls_before


def test_ten2_blank_tenant_fails_before_provider() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, _ = _build_runtime(fake, task_id)
    step, calls_before = _orchestrator_create(
        runtime,
        fake,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id="   ",
    )
    assert step.reason == "tenant_id_required"
    assert fake.create_calls == calls_before


def test_ten3_tenant_mismatch_fails_before_provider() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, _ = _build_runtime(fake, task_id)
    step, calls_before = _orchestrator_create(
        runtime,
        fake,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id=_OTHER_TENANT,
    )
    assert step.reason == "tenant_id_mismatch"
    assert fake.create_calls == calls_before


def test_ten4_matching_tenant_passes_identity_gate() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, _ = _build_runtime(fake, task_id)
    step, _ = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert step.state is not GovernedExternalWorkHostState.EXECUTION_FAILED
    assert step.governed_result is not None
    assert step.governed_result.tenant_id == _TENANT


def test_e2e_tenant_continuity_through_receipt() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, _ = _build_runtime(fake, task_id)
    step, _ = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert step.governed_result is not None
    assert step.governed_result.tenant_id == _TENANT
    assert step.attestation is not None
    assert step.attestation.event is not None
    assert step.attestation.event.tenant_id == _TENANT


def test_cross_tenant_negative_no_provider_and_no_ger() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, _ = _build_runtime(fake, task_id)
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
        ),
    )
    try:
        with bound_gr3_active_execution(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            task_id=task_id,
        ):
            step = runtime.orchestrator.create(
                task_id=str(task_id),
                run_id=str(run_id),
                principal_id=_PRINCIPAL,
                tenant_id=_OTHER_TENANT,
                metadata=_create_meta(str(task_id), str(run_id)),
                execution_id=execution_id,
            )
    finally:
        reset_active_execution_governance_identity(gov_token)
    assert step.reason == "tenant_id_mismatch"
    assert fake.create_calls == 0
    assert step.governed_result is None
    assert step.receipt is None
