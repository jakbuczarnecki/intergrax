# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-IMPL-01 — stable CREATE invocation identity (orchestrator + PIS gate path)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Final

import pytest

from applications.governed_contractor_application.host.external_work_enterprise_reliability_bridge import (
    GovernedExternalWorkEnterpriseReliabilityBridge,
)
from applications.governed_contractor_application.host.production_external_work_composition import (
    build_governed_external_work_production_runtime,
)
from applications.governed_contractor_application.tests.host.gr6_collaborative_work_test_support import (
    gr6_seeded_collaborative_work_repositories,
)
from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
    RecordingAdmissionPort,
    _PRINCIPAL,
    _TENANT,
    _WORKSPACE,
    _create_meta,
    _policy_bundle,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.lifecycle_states import GovernedExternalWorkHostState
from governed_contractor_application.host.stores import (
    InMemoryContinuationStateStore,
    InMemoryGovernedExecutionStore,
    InMemoryPolicyBundleArtifactStore,
    InMemoryProofReceiptStore,
)
from intergrax.contracts.governed_external_work_provider_invocation_identity import (
    derive_governed_external_work_create_provider_invocation_id,
)
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from external_contractor_adapter.side_effect_actions import ACTION_CREATE_EXTERNAL_WORK
from intergrax.contracts.provider_invocation import ProviderInvocation
from intergrax.contracts.provider_invocation_store import (
    ProviderInvocationIntentWriteResult,
)
from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
    SQLiteProviderInvocationStore,
    open_provider_invocation_store_at,
)
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0: Final = datetime(2026, 10, 18, 8, 0, 0, tzinfo=timezone.utc)
_PROVIDER: Final = "gec3_deterministic_fake"
_CREATE_IDEMP: Final = "idem-impl01-create"
_OTHER_IDEMP: Final = "idem-impl01-other-create"

# NOT PROVEN — pytest=0 until operator authorization (one foreground process, -p no:xdist):
# uv run pytest -p no:xdist tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_impl_01_stable_create_invocation_identity.py -q


def _build_runtime(
    fake: DeterministicExternalWorkFake,
    task_id: object,
    *,
    pis: SQLiteProviderInvocationStore,
    execution_store: InMemoryGovernedExecutionStore | None = None,
):
    cw = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    bridge = GovernedExternalWorkEnterpriseReliabilityBridge(
        admission_port=RecordingAdmissionPort(),
    )
    return build_governed_external_work_production_runtime(
        fake,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        task_scope=StaticActiveTaskScope(task_id),
        capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
        policy_bundle=_policy_bundle(),
        collaborative_work_repositories=cw,
        execution_store=execution_store or InMemoryGovernedExecutionStore(),
        receipt_store=InMemoryProofReceiptStore(),
        bundle_store=InMemoryPolicyBundleArtifactStore(),
        continuation_store=InMemoryContinuationStateStore(),
        provider_invocation_store=pis,
        reliability_bridge=bridge,
        clock=lambda: _T0,
    )


def _orchestrator_create(
    runtime,
    *,
    task_id,
    run_id,
    attempt_id,
    execution_id,
    idempotency_key: str = _CREATE_IDEMP,
):
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        task_id=task_id,
        governance_tenant_id=_TENANT,
        governance_workspace_id=_WORKSPACE,
        governance_principal_id=_PRINCIPAL,
    ):
        return runtime.orchestrator.create(
            task_id=str(task_id),
            run_id=str(run_id),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id), idempotency_key=idempotency_key),
            execution_id=execution_id,
        )


def test_impl01_orchestrator_create_mints_stable_invocation_id(tmp_path: Path) -> None:
    pis = open_provider_invocation_store_at(tmp_path / "pis.db")
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _build_runtime(fake, task_id, pis=pis)
    expected = derive_governed_external_work_create_provider_invocation_id(
        tenant_id=_TENANT,
        principal_id=_PRINCIPAL,
        execution_id=str(execution_id),
        provider_id=_PROVIDER,
        idempotency_key=_CREATE_IDEMP,
    )
    step = _orchestrator_create(
        runtime,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert fake.create_calls == 1
    assert step.governed_result is not None
    assert step.governed_result.provider_invocation.invocation_id == expected
    assert step.governed_result.provider_invocation.invocation_id.startswith("gca-pinv:")


def test_impl01_same_create_retry_no_second_provider_dispatch(tmp_path: Path) -> None:
    pis = open_provider_invocation_store_at(tmp_path / "pis-retry.db")
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    execution_store = InMemoryGovernedExecutionStore()
    runtime = _build_runtime(
        fake,
        task_id,
        pis=pis,
        execution_store=execution_store,
    )
    first = _orchestrator_create(
        runtime,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert fake.create_calls == 1
    inv_id = first.governed_result.provider_invocation.invocation_id
    intent_after_first = pis.get_invocation(inv_id)
    assert intent_after_first is not None
    outcome_after_first = pis.get_outcome(inv_id)

    execution_store._states.clear()
    execution_store._results.clear()
    execution_store._follow_on.clear()

    second = _orchestrator_create(
        runtime,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert fake.create_calls == 1
    assert second.governed_result is None
    assert second.state is GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN
    assert pis.get_invocation(inv_id) == intent_after_first
    assert (
        pis.insert_invocation_if_absent(intent_after_first)
        is ProviderInvocationIntentWriteResult.EQUIVALENT_NOOP
    )
    assert pis.get_outcome(inv_id) == outcome_after_first


def test_impl01_sqlite_pis_restart_same_create_no_second_dispatch(tmp_path: Path) -> None:
    db = tmp_path / "pis-restart.db"
    pis_a = open_provider_invocation_store_at(db)
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime_a = _build_runtime(fake, task_id, pis=pis_a)
    first = _orchestrator_create(
        runtime_a,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    inv_id = first.governed_result.provider_invocation.invocation_id
    assert fake.create_calls == 1

    pis_b = open_provider_invocation_store_at(db)
    runtime_b = _build_runtime(fake, task_id, pis=pis_b)
    second = _orchestrator_create(
        runtime_b,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert fake.create_calls == 1
    assert second.state is GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN
    assert pis_b.get_invocation(inv_id) is not None


def test_impl01_different_execution_id_admits_second_create(tmp_path: Path) -> None:
    pis = open_provider_invocation_store_at(tmp_path / "pis-new-exec.db")
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id_a = default_gr3_identity_bundle()
    from intergrax.contracts.execution_identity import mint_execution_id

    execution_id_b = mint_execution_id()
    runtime = _build_runtime(fake, task_id, pis=pis)
    _orchestrator_create(
        runtime,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id_a,
    )
    _orchestrator_create(
        runtime,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id_b,
        idempotency_key=_OTHER_IDEMP,
    )
    assert fake.create_calls == 2


def test_impl01_legacy_random_invocation_id_preserved_when_ger_survives(tmp_path: Path) -> None:
    pis = open_provider_invocation_store_at(tmp_path / "pis-legacy.db")
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    execution_store = InMemoryGovernedExecutionStore()
    runtime = _build_runtime(
        fake,
        task_id,
        pis=pis,
        execution_store=execution_store,
    )
    legacy_id = "inv-legacy-random-abcdef0123456789"
    legacy_inv = ProviderInvocation(
        invocation_id=legacy_id,
        provider_id=_PROVIDER,
        operation="create_work",
        task_id=str(task_id),
        run_id=str(run_id),
        idempotency_key=_CREATE_IDEMP,
        request_digest="sha256:" + ("aa" * 32),
        started_at=_T0,
    )
    pis.insert_invocation_if_absent(legacy_inv)

    from tests.unit.contracts.test_governed_execution_result import (
        _ger_v2,
        _outcome,
        _proof,
    )

    ger = _ger_v2(
        execution_id=str(execution_id),
        task_id=str(task_id),
        run_id=str(run_id),
        attempt_id=str(attempt_id),
        tenant_id=_TENANT,
        principal_id=_PRINCIPAL,
        idempotency_key=_CREATE_IDEMP,
        provider_invocation=legacy_inv,
        provider_outcome=_outcome(invocation_id=legacy_id),
        proof=_proof(
            task_id=str(task_id),
            run_id=str(run_id),
            tenant_id=_TENANT,
            principal_id=_PRINCIPAL,
            provider_id=_PROVIDER,
            idempotency_key=_CREATE_IDEMP,
            action=ACTION_CREATE_EXTERNAL_WORK,
        ),
    )
    execution_store.put_result(ger)

    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        task_id=task_id,
        governance_tenant_id=_TENANT,
        governance_workspace_id=_WORKSPACE,
        governance_principal_id=_PRINCIPAL,
    ):
        invocation = runtime.orchestrator._new_invocation(
            action=ACTION_CREATE_EXTERNAL_WORK,
            task_id=str(task_id),
            run_id=str(run_id),
            metadata=_create_meta(str(task_id), str(run_id), idempotency_key=_CREATE_IDEMP),
            started_at=_T0,
            execution_id=str(execution_id),
            tenant_id=_TENANT,
            principal_id=_PRINCIPAL,
        )

    assert invocation.invocation_id == legacy_id
    assert fake.create_calls == 0
