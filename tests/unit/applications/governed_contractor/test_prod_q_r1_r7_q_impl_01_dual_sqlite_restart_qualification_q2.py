# © Artur Czarnecki. All rights reserved.

"""IMPL-01 Q2 — real SQLite GCA + real SQLite PIS restart qualification.

START_HEAD: e20d937987577fdf856baa8910d3ddaa7e03556c

Uses ``SQLiteGovernedExternalWorkHostStoreBackend``, ``SQLiteProviderInvocationStore``,
``GovernedExternalWorkOrchestrator`` and the production dispatch gate. No second store.

Restart cases close both backends and construct new instances on the same files.
Row removal and F02 corruption run only after that committed close, as an explicit
test fault injection. They are not evidence that production SQLite drops a
successfully committed record on its own.

Legacy PIS-only random invocation ids (``inv-…``) without a surviving GER or F02 cannot
be correlated to ``derive_governed_external_work_create_provider_invocation_id``. Strict
production admission against that history is unsafe: ordinary CREATE would mint a new
stable id and dispatch again. Required fail-closed rollout (not optional freeze debt):

1. Admit strict production only when every historical CREATE intent is bound by a
   surviving GER or F02 that records that exact ``invocation_id`` (existing
   ``_resolve_create_provider_invocation_id`` reuse), or
2. Refuse startup until a separately authorized reconciliation rebinds that id onto
   the canonical execution GER/F02, or quarantines the legacy intent under GR-7.
   Ordinary CREATE redispatch is not that reconciliation.

NOT EXECUTED — pytest=0 / uv run=0 until operator authorization.
One process, 60s external limit (pytest-timeout is not a project dependency):

uv run pytest -p no:xdist tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_impl_01_dual_sqlite_restart_qualification_q2.py -q
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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
    _PROVIDER,
    _TENANT,
    _WORKSPACE,
    _create_meta,
    _policy_bundle,
)
from external_contractor_adapter.side_effect_actions import ACTION_CREATE_EXTERNAL_WORK
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.lifecycle_states import GovernedExternalWorkHostState
from governed_contractor_application.host.orchestrator import GovernedExternalWorkOrchestrator
from governed_contractor_application.host.provider_invocation_lifecycle import (
    GovernedProviderInvocationDispatchGate,
)
from governed_contractor_application.host.sqlite_governed_external_work_host_store_bundle import (
    SQLiteGovernedExternalWorkHostStoreBackend,
)
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.governed_execution_result import GovernedExecutionResultV2
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.contracts.governed_external_work_provider_invocation_identity import (
    derive_governed_external_work_create_provider_invocation_id,
)
from intergrax.contracts.provider_invocation_store import ProviderInvocationIntentWriteResult
from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
    SQLiteProviderInvocationStore,
)
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 10, 18, 9, 0, 0, tzinfo=timezone.utc)
_CREATE_IDEMP = "idem-impl01-q2-create"
_GCA_EXECUTION_TABLES = (
    "gca_execution_results",
    "gca_execution_states",
    "gca_execution_events",
    "gca_create_follow_on",
)


@dataclass(frozen=True)
class _Identity:
    task_id: object
    run_id: object
    attempt_id: object
    execution_id: object

    @property
    def execution_key(self) -> str:
        return str(self.execution_id)


@dataclass(frozen=True)
class _Seed:
    gca_path: Path
    pis_path: Path
    identity: _Identity
    invocation_id: str


def _mint() -> _Identity:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    return _Identity(task_id, run_id, attempt_id, execution_id)


def _expected_invocation_id(identity: _Identity) -> str:
    return derive_governed_external_work_create_provider_invocation_id(
        tenant_id=_TENANT,
        principal_id=_PRINCIPAL,
        execution_id=identity.execution_key,
        provider_id=_PROVIDER,
        idempotency_key=_CREATE_IDEMP,
    )


def _open_pair(
    gca_path: Path,
    pis_path: Path,
) -> tuple[SQLiteGovernedExternalWorkHostStoreBackend, SQLiteProviderInvocationStore]:
    return (
        SQLiteGovernedExternalWorkHostStoreBackend(gca_path),
        SQLiteProviderInvocationStore(pis_path),
    )


def _close_pair(
    gca: SQLiteGovernedExternalWorkHostStoreBackend,
    pis: SQLiteProviderInvocationStore,
) -> None:
    gca.close()
    pis.close()


def _runtime(fake: DeterministicExternalWorkFake, identity: _Identity, gca, pis):
    bundle = gca.bundle
    cw = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    bridge = GovernedExternalWorkEnterpriseReliabilityBridge(
        admission_port=RecordingAdmissionPort(),
    )
    runtime = build_governed_external_work_production_runtime(
        fake,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        task_scope=StaticActiveTaskScope(identity.task_id),
        capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
        policy_bundle=_policy_bundle(),
        collaborative_work_repositories=cw,
        execution_store=bundle.execution_store,
        receipt_store=bundle.receipt_store,
        bundle_store=bundle.bundle_store,
        continuation_store=bundle.continuation_store,
        provider_invocation_store=pis,
        reliability_bridge=bridge,
        clock=lambda: _T0,
    )
    assert isinstance(runtime.orchestrator, GovernedExternalWorkOrchestrator)
    assert runtime.orchestrator._provider_invocation_store is pis
    assert runtime.orchestrator._execution_store is bundle.execution_store
    gate = runtime.adapter._invocation_dispatch
    assert isinstance(gate, GovernedProviderInvocationDispatchGate)
    assert gate.store is pis
    assert runtime.adapter._reliability_aware_invocation_dispatch is gate
    return runtime


def _create(runtime, identity: _Identity):
    with bound_gr3_active_execution(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=identity.execution_id,
        task_id=identity.task_id,
        governance_tenant_id=_TENANT,
        governance_workspace_id=_WORKSPACE,
        governance_principal_id=_PRINCIPAL,
    ):
        return runtime.orchestrator.create(
            task_id=str(identity.task_id),
            run_id=str(identity.run_id),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(
                str(identity.task_id),
                str(identity.run_id),
                idempotency_key=_CREATE_IDEMP,
            ),
            execution_id=identity.execution_id,
        )


def _table_count(path: Path, table: str) -> int:
    allowed = {
        "provider_invocation_intents",
        "provider_invocation_outcomes",
        "gca_execution_results",
        "gca_create_follow_on",
        "gca_execution_states",
    }
    if table not in allowed:
        raise AssertionError(table)
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    finally:
        conn.close()
    assert row is not None
    return int(row[0])


def _exec(path: Path, sql: str, params: tuple[object, ...] = ()) -> int:
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return int(cur.rowcount)
    finally:
        conn.close()


def _inject_controlled_gca_execution_loss(path: Path, execution_id: str) -> None:
    """Delete committed GCA execution rows after the backend is closed.

    The caller must already have observed those rows on a prior open. This
    fixture damages the file on purpose. It does not show spontaneous loss.
    """
    assert _table_count(path, "gca_execution_results") == 1
    assert _table_count(path, "gca_execution_states") == 1
    assert _table_count(path, "gca_create_follow_on") == 1
    for table in _GCA_EXECUTION_TABLES:
        _exec(path, f"DELETE FROM {table} WHERE execution_id = ?", (execution_id,))
    assert _table_count(path, "gca_execution_results") == 0
    assert _table_count(path, "gca_execution_states") == 0
    assert _table_count(path, "gca_create_follow_on") == 0


def _inject_controlled_pis_outcome_loss(path: Path) -> None:
    """Delete the committed PIS outcome row after the store is closed."""
    assert _table_count(path, "provider_invocation_outcomes") == 1
    assert _exec(path, "DELETE FROM provider_invocation_outcomes") == 1
    assert _table_count(path, "provider_invocation_outcomes") == 0
    assert _table_count(path, "provider_invocation_intents") == 1


def _inject_controlled_f02_removal(path: Path, execution_id: str) -> None:
    """Delete only the committed F02 row; GER and PIS stay on disk."""
    assert _table_count(path, "gca_create_follow_on") == 1
    assert _table_count(path, "gca_execution_results") == 1
    removed = _exec(
        path,
        "DELETE FROM gca_create_follow_on WHERE execution_id = ?",
        (execution_id,),
    )
    assert removed == 1
    assert _table_count(path, "gca_create_follow_on") == 0
    assert _table_count(path, "gca_execution_results") == 1


def _inject_controlled_f02_corruption(path: Path, execution_id: str) -> None:
    """Replace a committed F02 payload with unreadable text after close."""
    assert _table_count(path, "gca_create_follow_on") == 1
    updated = _exec(
        path,
        "UPDATE gca_create_follow_on SET payload = ? WHERE execution_id = ?",
        ("not-json", execution_id),
    )
    assert updated == 1


def _seed(tmp_path: Path, tag: str) -> _Seed:
    identity = _mint()
    gca_path = tmp_path / f"{tag}-gca.sqlite3"
    pis_path = tmp_path / f"{tag}-pis.sqlite3"
    fake = DeterministicExternalWorkFake()
    gca, pis = _open_pair(gca_path, pis_path)
    try:
        runtime = _runtime(fake, identity, gca, pis)
        step = _create(runtime, identity)
        assert fake.create_calls == 1
        assert step.governed_result is not None
        invocation_id = step.governed_result.provider_invocation.invocation_id
        assert invocation_id == _expected_invocation_id(identity)
        assert invocation_id.startswith("gca-pinv:")
        assert pis.get_invocation(invocation_id) is not None
        outcome = pis.get_outcome(invocation_id)
        assert outcome is not None
        assert outcome.invocation_id == invocation_id
        ger = gca.bundle.execution_store.get_result(identity.execution_key)
        assert isinstance(ger, GovernedExecutionResultV2)
        assert ger.action == ACTION_CREATE_EXTERNAL_WORK
        follow_on = gca.bundle.execution_store.get_create_follow_on_record(
            identity.execution_key,
        )
        assert follow_on is not None
        assert follow_on.create_provider_invocation_id == invocation_id
        assert gca.bundle.execution_store.get_state(identity.execution_key) is not None
    finally:
        _close_pair(gca, pis)
    assert gca_path.is_file() and pis_path.is_file()
    assert _table_count(pis_path, "provider_invocation_intents") == 1
    assert _table_count(pis_path, "provider_invocation_outcomes") == 1
    assert _table_count(gca_path, "gca_execution_results") == 1
    assert _table_count(gca_path, "gca_create_follow_on") == 1
    return _Seed(gca_path, pis_path, identity, invocation_id)


def _reopen(seed: _Seed, fake: DeterministicExternalWorkFake):
    gca, pis = _open_pair(seed.gca_path, seed.pis_path)
    runtime = _runtime(fake, seed.identity, gca, pis)
    return gca, pis, runtime


def test_q2_p01_first_create_persists_pis_intent_outcome_and_gca_records(
    tmp_path: Path,
) -> None:
    seed = _seed(tmp_path, "q2-p01")
    gca, pis = _open_pair(seed.gca_path, seed.pis_path)
    try:
        assert type(gca) is SQLiteGovernedExternalWorkHostStoreBackend
        assert type(pis) is SQLiteProviderInvocationStore
        assert pis.get_invocation(seed.invocation_id) is not None
        assert pis.get_outcome(seed.invocation_id) is not None
        ger = gca.bundle.execution_store.get_result(seed.identity.execution_key)
        assert isinstance(ger, GovernedExecutionResultV2)
        follow_on = gca.bundle.execution_store.get_create_follow_on_record(
            seed.identity.execution_key,
        )
        assert follow_on is not None
        assert follow_on.create_provider_invocation_id == seed.invocation_id
    finally:
        _close_pair(gca, pis)


def test_q2_p02_reopen_both_backends_retry_create_zero_additional_provider_calls(
    tmp_path: Path,
) -> None:
    seed = _seed(tmp_path, "q2-p02")
    gca_a, pis_a = _open_pair(seed.gca_path, seed.pis_path)
    _close_pair(gca_a, pis_a)
    fake = DeterministicExternalWorkFake()
    gca_b, pis_b, runtime = _reopen(seed, fake)
    try:
        assert gca_b is not gca_a
        assert pis_b is not pis_a
        assert type(gca_b) is SQLiteGovernedExternalWorkHostStoreBackend
        assert type(pis_b) is SQLiteProviderInvocationStore
        second = _create(runtime, seed.identity)
        assert fake.create_calls == 0
        assert second.reason == "create_redispatch_prohibited"
        assert second.governed_result is not None
        assert second.governed_result.provider_invocation.invocation_id == seed.invocation_id
        assert pis_b.get_outcome(seed.invocation_id) is not None
    finally:
        _close_pair(gca_b, pis_b)


def test_q2_n01_remove_gca_execution_records_retain_pis_no_second_provider_call(
    tmp_path: Path,
) -> None:
    seed = _seed(tmp_path, "q2-n01")
    _inject_controlled_gca_execution_loss(seed.gca_path, seed.identity.execution_key)
    assert _table_count(seed.pis_path, "provider_invocation_intents") == 1
    assert _table_count(seed.pis_path, "provider_invocation_outcomes") == 1
    fake = DeterministicExternalWorkFake()
    gca, pis, runtime = _reopen(seed, fake)
    try:
        second = _create(runtime, seed.identity)
        assert fake.create_calls == 0
        assert second.reason == "create_duplicate_admission_pis_outcome"
        assert second.governed_result is None
        survived = pis.get_outcome(seed.invocation_id)
        assert survived is not None
        assert survived.status is ProviderInvocationStatus.SUCCEEDED
        # SUCCEEDED has no host-success projection on this path; the step stays
        # UNKNOWN while the committed PIS outcome row remains SUCCEEDED.
        assert second.state is GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN
    finally:
        _close_pair(gca, pis)
    assert _table_count(seed.pis_path, "provider_invocation_intents") == 1


def test_q2_n02_pis_intent_without_outcome_denies_ordinary_redispatch(
    tmp_path: Path,
) -> None:
    seed = _seed(tmp_path, "q2-n02")
    _inject_controlled_pis_outcome_loss(seed.pis_path)
    _inject_controlled_gca_execution_loss(seed.gca_path, seed.identity.execution_key)
    fake = DeterministicExternalWorkFake()
    gca, pis, runtime = _reopen(seed, fake)
    try:
        assert pis.get_invocation(seed.invocation_id) is not None
        assert pis.get_outcome(seed.invocation_id) is None
        second = _create(runtime, seed.identity)
        assert fake.create_calls == 0
        assert second.reason == "create_invocation_intent_without_persisted_outcome"
        assert second.state is GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN
        assert second.governed_result is None
        assert pis.get_outcome(seed.invocation_id) is None
    finally:
        _close_pair(gca, pis)


def test_q2_n03_ger_without_f02_denies_follow_on_and_create_redispatch(
    tmp_path: Path,
) -> None:
    seed = _seed(tmp_path, "q2-n03")
    _inject_controlled_f02_removal(seed.gca_path, seed.identity.execution_key)
    fake = DeterministicExternalWorkFake()
    gca, pis, runtime = _reopen(seed, fake)
    try:
        assert gca.bundle.execution_store.get_result(seed.identity.execution_key) is not None
        assert (
            gca.bundle.execution_store.get_create_follow_on_record(seed.identity.execution_key)
            is None
        )
        follow_on = runtime.orchestrator.resolve_create_adapter_result_for_follow_on(
            seed.identity.execution_key,
            tenant_id=_TENANT,
            principal_id=_PRINCIPAL,
        )
        assert follow_on is None
        second = _create(runtime, seed.identity)
        assert fake.create_calls == 0
        assert second.reason == "create_follow_on_record_missing_redispatch_prohibited"
        assert second.state is GovernedExternalWorkHostState.HOST_FINALIZATION_INCOMPLETE
        assert second.governed_result is not None
        assert second.governed_result.provider_invocation.invocation_id == seed.invocation_id
    finally:
        _close_pair(gca, pis)


def test_q2_n04_corrupt_f02_fail_closed(tmp_path: Path) -> None:
    seed = _seed(tmp_path, "q2-n04")
    _inject_controlled_f02_corruption(seed.gca_path, seed.identity.execution_key)
    fake = DeterministicExternalWorkFake()
    gca, pis, runtime = _reopen(seed, fake)
    try:
        assert (
            runtime.orchestrator.resolve_create_adapter_result_for_follow_on(
                seed.identity.execution_key,
                tenant_id=_TENANT,
                principal_id=_PRINCIPAL,
            )
            is None
        )
        with pytest.raises(ValueError, match="corrupted_create_follow_on"):
            _create(runtime, seed.identity)
        assert fake.create_calls == 0
        assert pis.get_outcome(seed.invocation_id) is not None
    finally:
        _close_pair(gca, pis)


def test_q2_n05_reopen_equivalent_create_stable_id_no_new_pis_intent(
    tmp_path: Path,
) -> None:
    seed = _seed(tmp_path, "q2-n05")
    assert _table_count(seed.pis_path, "provider_invocation_intents") == 1
    fake = DeterministicExternalWorkFake()
    gca_a, pis_a = _open_pair(seed.gca_path, seed.pis_path)
    prior = pis_a.get_invocation(seed.invocation_id)
    _close_pair(gca_a, pis_a)
    assert prior is not None
    gca_b, pis_b, runtime = _reopen(seed, fake)
    try:
        assert gca_b is not gca_a and pis_b is not pis_a
        second = _create(runtime, seed.identity)
        assert fake.create_calls == 0
        assert second.governed_result is not None
        assert (
            second.governed_result.provider_invocation.invocation_id
            == seed.invocation_id
            == _expected_invocation_id(seed.identity)
        )
        assert pis_b.get_invocation(seed.invocation_id) == prior
        assert (
            pis_b.insert_invocation_if_absent(prior)
            is ProviderInvocationIntentWriteResult.EQUIVALENT_NOOP
        )
    finally:
        _close_pair(gca_b, pis_b)
    assert _table_count(seed.pis_path, "provider_invocation_intents") == 1
