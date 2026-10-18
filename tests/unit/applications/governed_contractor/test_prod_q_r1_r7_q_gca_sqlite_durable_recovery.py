# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-IMPL-01 — SQLite GCA durable host bundle and partial-finalization tests."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timezone
from pathlib import Path
import sqlite3
from unittest.mock import patch

import pytest

from external_contractor_adapter.tests.fakes.adapter_test_wiring import (
    EXTERNAL_WORK_TEST_RUN_ID,
    EXTERNAL_WORK_TEST_TASK_ID,
    allow_adapter,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.offline_demo import build_demo_policy_bundle
from governed_contractor_application.host.provider_invocation_lifecycle import (
    GovernedProviderInvocationDispatchGate,
)
from governed_contractor_application.host.stores import (
    InMemoryProviderInvocationStore,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    reset_active_execution_identity,
    validate_task_id,
)
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)

from governed_contractor_application.host.governed_external_work_host_stores import (
    validate_production_governed_external_work_host_store_bundle,
)
from governed_contractor_application.host.lifecycle_states import (
    GovernedExternalWorkHostState,
)
from governed_contractor_application.host.orchestrator import (
    GovernedExternalWorkOrchestrator,
)
from governed_contractor_application.host.sqlite_governed_external_work_host_store_bundle import (
    GovernedExternalWorkHostPersistenceError,
    open_sqlite_governed_external_work_host_store_bundle_at,
)
from governed_contractor_application.host.stores import GovernedExecutionCreateFollowOnConflictError
from external_contractor_adapter.side_effect_actions import ACTION_CREATE_EXTERNAL_WORK
from intergrax.contracts.governed_execution_result import GovernedExecutionResultV2
from intergrax.contracts.governed_external_work_create_follow_on import (
    GovernedExternalWorkCreateFollowOnRecordV1,
)
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle
from tests.unit.contracts.test_governed_execution_result import _ger_v2

pytestmark = [pytest.mark.unit, pytest.mark.gate]

# NOT PROVEN at authoring time — resource budget pytest=0; execute only after operator approval:
# uv run pytest -p no:xdist tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_gca_sqlite_durable_recovery.py -q

_T0 = datetime(2026, 7, 21, 11, 0, 0, tzinfo=timezone.utc)
_TASK = str(EXTERNAL_WORK_TEST_TASK_ID)
_RUN = str(EXTERNAL_WORK_TEST_RUN_ID)


@contextmanager
def _active_execution(execution_id: str):
    token = bind_active_execution_identity(
        task_id=validate_task_id(_TASK),
        run_id=_RUN,
        execution_id=execution_id,
        attempt_id=mint_attempt_id(),
    )
    try:
        yield execution_id
    finally:
        reset_active_execution_identity(token)


def _sqlite_orchestrator(tmp_path: Path, db_name: str = "gca.db"):
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        tmp_path / db_name,
    )
    bundle = build_demo_policy_bundle(issued_at=_T0)
    policy = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0)
    fake = DeterministicExternalWorkFake()
    invocation_store = InMemoryProviderInvocationStore()
    adapter, _ = allow_adapter(
        fake,
        policy=policy,
        tenant_id="tenant-a",
        workspace_id="ws-a",
        principal_id="principal-a",
        active_task_id=_TASK,
        invocation_dispatch=GovernedProviderInvocationDispatchGate(
            store=invocation_store,
            clock=lambda: _T0,
        ),
    )
    orch = GovernedExternalWorkOrchestrator(
        adapter=adapter,
        policy=policy,
        bundle=bundle,
        attestor=None,
        capabilities=quote_first_partner_capability_fixture(provider_id="gec3_deterministic_fake"),
        execution_store=backend.bundle.execution_store,
        receipt_store=backend.bundle.receipt_store,
        bundle_store=backend.bundle.bundle_store,
        continuation_store=backend.bundle.continuation_store,
        provider_invocation_store=invocation_store,
        clock=lambda: _T0,
    )
    return orch, fake, backend


def _sample_bundle_record(execution_id: str) -> GovernedExternalWorkCreateFollowOnRecordV1:
    from intergrax.contracts.external_work import (
        ExternalTaskCorrelation,
        ExternalWorkSnapshot,
        ExternalWorkStatus,
    )

    now = datetime.now(UTC)
    snapshot = ExternalWorkSnapshot(
        correlation=ExternalTaskCorrelation(
            task_id="task-1",
            provider_id="prov",
            external_task_id="ew-1",
        ),
        status=ExternalWorkStatus.QUOTE_AVAILABLE,
        created_at=now,
        updated_at=now,
    )
    return GovernedExternalWorkCreateFollowOnRecordV1(
        execution_id=execution_id,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-a",
        principal_id="principal-a",
        create_action="external_work.create",
        create_provider_invocation_id="inv-1",
        snapshot=snapshot,
        provider="prov",
        snapshot_content_digest="digest",
        recorded_at=datetime.now(UTC),
    )


def test_sqlite_bundle_reports_durable_and_validates_for_production(
    tmp_path: Path,
) -> None:
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        tmp_path / "gca.db",
    )
    validate_production_governed_external_work_host_store_bundle(backend.bundle)
    assert backend.bundle.is_durable is True
    backend.close()


def test_sqlite_ger_round_trip_close_reopen_new_backend(tmp_path: Path) -> None:
    db = tmp_path / "gca.db"
    execution_id = "exec-ger-1"
    ger = _ger_v2(execution_id=execution_id, action=ACTION_CREATE_EXTERNAL_WORK)
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    backend.bundle.execution_store.put_result(ger)
    backend.close()
    reopened = open_sqlite_governed_external_work_host_store_bundle_at(db)
    loaded = reopened.bundle.execution_store.get_result(execution_id)
    assert loaded is not None
    assert str(loaded.execution_id) == execution_id
    assert loaded.action == ACTION_CREATE_EXTERNAL_WORK
    reopened.close()


def test_sqlite_host_state_round_trip_close_reopen_new_backend(tmp_path: Path) -> None:
    db = tmp_path / "gca.db"
    execution_id = "exec-state-1"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    backend.bundle.execution_store.put_state(
        execution_id,
        GovernedExternalWorkHostState.HOST_FINALIZATION_INCOMPLETE,
    )
    backend.close()
    reopened = open_sqlite_governed_external_work_host_store_bundle_at(db)
    assert (
        reopened.bundle.execution_store.get_state(execution_id)
        is GovernedExternalWorkHostState.HOST_FINALIZATION_INCOMPLETE
    )
    reopened.close()


def test_f02_persist_round_trip_close_reopen(tmp_path: Path) -> None:
    db = tmp_path / "gca.db"
    execution_id = "exec-f02-1"
    record = _sample_bundle_record(execution_id)
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    backend.bundle.execution_store.put_create_follow_on_record(execution_id, record)
    backend.close()
    reopened = open_sqlite_governed_external_work_host_store_bundle_at(db)
    loaded = reopened.bundle.execution_store.get_create_follow_on_record(execution_id)
    assert loaded is not None
    assert str(loaded.execution_id) == execution_id
    reopened.close()


def test_f02_conflict_raises(tmp_path: Path) -> None:
    db = tmp_path / "gca.db"
    execution_id = "exec-conflict"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    store = backend.bundle.execution_store
    first = _sample_bundle_record(execution_id)
    second = first.model_copy(update={"snapshot_content_digest": "other"})
    store.put_create_follow_on_record(execution_id, first)
    with pytest.raises(GovernedExecutionCreateFollowOnConflictError):
        store.put_create_follow_on_record(execution_id, second)
    backend.close()


def test_corrupt_f02_fail_closed(tmp_path: Path) -> None:
    db = tmp_path / "gca.db"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    backend.close()
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO gca_create_follow_on (execution_id, payload) VALUES (?, ?)",
        ("exec-bad", "not-json"),
    )
    conn.commit()
    conn.close()
    reopened = open_sqlite_governed_external_work_host_store_bundle_at(db)
    with pytest.raises(ValueError, match="corrupted_create_follow_on"):
        reopened.bundle.execution_store.get_create_follow_on_record("exec-bad")
    reopened.close()


def test_closed_sqlite_backend_rejects_writes(tmp_path: Path) -> None:
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        tmp_path / "gca.db",
    )
    backend.close()
    with pytest.raises(GovernedExternalWorkHostPersistenceError):
        backend.bundle.execution_store.put_state(
            "x",
            GovernedExternalWorkHostState.REQUESTED,
        )


def test_sqlite_backend_double_close_is_safe(tmp_path: Path) -> None:
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        tmp_path / "gca.db",
    )
    backend.close()
    backend.close()


def test_sqlite_write_failure_surfaces_persistence_error(tmp_path: Path) -> None:
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        tmp_path / "gca.db",
    )
    ger = _ger_v2(execution_id="exec-write-fail")
    with patch.object(
        backend._db,
        "_execute",
        side_effect=GovernedExternalWorkHostPersistenceError("disk full"),
    ):
        with pytest.raises(GovernedExternalWorkHostPersistenceError, match="disk full"):
            backend.bundle.execution_store.put_result(ger)
    backend.close()


def test_ambiguous_commit_persists_uncertain_flag_and_reopen_fail_closed(
    tmp_path: Path,
) -> None:
    db = tmp_path / "gca.db"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    with (
        patch.object(sqlite3.Connection, "commit", side_effect=sqlite3.OperationalError("commit")),
        patch.object(
            sqlite3.Connection,
            "rollback",
            side_effect=sqlite3.OperationalError("rollback"),
        ),
    ):
        with pytest.raises(GovernedExternalWorkHostPersistenceError):
            backend.bundle.execution_store.put_state(
                "exec-uncertain",
                GovernedExternalWorkHostState.REQUESTED,
            )
    backend.close()

    reopened = open_sqlite_governed_external_work_host_store_bundle_at(db)
    with pytest.raises(
        GovernedExternalWorkHostPersistenceError,
        match="transaction outcome uncertain",
    ):
        reopened.bundle.execution_store.get_state("exec-uncertain")
    reopened.close()


def test_commit_failure_with_rollback_success_marks_uncertain_and_reopen_fail_closed(
    tmp_path: Path,
) -> None:
    db = tmp_path / "gca.db"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    with patch.object(
        sqlite3.Connection,
        "commit",
        side_effect=sqlite3.OperationalError("commit failed"),
    ):
        with pytest.raises(GovernedExternalWorkHostPersistenceError, match="commit failed"):
            backend.bundle.execution_store.put_state(
                "exec-commit-ambiguous",
                GovernedExternalWorkHostState.REQUESTED,
            )
    backend.close()

    reopened = open_sqlite_governed_external_work_host_store_bundle_at(db)
    with pytest.raises(
        GovernedExternalWorkHostPersistenceError,
        match="transaction outcome uncertain",
    ):
        reopened.bundle.execution_store.get_state("exec-commit-ambiguous")
    reopened.close()


def test_uncertain_marker_persistence_failure_still_fail_closed_in_process(
    tmp_path: Path,
) -> None:
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        tmp_path / "gca.db",
    )
    with (
        patch.object(sqlite3.Connection, "commit", side_effect=sqlite3.OperationalError("commit")),
        patch.object(
            backend._db,
            "_persist_uncertain_write_flag",
            side_effect=sqlite3.OperationalError("meta write failed"),
        ),
    ):
        with pytest.raises(GovernedExternalWorkHostPersistenceError):
            backend.bundle.execution_store.put_state(
                "exec-marker-fail",
                GovernedExternalWorkHostState.REQUESTED,
            )
    with pytest.raises(
        GovernedExternalWorkHostPersistenceError,
        match="transaction outcome uncertain",
    ):
        backend.bundle.execution_store.get_state("exec-marker-fail")
    backend.close()


def test_sqlite_read_failure_not_treated_as_missing_record(tmp_path: Path) -> None:
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        tmp_path / "gca.db",
    )
    with patch.object(
        backend._db,
        "_execute",
        side_effect=GovernedExternalWorkHostPersistenceError("database disk image is malformed"),
    ):
        with pytest.raises(GovernedExternalWorkHostPersistenceError, match="malformed"):
            backend.bundle.execution_store.get_create_follow_on_record("missing")
    backend.close()


def test_host_finalization_incomplete_without_ger_blocks_create_redispatch(
    tmp_path: Path,
) -> None:
    db = tmp_path / "gca.db"
    execution_id = "exec-incomplete-only"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    backend.bundle.execution_store.put_state(
        execution_id,
        GovernedExternalWorkHostState.HOST_FINALIZATION_INCOMPLETE,
    )
    orch = GovernedExternalWorkOrchestrator.__new__(GovernedExternalWorkOrchestrator)
    orch._execution_store = backend.bundle.execution_store
    blocked = orch._prohibit_ordinary_create_redispatch(execution_id)
    assert blocked is not None
    assert blocked.state is GovernedExternalWorkHostState.HOST_FINALIZATION_INCOMPLETE
    assert blocked.reason == "create_redispatch_prohibited"
    assert blocked.governed_result is None
    backend.close()


def test_ger_without_f02_sets_incomplete_on_redispatch_probe(tmp_path: Path) -> None:
    db = tmp_path / "gca.db"
    execution_id = "exec-ger-no-f02"
    ger = _ger_v2(execution_id=execution_id, action=ACTION_CREATE_EXTERNAL_WORK)
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    backend.bundle.execution_store.put_result(ger)
    orch = GovernedExternalWorkOrchestrator.__new__(GovernedExternalWorkOrchestrator)
    orch._execution_store = backend.bundle.execution_store
    blocked = orch._prohibit_ordinary_create_redispatch(execution_id)
    assert blocked is not None
    assert blocked.state is GovernedExternalWorkHostState.HOST_FINALIZATION_INCOMPLETE
    assert blocked.reason == "create_follow_on_record_missing_redispatch_prohibited"
    assert isinstance(blocked.governed_result, GovernedExecutionResultV2)
    backend.close()


def test_strict_gca_session_open_failure_closes_owned_backend(tmp_path: Path) -> None:
    from governed_contractor_application.host.strict_gca_production_process_composition import (
        open_gca_strict_production_serving_session,
    )
    from intergrax.applications._shared.strict_production_serving_launcher import (
        StrictProductionProcessStorage,
        StrictProductionServingEnvironment,
    )
    from tests.unit.applications.test_prod_q_r1_r7_d2_production_process_lifecycle import (
        _governance,
    )

    storage = StrictProductionProcessStorage(
        agent_distribution_db=tmp_path / "dist.db",
        continuation_state_file=tmp_path / "continuation.json",
        provider_invocation_db=tmp_path / "pis.db",
        trusted_artifact_parent_roots=(tmp_path,),
        governed_external_work_host_db=tmp_path / "gca.db",
    )
    environment = StrictProductionServingEnvironment(
        application_id="app",
        application_environment_id="env",
    )
    with patch(
        "governed_contractor_application.host.strict_gca_production_process_composition.open_strict_production_serving_session",
        side_effect=RuntimeError("startup failed"),
    ):
        with pytest.raises(RuntimeError, match="startup failed"):
            open_gca_strict_production_serving_session(
                storage=storage,
                environment=environment,
                governance=_governance(),
                deployment_adapter=object(),  # type: ignore[arg-type]
            )
    reopened = open_sqlite_governed_external_work_host_store_bundle_at(
        storage.governed_external_work_host_db,
    )
    reopened.bundle.execution_store.put_state(
        "after-leak-check",
        GovernedExternalWorkHostState.REQUESTED,
    )
    reopened.close()


def test_execution_outcome_unknown_denies_follow_on_resolution_without_provider(
    tmp_path: Path,
) -> None:
    execution_id = "exec-unknown-follow-on"
    orch, fake, backend = _sqlite_orchestrator(tmp_path)
    backend.bundle.execution_store.put_state(
        execution_id,
        GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN,
    )
    calls_before = fake.create_calls
    assert (
        orch.resolve_create_adapter_result_for_follow_on(
            execution_id,
            tenant_id="tenant-a",
            principal_id="principal-a",
        )
        is None
    )
    assert fake.create_calls == calls_before
    backend.close()


def test_execution_outcome_unknown_without_ger_blocks_orchestrator_create(
    tmp_path: Path,
) -> None:
    execution_id = "exec-unknown-no-ger"
    orch, fake, backend = _sqlite_orchestrator(tmp_path)
    backend.bundle.execution_store.put_state(
        execution_id,
        GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN,
    )
    calls_before = fake.create_calls
    with _active_execution(execution_id):
        step = orch.create(
            task_id=_TASK,
            run_id=_RUN,
            principal_id="principal-a",
            tenant_id="tenant-a",
            metadata={"idempotency_key": "idem-unknown"},
            execution_id=execution_id,
        )
    assert step.state is GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN
    assert step.reason == "create_redispatch_prohibited"
    assert fake.create_calls == calls_before
    backend.close()


def test_uncertainty_marker_read_error_fail_closed_on_reopen(tmp_path: Path) -> None:
    db = tmp_path / "gca.db"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    backend.close()
    with patch(
        "governed_contractor_application.host.sqlite_governed_external_work_host_store_bundle."
        "_SqliteGovernedExternalWorkHostDatabase._read_uncertain_write_flag",
        side_effect=GovernedExternalWorkHostPersistenceError(
            "gca host store uncertainty marker unreadable",
        ),
    ):
        with pytest.raises(
            GovernedExternalWorkHostPersistenceError,
            match="uncertainty marker unreadable",
        ):
            open_sqlite_governed_external_work_host_store_bundle_at(db)


def test_uncertainty_marker_read_error_denies_production_bundle_admission(
    tmp_path: Path,
) -> None:
    db = tmp_path / "gca.db"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    backend.close()
    with patch(
        "governed_contractor_application.host.sqlite_governed_external_work_host_store_bundle."
        "_SqliteGovernedExternalWorkHostDatabase._read_uncertain_write_flag",
        side_effect=GovernedExternalWorkHostPersistenceError(
            "gca host store uncertainty marker unreadable",
        ),
    ):
        with pytest.raises(GovernedExternalWorkHostPersistenceError):
            reopened = open_sqlite_governed_external_work_host_store_bundle_at(db)
            validate_production_governed_external_work_host_store_bundle(
                reopened.bundle,
            )


def test_f03_ambiguous_commit_marker_persist_failure_reopen_fail_closed(
    tmp_path: Path,
) -> None:
    db = tmp_path / "gca.db"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    with (
        patch.object(sqlite3.Connection, "commit", side_effect=sqlite3.OperationalError("commit")),
        patch.object(
            backend._db,
            "_persist_uncertain_write_flag",
            side_effect=sqlite3.OperationalError("meta write failed"),
        ),
    ):
        with pytest.raises(GovernedExternalWorkHostPersistenceError):
            backend.bundle.execution_store.put_state(
                "exec-f03-reopen",
                GovernedExternalWorkHostState.REQUESTED,
            )
    backend.close()

    with pytest.raises(
        GovernedExternalWorkHostPersistenceError,
        match="missing gca host store schema metadata",
    ):
        open_sqlite_governed_external_work_host_store_bundle_at(db)


def test_f03_reopen_blocks_orchestrator_create_after_ambiguous_commit(
    tmp_path: Path,
) -> None:
    db = tmp_path / "gca-orch.db"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    execution_id = "exec-f03-create-block"
    with (
        patch.object(sqlite3.Connection, "commit", side_effect=sqlite3.OperationalError("commit")),
        patch.object(
            backend._db,
            "_persist_uncertain_write_flag",
            side_effect=sqlite3.OperationalError("meta write failed"),
        ),
    ):
        with pytest.raises(GovernedExternalWorkHostPersistenceError):
            backend.bundle.execution_store.put_state(
                execution_id,
                GovernedExternalWorkHostState.REQUESTED,
            )
    backend.close()

    with pytest.raises(GovernedExternalWorkHostPersistenceError):
        _sqlite_orchestrator(tmp_path, db_name="gca-orch.db")


def test_policy_bundle_round_trip(tmp_path: Path) -> None:
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        tmp_path / "gca.db",
    )
    bundle = ImmutableRuntimePolicyBundle(
        bundle_id="b1",
        version="1",
        rules=(),
    )
    backend.bundle.bundle_store.put_bundle(bundle)
    loaded = backend.bundle.bundle_store.get_bundle("b1", "1")
    assert loaded is not None
    assert loaded.bundle_id == "b1"
    backend.close()
