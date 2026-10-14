# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R6-R7 — lease cleanup, rollback/admission evidence & fault matrix."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest

import intergrax.agent_distribution.activation_domain_coordinator as adc_module
from intergrax.agent_distribution.activation import (
    ActivationService,
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.activation_domain_coordinator import (
    ServingExecutionAdmissionRequest,
    _SqliteAdmissionDomainLease,
    _open_sqlite_admission_domain_lease,
    _sqlite_admission_domain_lease,
    build_activation_domain_coordinator_for_sqlite_bundle,
    verify_serving_execution_eligibility_from_state,
)
from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from testing_support.agent_distribution.venv_bundle_deployment_fixtures import (
    file_artifact_locator,
    write_minimal_venv_bundle,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_venv_bundle_colocated_deployment_adapter import (
    _authorize_adapter_execution_gate,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.serving_execution_admission import (
    ServingExecutionAdmissionError,
)
from intergrax.agent_distribution.sqlite_stores import build_sqlite_agent_distribution_store_bundle
from testing_support.agent_distribution.activation_materialization_fixtures import (
    persist_materialization_for_revision,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_r6_r5_serving_admission_linearization import (
    _sqlite_bundle_from_seeded,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure import (
    _seeded_coordinator,
)
pytestmark = [pytest.mark.unit]

_APP = "app-a"
_ENV = "env-prod"


def _venv_in_flight(
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
    serving_unit_ref: str,
) -> int:
    return adapter._units[serving_unit_ref].admission.in_flight_count()  # noqa: SLF001


def _lock_released(db: object) -> bool:
    return db._activation_lock.is_unheld()  # noqa: SLF001


def _revision(
    revision_id: str,
    *,
    artifact: str,
    lock: MaterializedRuntimeLock,
    state: RuntimeRevisionState,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-hash",
        materialized_runtime_lock_id=lock.lock_id,
        materialized_runtime_lock_digest=lock.lock_digest,
        runtime_graph_digest=f"graph-{revision_id}",
        materialization_artifact_digest=artifact,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=state,
        activated_at=datetime.now(UTC) if state is RuntimeRevisionState.ACTIVE else None,
    )


def _bootstrap_sqlite_activation_pair(tmp_path: Path):
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "s3-rollback.db")
    deployment_adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    projection = FakeRuntimeServingProjectionCoordinator()
    revision_service = RuntimeRevisionService(bundle.revision_store)
    activation = ActivationService(
        revision_store=bundle.revision_store,
        deployment_instance_store=bundle.deployment_instance_store,
        serving_store=bundle.serving_store,
        activation_store=bundle.activation_store,
        deployment_adapter=deployment_adapter,
        projection_coordinator=projection,
        materialization_store=bundle.materialization_store,
    )
    revision_materials: dict[str, tuple[str, MaterializedRuntimeLock, str]] = {}
    for revision_id, suffix in (("rev-a", "a"), ("rev-b", "b")):
        root = tmp_path / f"bundle-{suffix}"
        digest, lock = write_minimal_venv_bundle(root)
        locator = file_artifact_locator(root)
        revision_materials[revision_id] = (digest, lock, locator)
        candidate = _revision(
            revision_id,
            state=RuntimeRevisionState.CANDIDATE,
            artifact=digest,
            lock=lock,
        )
        revision_service.persist_candidate_revision(candidate)
        validated = _revision(
            revision_id,
            state=RuntimeRevisionState.VALIDATED,
            artifact=digest,
            lock=lock,
        )
        revision_service.mark_validated(revision_id, validated_revision=validated)
        persist_materialization_for_revision(
            bundle.materialization_store,
            validated,
            artifact_locator=locator,
        )
    digest_a, _lock_a, locator_a = revision_materials["rev-a"]
    digest_b, lock_b, locator_b = revision_materials["rev-b"]
    activation.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-a",
        artifact_locator=locator_a,
    )
    activation.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-a",
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
        expected_artifact_digest=digest_a,
    )
    activation.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-b",
        artifact_locator=locator_b,
    )
    activation.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-b",
        expected_prior_traffic_revision_id="rev-a",
        expected_serving_pointer_revision=1,
        expected_artifact_digest=digest_b,
    )
    instance_b = bundle.deployment_instance_store.get_instance(_APP, _ENV, "rev-b")
    assert instance_b is not None and instance_b.serving_unit_ref is not None
    unit_b = instance_b.serving_unit_ref
    coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    request_b = ServingExecutionAdmissionRequest(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-b",
        serving_unit_ref=unit_b,
        materialization_artifact_digest=digest_b,
    )
    rev_b = _revision(
        "rev-b",
        state=RuntimeRevisionState.ACTIVE,
        artifact=digest_b,
        lock=lock_b,
    )
    _authorize_adapter_execution_gate(
        deployment_adapter,
        serving_store=bundle.serving_store,
        instance_store=bundle.deployment_instance_store,
        revision_store=bundle.revision_store,
        revision=rev_b,
        serving_unit_ref=unit_b,
    )
    return bundle, activation, coordinator, deployment_adapter, request_b, unit_b


def test_s3_a_activation_service_rollback_before_b_admission(tmp_path: Path) -> None:
    bundle, activation, coordinator, adapter, request_b, unit_b = _bootstrap_sqlite_activation_pair(
        tmp_path
    )
    activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-b",
        expected_serving_pointer_revision=2,
    )
    scope = bundle.serving_store.get_serving_record(_APP, _ENV)
    assert scope is not None
    assert scope.traffic_serving_revision_id == "rev-a"
    assert _venv_in_flight(adapter, unit_b) == 0
    with pytest.raises(ServingExecutionAdmissionError):
        coordinator.coordinate_physical_execution_reservation(
            request_b,
            physical_reserve=lambda: adapter.reserve_unit_execution(unit_b),
        )
    assert _venv_in_flight(adapter, unit_b) == 0


def test_s3_b_rollback_waits_for_inflight_sqlite_admission(tmp_path: Path) -> None:
    bundle, activation, coordinator, adapter, request_b, unit_b = _bootstrap_sqlite_activation_pair(
        tmp_path
    )
    t1_before_reserve = threading.Event()
    t2_started_rollback = threading.Event()
    t1_reserved = threading.Event()
    rollback_done = threading.Event()
    held: list[object] = []
    worker_errors: list[BaseException] = []
    original_verify = verify_serving_execution_eligibility_from_state

    def _verify_hold(state, req):  # type: ignore[no-untyped-def]
        evidence = original_verify(state, req)
        t1_before_reserve.set()
        assert t2_started_rollback.wait(timeout=5.0)
        return evidence

    def _reserve() -> object:
        token = adapter.reserve_unit_execution(unit_b)
        held.append(token)
        assert _venv_in_flight(adapter, unit_b) == 1
        t1_reserved.set()
        return token

    def _t1() -> None:
        try:
            with mock.patch.object(
                adc_module,
                "verify_serving_execution_eligibility_from_state",
                side_effect=_verify_hold,
            ):
                coordinator.coordinate_physical_execution_reservation(
                    request_b,
                    physical_reserve=_reserve,
                )
        except BaseException as exc:
            worker_errors.append(exc)

    def _t2() -> None:
        try:
            assert t1_before_reserve.wait(timeout=5.0)
            t2_started_rollback.set()
            assert not rollback_done.wait(timeout=0.05)
            activation.rollback(
                application_id=_APP,
                application_environment_id=_ENV,
                expected_current_traffic_revision_id="rev-b",
                expected_serving_pointer_revision=2,
            )
            rollback_done.set()
        except BaseException as exc:
            worker_errors.append(exc)

    t1 = threading.Thread(target=_t1)
    t2 = threading.Thread(target=_t2)
    t1.start()
    t2.start()
    assert t1_before_reserve.wait(timeout=5.0)
    assert t2_started_rollback.wait(timeout=5.0)
    assert not rollback_done.is_set()
    assert t1_reserved.wait(timeout=5.0)
    assert rollback_done.wait(timeout=10.0)
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)
    assert not worker_errors
    assert _venv_in_flight(adapter, unit_b) == 1
    scope = bundle.serving_store.get_serving_record(_APP, _ENV)
    assert scope is not None
    assert scope.traffic_serving_revision_id == "rev-a"
    with pytest.raises(ServingExecutionAdmissionError):
        coordinator.coordinate_physical_execution_reservation(
            request_b,
            physical_reserve=lambda: adapter.reserve_unit_execution(unit_b),
        )
    assert _venv_in_flight(adapter, unit_b) == 1
    held[0].release()
    assert _venv_in_flight(adapter, unit_b) == 0


def test_nested_activation_store_mutate_while_lease_holds_fails_closed(
    tmp_path: Path,
) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "nested-coord.db")
    with pytest.raises(ServingExecutionAdmissionError, match="nested activation domain lock"):
        with _sqlite_admission_domain_lease(bundle.database):
            bundle.activation_store.atomic_commit_rollback(
                application_id=_APP,
                application_environment_id=_ENV,
                expected_current_revision_id="rev-b",
                expected_pointer_revision=2,
                target_revision_id="rev-a",
                committed_at=datetime.now(UTC),
            )
    assert _lock_released(bundle.database)


def test_restart_stale_physical_unit_fail_closed_admission(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision, serving_store, instance_store, revision_store, coordinator, request, admission = (
        _seeded_coordinator(tmp_path)
    )
    scope = serving_store.get_serving_record(
        revision.application_id,
        revision.application_environment_id,
    )
    assert scope is not None
    assert scope.traffic_serving_revision_id == revision.runtime_revision_id
    adapter.stop(revision, serving_unit_ref=unit_ref)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="stopped"):
        admission.admit_execution(unit_ref)
    reserves = 0

    def _reserve() -> object:
        nonlocal reserves
        reserves += 1
        return adapter.reserve_unit_execution(unit_ref)

    with pytest.raises(VenvBundleColocatedDeploymentError, match="stopped"):
        coordinator.coordinate_physical_execution_reservation(
            request,
            physical_reserve=_reserve,
        )
    assert reserves == 1


def test_durable_serving_without_gate_authorization_cannot_reserve(tmp_path: Path) -> None:
    adapter, unit_ref, digest, revision, serving_store, instance_store, revision_store, *_rest = (
        _seeded_coordinator(tmp_path)
    )
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    assert serving_store.get_serving_record(
        revision.application_id,
        revision.application_environment_id,
    ) is not None
    with pytest.raises(Exception):
        adapter.reserve_unit_execution(unit_ref)


def test_load_state_operational_error_releases_lock(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "load-fail.db")
    with mock.patch.object(
        bundle.database,
        "_load_state",
        side_effect=sqlite3.OperationalError("snapshot load failed"),
    ):
        with pytest.raises(ServingExecutionAdmissionError):
            _open_sqlite_admission_domain_lease(bundle.database)
    assert _lock_released(bundle.database)


def test_load_state_generic_error_releases_lock(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "load-generic.db")
    with mock.patch.object(bundle.database, "_load_state", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            _open_sqlite_admission_domain_lease(bundle.database)
    assert _lock_released(bundle.database)


def test_acquire_cleanup_rollback_failure_still_releases_lock(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "rb-fail.db")
    real_connect = bundle.database._connect

    class _Conn:
        __slots__ = ("_inner",)

        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, *args, **kwargs):
            return self._inner.execute(*args, **kwargs)

        def rollback(self) -> None:
            raise sqlite3.OperationalError("rollback failed during acquisition cleanup")

        def close(self) -> None:
            self._inner.close()

    def _connect() -> _Conn:
        return _Conn(real_connect())

    with (
        mock.patch.object(bundle.database, "_connect", side_effect=_connect),
        mock.patch.object(
            bundle.database,
            "_load_state",
            side_effect=sqlite3.OperationalError("snapshot load failed"),
        ),
    ):
        with pytest.raises(ServingExecutionAdmissionError, match="cleanup also failed"):
            _open_sqlite_admission_domain_lease(bundle.database)
    assert _lock_released(bundle.database)


def test_acquire_primary_and_cleanup_both_visible(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "dual-fail.db")

    def _connect_fail() -> sqlite3.Connection:
        raise sqlite3.OperationalError("connect failed")

    with mock.patch.object(bundle.database, "_connect", side_effect=_connect_fail):
        with pytest.raises(ServingExecutionAdmissionError, match="coordination failed") as raised:
            _open_sqlite_admission_domain_lease(bundle.database)
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
    assert _lock_released(bundle.database)


def test_commit_durable_failure_then_successful_lease_finalize(tmp_path: Path) -> None:
    adapter, unit_ref, _digest, revision, *_stores, request, _admission = _seeded_coordinator(
        tmp_path
    )
    bundle = _sqlite_bundle_from_seeded(tmp_path, revision, unit_ref, "r7-commit-fail.db")
    coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    unit.coordinator_admission_open_staged = None
    unit.admission.begin_drain()
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)

    def _fail_commit(self: _SqliteAdmissionDomainLease) -> None:
        raise sqlite3.OperationalError("commit rejected")

    with mock.patch.object(_SqliteAdmissionDomainLease, "commit_durable", _fail_commit):
        with pytest.raises(sqlite3.OperationalError, match="commit rejected"):
            coordinator.coordinate_unit_serving_admission_open(
                request,
                unit_admission_open=port,
            )
    assert _lock_released(bundle.database)
    assert not unit.serving_admission_authorized


def test_commit_durable_uncertain_when_rollback_also_fails(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "commit-rb-fail.db")
    lease = _open_sqlite_admission_domain_lease(bundle.database)
    inner = lease._conn  # noqa: SLF001

    class _Conn:
        __slots__ = ("_inner",)

        def __init__(self, wrapped: sqlite3.Connection) -> None:
            self._inner = wrapped

        def commit(self) -> None:
            raise sqlite3.OperationalError("commit rejected")

        def rollback(self) -> None:
            raise sqlite3.OperationalError("rollback failed")

        def close(self) -> None:
            self._inner.close()

    lease._conn = _Conn(inner)  # noqa: SLF001
    with pytest.raises(ServingExecutionAdmissionError, match="outcome uncertain"):
        lease.commit_durable()
    assert lease._transaction_active  # noqa: SLF001
    lease.close()
    assert _lock_released(bundle.database)


def test_lease_close_connection_failure_still_releases_lock(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "close-fail.db")
    lease = _open_sqlite_admission_domain_lease(bundle.database)
    inner = lease._conn  # noqa: SLF001

    class _Conn:
        __slots__ = ("_inner",)

        def __init__(self, wrapped: sqlite3.Connection) -> None:
            self._inner = wrapped

        def rollback(self) -> None:
            self._inner.rollback()

        def commit(self) -> None:
            self._inner.commit()

        def close(self) -> None:
            raise sqlite3.OperationalError("close failed")

    lease._conn = _Conn(inner)  # noqa: SLF001
    with pytest.raises(ServingExecutionAdmissionError, match="connection close failed"):
        lease.close()
    assert _lock_released(bundle.database)
    lease.close()


def test_failed_lease_allows_subsequent_operation(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "retry-after-fail.db")
    with mock.patch.object(
        bundle.database,
        "_load_state",
        side_effect=sqlite3.OperationalError("first load failed"),
    ):
        with pytest.raises(ServingExecutionAdmissionError):
            _open_sqlite_admission_domain_lease(bundle.database)
    lease = _open_sqlite_admission_domain_lease(bundle.database)
    lease.rollback()
    lease.close()
    assert _lock_released(bundle.database)


def test_connect_raises_before_connection_no_lock_leak(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "connect-fail.db")
    with mock.patch.object(
        bundle.database,
        "_connect",
        side_effect=sqlite3.OperationalError("connect failed"),
    ):
        with pytest.raises(ServingExecutionAdmissionError):
            _open_sqlite_admission_domain_lease(bundle.database)
    assert _lock_released(bundle.database)


def test_begin_immediate_raises_releases_lock(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "begin-fail.db")
    real_connect = bundle.database._connect

    class _Conn:
        __slots__ = ("_inner",)

        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, statement, *args, **kwargs):
            if "BEGIN IMMEDIATE" in str(statement):
                raise sqlite3.OperationalError("begin failed")
            return self._inner.execute(statement, *args, **kwargs)

        def rollback(self) -> None:
            self._inner.rollback()

        def close(self) -> None:
            self._inner.close()

    def _connect() -> _Conn:
        return _Conn(real_connect())

    with mock.patch.object(bundle.database, "_connect", side_effect=_connect):
        with pytest.raises(ServingExecutionAdmissionError):
            _open_sqlite_admission_domain_lease(bundle.database)
    assert _lock_released(bundle.database)
