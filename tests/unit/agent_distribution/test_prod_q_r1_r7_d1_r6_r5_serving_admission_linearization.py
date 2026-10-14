# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R6-R5 — commit/publish linearization & rollback concurrency."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest

from intergrax.agent_distribution.activation_domain_coordinator import (
    _SqliteAdmissionDomainLease,
    build_activation_domain_coordinator_for_sqlite_bundle,
)
from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.serving_execution_admission import (
    ServingExecutionAdmissionError,
)
from intergrax.agent_distribution.sqlite_stores import build_sqlite_agent_distribution_store_bundle
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure import (
    _seeded_coordinator,
)

pytestmark = [pytest.mark.unit]


def _sqlite_bundle_from_seeded(tmp_path: Path, revision, unit_ref, db_name: str):
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / db_name)
    activated_at = datetime.now(UTC)
    bundle.revision_store.persist_candidate_revision(
        revision.model_copy(
            update={
                "revision_state": RuntimeRevisionState.ACTIVE,
                "activated_at": activated_at,
            },
        ),
    )
    bundle.serving_store.atomic_swap_serving_revision(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=revision.runtime_revision_id,
        prior_revision_id=None,
        committed_at=activated_at,
    )
    from intergrax.agent_distribution.deployment import DeploymentInstanceRecord

    bundle.deployment_instance_store.persist_instance(
        DeploymentInstanceRecord(
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            runtime_revision_id=revision.runtime_revision_id,
            serving_unit_ref=unit_ref,
            instance_state=DeploymentInstanceState.SERVING,
            readiness_evidence_ref="evidence",
        ),
    )
    return bundle


def test_sqlite_unit_open_blocks_pointer_swap_until_publish(tmp_path: Path) -> None:
    """R5-B01: sanctioned pointer mutation cannot interleave commit and publish."""
    adapter, unit_ref, _digest, revision, *_stores, request, _admission = _seeded_coordinator(
        tmp_path
    )
    bundle = _sqlite_bundle_from_seeded(tmp_path, revision, unit_ref, "r5-linear.db")
    coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    unit.coordinator_admission_open_staged = None
    unit.admission.begin_drain()
    at_publish = threading.Event()
    swap_finished = threading.Event()
    swap_started = threading.Event()
    release_publish = threading.Event()
    real_publish = adapter._publish_coordinator_verified_unit_admission_open

    def _publish_hold(*args):  # type: ignore[no-untyped-def]
        evidence = args[-1]
        at_publish.set()
        assert release_publish.wait(timeout=5.0)
        real_publish(evidence)

    def _swap() -> None:
        assert at_publish.wait(timeout=5.0)
        swap_started.set()
        scope = bundle.serving_store.get_serving_record(
            revision.application_id,
            revision.application_environment_id,
        )
        assert scope is not None
        bundle.serving_store.atomic_swap_serving_revision(
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            expected_current_revision_id=scope.traffic_serving_revision_id,
            expected_pointer_revision=scope.serving_pointer_revision,
            new_revision_id="rev-r5-interleave",
            prior_revision_id=revision.runtime_revision_id,
            committed_at=datetime.now(UTC),
        )
        swap_finished.set()

    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)

    def _unit_open() -> None:
        with mock.patch.object(adapter, "_publish_coordinator_verified_unit_admission_open", _publish_hold):
            coordinator.coordinate_unit_serving_admission_open(
                request,
                unit_admission_open=port,
            )

    open_thread = threading.Thread(target=_unit_open)
    swap_thread = threading.Thread(target=_swap)
    open_thread.start()
    assert at_publish.wait(timeout=5.0)
    swap_thread.start()
    assert swap_started.wait(timeout=5.0)
    assert not swap_finished.wait(timeout=0.15)
    release_publish.set()
    open_thread.join(timeout=10.0)
    swap_thread.join(timeout=10.0)
    assert swap_finished.is_set()


def test_sqlite_unit_open_s2_concurrent_swap_waits_for_physical_reserve(
    tmp_path: Path,
) -> None:
    from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_r6_r3_coordinator_scoped_gate_closure import (
        _run_s2_concurrent_admission_wins,
    )

    adapter, unit_ref, _digest, revision, *_stores, request, _admission = _seeded_coordinator(
        tmp_path
    )
    bundle = _sqlite_bundle_from_seeded(tmp_path, revision, unit_ref, "r5-s2-sqlite.db")
    coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    unit.coordinator_admission_open_staged = None
    unit.admission.begin_drain()
    coordinator.coordinate_unit_serving_admission_open(
        request,
        unit_admission_open=adapter.coordinator_verified_unit_serving_admission_open_port(
            unit_ref
        ),
    )

    def _reserve() -> object:
        return adapter.reserve_unit_execution(unit_ref)

    _run_s2_concurrent_admission_wins(
        serving_store=bundle.serving_store,
        coordinator=coordinator,
        request=request,
        revision=revision,
        unit_ref=unit_ref,
        adapter=adapter,
        physical_reserve=_reserve,
    )


def test_activation_domain_lock_held_from_begin_through_publish(tmp_path: Path) -> None:
    lock_held_at_publish = threading.Event()
    adapter, unit_ref, _digest, revision, *_stores, request, _admission = _seeded_coordinator(
        tmp_path
    )
    sqlite_bundle = _sqlite_bundle_from_seeded(tmp_path, revision, unit_ref, "r5-lock2.db")
    db = sqlite_bundle.database

    original_commit = _SqliteAdmissionDomainLease.commit_durable

    def _commit_track(self: _SqliteAdmissionDomainLease) -> None:
        original_commit(self)
        assert db._activation_lock.is_owned_by_current_thread()  # noqa: SLF001
        lock_held_at_publish.set()
    coordinator = build_activation_domain_coordinator_for_sqlite_bundle(sqlite_bundle)
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    unit.coordinator_admission_open_staged = None
    unit.admission.begin_drain()

    with mock.patch.object(_SqliteAdmissionDomainLease, "commit_durable", _commit_track):
        coordinator.coordinate_unit_serving_admission_open(
            request,
            unit_admission_open=adapter.coordinator_verified_unit_serving_admission_open_port(
                unit_ref
            ),
        )
    assert lock_held_at_publish.is_set()


def test_publish_failure_after_sqlite_commit_compensates(tmp_path: Path) -> None:
    adapter, unit_ref, _digest, revision, *_stores, request, _admission = _seeded_coordinator(
        tmp_path
    )
    bundle = _sqlite_bundle_from_seeded(tmp_path, revision, unit_ref, "r5-pub-fail.db")
    coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    unit.coordinator_admission_open_staged = None
    unit.admission.begin_drain()
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    with mock.patch.object(
        adapter,
        "_publish_coordinator_verified_unit_admission_open",
        side_effect=RuntimeError("publish failed"),
    ):
        with pytest.raises(RuntimeError, match="publish failed"):
            coordinator.coordinate_unit_serving_admission_open(
                request,
                unit_admission_open=port,
            )
    assert unit.coordinator_admission_open_staged is None
    assert not unit.serving_admission_authorized
