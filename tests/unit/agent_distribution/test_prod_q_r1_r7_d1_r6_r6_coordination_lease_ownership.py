# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R6-R6 — operation-owned SQLite lease & compensation-before-unlock."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest

import intergrax.agent_distribution.activation_domain_coordinator as adc_module
from intergrax.agent_distribution.activation_domain_coordinator import (
    _compensate_unit_open_after_coordination_failure,
    build_activation_domain_coordinator_for_sqlite_bundle,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_r6_r5_serving_admission_linearization import (
    _sqlite_bundle_from_seeded,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure import (
    _seeded_coordinator,
)

pytestmark = [pytest.mark.unit]


def test_no_process_global_sqlite_lease_registry() -> None:
    assert not hasattr(adc_module, "_ACTIVE_SQLITE_ADMISSION_LEASES")


def test_s7_publish_failure_compensation_holds_domain_before_pointer_swap(
    tmp_path: Path,
) -> None:
    """R6-B01: physical compensation completes before activation lock release; swap waits."""
    adapter, unit_ref, _digest, revision, *_stores, request, _admission = _seeded_coordinator(
        tmp_path
    )
    bundle = _sqlite_bundle_from_seeded(tmp_path, revision, unit_ref, "r6-s7.db")
    db = bundle.database
    coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    unit.coordinator_admission_open_staged = None
    unit.admission.begin_drain()
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    at_compensate = threading.Event()
    release_compensate = threading.Event()
    swap_finished = threading.Event()
    swap_started = threading.Event()
    lock_held_during_compensate = threading.Event()
    real_compensate = _compensate_unit_open_after_coordination_failure

    def _compensate_hold(*args, **kwargs):  # type: ignore[no-untyped-def]
        at_compensate.set()
        assert db._activation_lock._is_owned()  # noqa: SLF001 — qualification instrumentation
        lock_held_during_compensate.set()
        assert release_compensate.wait(timeout=5.0)
        real_compensate(*args, **kwargs)

    def _swap() -> None:
        assert at_compensate.wait(timeout=5.0)
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
            new_revision_id="rev-r6-s7-interleave",
            prior_revision_id=revision.runtime_revision_id,
            committed_at=datetime.now(UTC),
        )
        swap_finished.set()

    with (
        mock.patch.object(
            adapter,
            "_publish_coordinator_verified_unit_admission_open",
            side_effect=RuntimeError("publish failed"),
        ),
        mock.patch.object(
            adc_module,
            "_compensate_unit_open_after_coordination_failure",
            _compensate_hold,
        ),
    ):
        open_errors: list[BaseException] = []

        def _unit_open() -> None:
            try:
                coordinator.coordinate_unit_serving_admission_open(
                    request,
                    unit_admission_open=port,
                )
            except BaseException as exc:  # noqa: BLE001
                open_errors.append(exc)

        open_thread = threading.Thread(target=_unit_open)
        swap_thread = threading.Thread(target=_swap)
        open_thread.start()
        assert at_compensate.wait(timeout=5.0)
        swap_thread.start()
        assert swap_started.wait(timeout=5.0)
        assert lock_held_during_compensate.is_set()
        assert not swap_finished.wait(timeout=0.15)
        release_compensate.set()
        open_thread.join(timeout=10.0)
        swap_thread.join(timeout=10.0)
    assert len(open_errors) == 1
    assert "publish failed" in str(open_errors[0])
    assert swap_finished.is_set()
    assert unit.coordinator_admission_open_staged is None
    assert not unit.serving_admission_authorized


def test_s8_compensation_failure_chains_publish_error_and_blocks_swap_until_release(
    tmp_path: Path,
) -> None:
    adapter, unit_ref, _digest, revision, *_stores, request, _admission = _seeded_coordinator(
        tmp_path
    )
    bundle = _sqlite_bundle_from_seeded(tmp_path, revision, unit_ref, "r6-s8.db")
    coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    unit.coordinator_admission_open_staged = None
    unit.admission.begin_drain()
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    at_compensate = threading.Event()
    release_compensate = threading.Event()
    swap_finished = threading.Event()

    def _compensate_fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        at_compensate.set()
        assert release_compensate.wait(timeout=5.0)
        raise RuntimeError("compensation failed")

    def _swap() -> None:
        assert at_compensate.wait(timeout=5.0)
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
            new_revision_id="rev-r6-s8",
            prior_revision_id=revision.runtime_revision_id,
            committed_at=datetime.now(UTC),
        )
        swap_finished.set()

    errors: list[BaseException] = []

    def _unit_open() -> None:
        try:
            coordinator.coordinate_unit_serving_admission_open(
                request,
                unit_admission_open=port,
            )
        except BaseException as exc:  # noqa: BLE001 — qualification capture
            errors.append(exc)

    with (
        mock.patch.object(
            adapter,
            "_publish_coordinator_verified_unit_admission_open",
            side_effect=RuntimeError("publish failed"),
        ),
        mock.patch.object(
            adc_module,
            "_compensate_unit_open_after_coordination_failure",
            _compensate_fail,
        ),
    ):
        swap_thread = threading.Thread(target=_swap)
        open_thread = threading.Thread(target=_unit_open)
        swap_thread.start()
        open_thread.start()
        assert at_compensate.wait(timeout=5.0)
        assert not swap_finished.wait(timeout=0.15)
        release_compensate.set()
        open_thread.join(timeout=10.0)
        swap_thread.join(timeout=10.0)
    assert len(errors) == 1
    exc_info = errors[0]
    assert isinstance(exc_info, RuntimeError)
    assert "compensation failed" in str(exc_info)
    assert exc_info.__cause__ is not None
    assert "publish failed" in str(exc_info.__cause__)
    assert swap_finished.is_set()
    assert not unit.serving_admission_authorized


def test_activation_domain_lock_held_through_compensation_after_publish_failure(
    tmp_path: Path,
) -> None:
    adapter, unit_ref, _digest, revision, *_stores, request, _admission = _seeded_coordinator(
        tmp_path
    )
    bundle = _sqlite_bundle_from_seeded(tmp_path, revision, unit_ref, "r6-lock-comp.db")
    db = bundle.database
    coordinator = build_activation_domain_coordinator_for_sqlite_bundle(bundle)
    unit = adapter._units[unit_ref]
    unit.serving_admission_authorized = False
    unit.coordinator_admission_open_staged = None
    unit.admission.begin_drain()
    port = adapter.coordinator_verified_unit_serving_admission_open_port(unit_ref)
    lock_held = threading.Event()
    real_compensate = _compensate_unit_open_after_coordination_failure

    def _compensate_track(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert db._activation_lock._is_owned()  # noqa: SLF001
        lock_held.set()
        real_compensate(*args, **kwargs)

    with (
        mock.patch.object(
            adapter,
            "_publish_coordinator_verified_unit_admission_open",
            side_effect=RuntimeError("publish failed"),
        ),
        mock.patch.object(
            adc_module,
            "_compensate_unit_open_after_coordination_failure",
            _compensate_track,
        ),
    ):
        with pytest.raises(RuntimeError, match="publish failed"):
            coordinator.coordinate_unit_serving_admission_open(
                request,
                unit_admission_open=port,
            )
    assert lock_held.is_set()


def test_sqlite_lease_close_is_idempotent(tmp_path: Path) -> None:
    _adapter, _unit_ref, _digest, revision, *_rest = _seeded_coordinator(tmp_path)
    bundle = _sqlite_bundle_from_seeded(
        tmp_path, revision, _unit_ref, "r6-lease-idempotent.db"
    )
    lease = adc_module._open_sqlite_admission_domain_lease(bundle.database)
    lease.rollback()
    lease.close()
    lease.close()
