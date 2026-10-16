# © Artur Czarnecki. All rights reserved.

"""R7-Q §C dimension 15 — integrated strict HTTP execution admission gate.

Proves request-level execution acquires coordinator-authorized physical admission before
inner host task execution; denials are fail-closed with zero inner invocations and balanced
in-flight accounting.

Reused evidence (not replayed via nested pytest):
- P2-C ``test_prod_q_r1_r7_q_p2_c_http_execution_admission.py`` — HTTP admission continuity.
- DIM10 — wrong revision / wrong deployment-instance unit ref at HTTP boundary.
- DIM14-04 — physical ``stop()`` then stale ASGI HTTP denial.
"""

from __future__ import annotations

from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from intergrax.agent_distribution.activation_domain_coordinator import (
    ServingExecutionAdmissionRequest,
)
from intergrax.applications._shared.colocated_serving_unit_execution_admission import (
    StoreBackedColocatedServingUnitExecutionAdmission,
)
from intergrax.contracts.execution_identity import require_active_execution_identity
from intergrax.runtime.task.task import TaskResult, TaskState
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    repo_root,
)
from tests.qualification.prod_q._prod_q_strict_http_caller_auth import (
    strict_http_auth_headers,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_c_dim14_stale_physical_unit_fail_closed_gate import (
    _stop_colocated_physical_unit,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
    _INNER_DELEGATE,
    _build_strict_http_fixture,
    _close_serving_unit_admission,
    _completed_task_result,
    _physical_in_flight,
    _run_request_json,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "f440d23b08e8e25865eba1fbd2a5caec0c524951"


@pytest.fixture(autouse=True)
def _isolate_production_process_lifecycle_slot() -> None:
    from intergrax.applications._shared import production_process_lifecycle as ppl

    active = ppl._active_production_process_lifecycle
    if active is not None and not active._process_closed:
        active.close_process_resources()
    yield
    active_after = ppl._active_production_process_lifecycle
    if active_after is not None and not active_after._process_closed:
        active_after.close_process_resources()


def test_c_dim15_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def _assert_http_denied_no_execution(
    response,
    *,
    inner_mock: AsyncMock,
    adapter,
    unit_ref: str,
    successful_reserves: list[int],
) -> None:
    assert response.status_code == 502
    assert response.json()["detail"].startswith("run_error:")
    inner_mock.assert_not_called()
    assert successful_reserves == []
    assert _physical_in_flight(adapter, unit_ref) == 0


def test_c_dim15_01_positive_authorized_http_execution_admission_accounting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """DIM15-01 — one admit, one inner, one release; in-flight returns to zero."""
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim15-01-positive",
    )
    handoff = fixture.session.serving_runtime_handoff
    assert handoff is not None
    admission = handoff.colocated_execution_admission
    assert admission is not None
    coordinator = admission.activation_domain_coordinator
    real_coordinate = coordinator.coordinate_physical_execution_reservation
    coordinate_calls: list[ServingExecutionAdmissionRequest] = []
    real_admit = StoreBackedColocatedServingUnitExecutionAdmission.admit_execution
    real_release = StoreBackedColocatedServingUnitExecutionAdmission.release_execution
    admit_count = 0
    release_count = 0
    successful_reserves: list[str] = []
    real_reserve = fixture.adapter.reserve_unit_execution

    def _recording_coordinate(request, *, physical_reserve):
        coordinate_calls.append(request)
        return real_coordinate(request, physical_reserve=physical_reserve)

    def _recording_admit(self, serving_unit_ref: str) -> None:
        nonlocal admit_count
        admit_count += 1
        real_admit(self, serving_unit_ref)

    def _recording_release(self, serving_unit_ref: str) -> None:
        nonlocal release_count
        release_count += 1
        real_release(self, serving_unit_ref)

    def _recording_reserve(unit_ref: str):
        reservation = real_reserve(unit_ref)
        successful_reserves.append(unit_ref)
        return reservation

    inner_calls = 0

    async def _inner_once(*_args, **_kwargs) -> TaskResult:
        nonlocal inner_calls
        inner_calls += 1
        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 1
        return _completed_task_result()

    assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0
    with patch.object(
        coordinator,
        "coordinate_physical_execution_reservation",
        side_effect=_recording_coordinate,
    ), patch.object(
        StoreBackedColocatedServingUnitExecutionAdmission,
        "admit_execution",
        _recording_admit,
    ), patch.object(
        StoreBackedColocatedServingUnitExecutionAdmission,
        "release_execution",
        _recording_release,
    ), patch.object(
        fixture.adapter,
        "reserve_unit_execution",
        side_effect=_recording_reserve,
    ), patch(_INNER_DELEGATE, side_effect=_inner_once):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
                headers=strict_http_auth_headers(),
            )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == TaskState.COMPLETED.value
    assert admit_count == 1
    assert release_count == 1
    assert inner_calls == 1
    assert len(coordinate_calls) == 1
    assert coordinate_calls[0].serving_unit_ref == fixture.unit_ref
    assert successful_reserves == [fixture.unit_ref]
    assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0


def test_c_dim15_02_denied_by_canonical_serving_state_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """DIM15-02 — supersede traffic revision; HTTP fail-closed (extends DIM10-02)."""
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim15-02-stale-serving",
    )
    stores = fixture.session.composition.agent_platform_runtime.stores
    scope_record = stores.serving_store.get_serving_record(
        fixture.application_id,
        fixture.application_environment_id,
    )
    assert scope_record is not None
    stores.serving_store.atomic_swap_serving_revision(
        application_id=fixture.application_id,
        application_environment_id=fixture.application_environment_id,
        expected_current_revision_id=scope_record.traffic_serving_revision_id,
        expected_pointer_revision=scope_record.serving_pointer_revision,
        new_revision_id="rev-c-dim15-02-superseded",
        prior_revision_id=fixture.runtime_revision_id,
        committed_at=scope_record.committed_at,
    )
    successful_reserves: list[int] = []
    real_reserve = fixture.adapter.reserve_unit_execution

    def _recording_reserve(_unit_ref: str):
        reservation = real_reserve(_unit_ref)
        successful_reserves.append(1)
        return reservation

    inner_mock = AsyncMock(return_value=_completed_task_result())
    with patch.object(
        fixture.adapter,
        "reserve_unit_execution",
        side_effect=_recording_reserve,
    ), patch(_INNER_DELEGATE, inner_mock):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
                headers=strict_http_auth_headers(),
            )
    _assert_http_denied_no_execution(
        response,
        inner_mock=inner_mock,
        adapter=fixture.adapter,
        unit_ref=fixture.unit_ref,
        successful_reserves=successful_reserves,
    )


def test_c_dim15_03_forged_deployment_instance_unit_ref_at_http_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """DIM15-03 — mismatched deployment-instance unit ref; real coordinator rejects at HTTP."""
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim15-03-forged-unit",
    )
    wrong_unit = f"{fixture.unit_ref}-non-authoritative"
    deployment_instance_store = (
        fixture.session.lifecycle.lifecycle_services.deployment_instance_store
    )
    instance = deployment_instance_store.get_instance(
        fixture.application_id,
        fixture.application_environment_id,
        fixture.runtime_revision_id,
    )
    assert instance is not None
    deployment_instance_store.update_instance(
        instance.model_copy(update={"serving_unit_ref": wrong_unit}),
        expected_state=instance.instance_state,
    )
    successful_reserves: list[int] = []
    real_reserve = fixture.adapter.reserve_unit_execution

    def _recording_reserve(_unit_ref: str):
        reservation = real_reserve(_unit_ref)
        successful_reserves.append(1)
        return reservation

    inner_mock = AsyncMock(return_value=_completed_task_result())
    with patch.object(
        fixture.adapter,
        "reserve_unit_execution",
        side_effect=_recording_reserve,
    ), patch(_INNER_DELEGATE, inner_mock):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
                headers=strict_http_auth_headers(),
            )
    _assert_http_denied_no_execution(
        response,
        inner_mock=inner_mock,
        adapter=fixture.adapter,
        unit_ref=fixture.unit_ref,
        successful_reserves=successful_reserves,
    )


def test_c_dim15_04_physically_closed_gate_via_adapter_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """DIM15-04 — adapter ``stop()`` closes gate; stale ASGI HTTP denied (DIM14-04 pattern)."""
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim15-04-stopped",
    )
    unit_ref = _stop_colocated_physical_unit(fixture.session, fixture.adapter)
    assert unit_ref == fixture.unit_ref
    successful_reserves: list[int] = []
    real_reserve = fixture.adapter.reserve_unit_execution

    def _recording_reserve(_ref: str):
        reservation = real_reserve(_ref)
        successful_reserves.append(1)
        return reservation

    inner_mock = AsyncMock()
    with patch.object(
        fixture.adapter,
        "reserve_unit_execution",
        side_effect=_recording_reserve,
    ), patch(_INNER_DELEGATE, inner_mock):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
                headers=strict_http_auth_headers(),
            )
    _assert_http_denied_no_execution(
        response,
        inner_mock=inner_mock,
        adapter=fixture.adapter,
        unit_ref=fixture.unit_ref,
        successful_reserves=successful_reserves,
    )


def test_c_dim15_05_concurrent_legitimate_requests_distinct_execution_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """DIM15-05 — two HTTP runs: distinct run/attempt; per-request coordinator admission."""
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim15-05-dual",
    )
    handoff = fixture.session.serving_runtime_handoff
    assert handoff is not None
    admission = handoff.colocated_execution_admission
    assert admission is not None
    coordinator = admission.activation_domain_coordinator
    real_coordinate = coordinator.coordinate_physical_execution_reservation
    coordinate_call_count = 0
    execution_pairs: list[tuple[str, str]] = []

    def _counting_coordinate(request, *, physical_reserve):
        nonlocal coordinate_call_count
        coordinate_call_count += 1
        assert request.serving_unit_ref == fixture.unit_ref
        return real_coordinate(request, physical_reserve=physical_reserve)

    async def _inner_capture(*_args, **_kwargs) -> TaskResult:
        run_id, attempt_id = require_active_execution_identity()
        execution_pairs.append((str(run_id), str(attempt_id)))
        return _completed_task_result()

    with patch.object(
        coordinator,
        "coordinate_physical_execution_reservation",
        side_effect=_counting_coordinate,
    ), patch(_INNER_DELEGATE, side_effect=_inner_capture):
        with TestClient(fixture.app) as client:
            for _ in range(2):
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
                assert response.status_code == 200

    assert len(execution_pairs) == 2
    assert execution_pairs[0] != execution_pairs[1]
    assert coordinate_call_count == 2
    assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0


def test_c_dim15_06_executor_failure_releases_admission_in_finally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """DIM15-06 — inner failure after admit; release in ``finally``; HTTP surfaces error."""
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim15-06-inner-fail",
    )
    inner_mock = AsyncMock(side_effect=RuntimeError("dim15-inner-failure"))

    with patch(_INNER_DELEGATE, inner_mock):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
                headers=strict_http_auth_headers(),
            )

    assert response.status_code == 502
    assert "RuntimeError" in response.json()["detail"]
    inner_mock.assert_called_once()
    assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0


def test_c_dim15_07_physical_serving_admission_closed_denies_without_reserve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """DIM15-07 — closed unit serving admission flag; zero inner and zero successful reserve."""
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim15-07-closed-flag",
    )
    _close_serving_unit_admission(fixture.adapter, fixture.unit_ref)
    successful_reserves: list[int] = []
    real_reserve = fixture.adapter.reserve_unit_execution

    def _recording_reserve(_unit_ref: str):
        reservation = real_reserve(_unit_ref)
        successful_reserves.append(1)
        return reservation

    inner_mock = AsyncMock(return_value=_completed_task_result())
    with patch.object(
        fixture.adapter,
        "reserve_unit_execution",
        side_effect=_recording_reserve,
    ), patch(_INNER_DELEGATE, inner_mock):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
                headers=strict_http_auth_headers(),
            )
    _assert_http_denied_no_execution(
        response,
        inner_mock=inner_mock,
        adapter=fixture.adapter,
        unit_ref=fixture.unit_ref,
        successful_reserves=successful_reserves,
    )
    assert response.json().get("state") is None
