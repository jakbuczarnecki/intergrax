# © Artur Czarnecki. All rights reserved.

"""R7-Q §C dimension 16 — strict serving drain/stop/rollback lifecycle safety."""

from __future__ import annotations

import concurrent.futures
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from intergrax.agent_distribution.deployment import (
    DeploymentInstanceState,
    ServingUnitAttestationError,
    ServingUnitAttestationRequest,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
    _ColocatedVenvPhysicalUnit,
    _UnitAdmissionGate,
)
from intergrax.applications._shared.colocated_serving_runtime_composition import (
    ColocatedServingAdmissionTopologyError,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingLauncherError,
    StrictProductionServingSession,
)
from intergrax.runtime.task.task import TaskResult
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    repo_root,
)
from tests.qualification.prod_q._prod_q_strict_http_caller_auth import (
    strict_http_auth_headers,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_c_dim10_asgi_runtime_identity_continuity_gate import (
    _dim10_http_fixture,
    _run_request_json,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_c_dim14_stale_physical_unit_fail_closed_gate import (
    _attestation_request_for_handoff,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
    _INNER_DELEGATE,
    _completed_task_result,
    _physical_in_flight,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "ceaca27f4b51995ceaf9af9db775eea7d708d800"
_D1_R6_ROLLBACK_EVIDENCE: Final = (
    "tests/unit/agent_distribution/"
    "test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure.py"
)


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


def _assert_production_slot_released() -> None:
    from intergrax.applications._shared import production_process_lifecycle as ppl

    assert ppl._active_production_process_lifecycle is None


def test_c_dim16_g00_qualification_baseline_ancestor() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def test_c_dim16_01_successful_active_session_drain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM16-01 — HTTP ok, drain on close, new execution denied, resources released."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-01-drain",
    ) as fixture:
        unit = fixture.adapter._units[fixture.unit_ref]
        assert unit.serving_admission_authorized is True
        assert unit.admission.admission_closed() is False
        inner_mock = AsyncMock(return_value=_completed_task_result())
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                before = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
                assert before.status_code == 200
                assert inner_mock.call_count == 1
                inner_mock.reset_mock()
                fixture.session.close()
                denied = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert denied.status_code == 502
        inner_mock.assert_not_called()
        assert unit.lifecycle_state is DeploymentInstanceState.DRAINING
        assert unit.admission.admission_closed() is True
        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0
        assert fixture.session._closed is True
        assert fixture.session.lifecycle._process_closed is True
        _assert_production_slot_released()


def test_c_dim16_02_in_flight_execution_during_drain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM16-02 — admitted execution completes; new admission denied after drain begins."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-02-inflight",
    ) as fixture:
        admitted = threading.Event()
        release_in_flight = threading.Event()

        async def _inner_hold(*_args, **_kwargs) -> TaskResult:
            admitted.set()
            assert release_in_flight.wait(timeout=5.0)
            return _completed_task_result()

        deny_mock = AsyncMock(return_value=_completed_task_result())
        with patch(_INNER_DELEGATE, side_effect=_inner_hold):
            with TestClient(fixture.app) as client:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    in_flight = pool.submit(
                        lambda: client.post(
                            fixture.run_url,
                            json=_run_request_json(fixture.tenant_id),
                            headers=strict_http_auth_headers(),
                        ),
                    )
                    assert admitted.wait(timeout=5.0)
                    fixture.session.close()
                    release_in_flight.set()
                    in_flight_response = in_flight.result(timeout=10.0)
                assert in_flight_response.status_code == 200
                assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0
                with patch(_INNER_DELEGATE, deny_mock):
                    after = client.post(
                        fixture.run_url,
                        json=_run_request_json(fixture.tenant_id),
                        headers=strict_http_auth_headers(),
                    )
        assert after.status_code == 502
        deny_mock.assert_not_called()


def test_c_dim16_03_physical_stop_after_drain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM16-03 — sanctioned ``stop()`` after session drain."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-03-stop",
    ) as fixture:
        revision, request, unit_ref = _attestation_request_for_handoff(fixture.session)
        fixture.session.close()
        fixture.adapter.stop(revision, serving_unit_ref=unit_ref)
        unit = fixture.adapter._units[unit_ref]
        assert unit.lifecycle_state is DeploymentInstanceState.STOPPED
        assert not unit.runtime_resource.is_live
        with pytest.raises(ServingUnitAttestationError):
            fixture.adapter.attest_serving_unit(revision, request=request)
        inner_mock = AsyncMock(return_value=_completed_task_result())
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert response.status_code == 502
        inner_mock.assert_not_called()


def test_c_dim16_04_drain_failure_still_cleans_lifecycle_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM16-04 — drain error preserved; lifecycle cleanup and slot release still run."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-04-drain-fail",
    ) as fixture:
        drain_fail = ColocatedServingAdmissionTopologyError("dim16-injected-drain-fail")
        close_calls = 0
        original_close = ProductionProcessLifecycle.close_process_resources

        def _tracked_close(self: ProductionProcessLifecycle) -> None:
            nonlocal close_calls
            close_calls += 1
            original_close(self)

        inner_mock = AsyncMock(return_value=_completed_task_result())
        with patch(
            (
                "intergrax.applications._shared.strict_production_serving_launcher."
                "begin_colocated_serving_execution_drain_on_strict_session_close"
            ),
            side_effect=drain_fail,
        ), patch.object(
            ProductionProcessLifecycle,
            "close_process_resources",
            _tracked_close,
        ), patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                with pytest.raises(
                    StrictProductionServingLauncherError,
                    match="dim16-injected-drain-fail",
                ):
                    fixture.session.close()
                denied = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert close_calls == 1
        assert fixture.session._closed is False
        assert fixture.session.lifecycle._process_closed is True
        _assert_production_slot_released()
        unit = fixture.adapter._units[fixture.unit_ref]
        assert unit.admission.admission_closed() is True
        assert denied.status_code == 502
        inner_mock.assert_not_called()


def test_c_dim16_05_drain_failure_and_cleanup_failure_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM16-05 — primary drain and cleanup errors both preserved."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-05-both-fail",
    ) as fixture:
        drain_fail = ColocatedServingAdmissionTopologyError("dim16-drain-fail")
        cleanup_fail = RuntimeError("dim16-cleanup-fail")
        with patch(
            (
                "intergrax.applications._shared.strict_production_serving_launcher."
                "begin_colocated_serving_execution_drain_on_strict_session_close"
            ),
            side_effect=drain_fail,
        ), patch.object(
            ProductionProcessLifecycle,
            "close_process_resources",
            side_effect=cleanup_fail,
        ) as close_mock:
            with pytest.raises(ExceptionGroup) as exc_info:
                fixture.session.close()
        errors = exc_info.value.exceptions
        assert len(errors) == 2
        assert isinstance(errors[0], StrictProductionServingLauncherError)
        assert errors[1] is cleanup_fail
        assert fixture.session._closed is False
        assert not fixture.session.lifecycle._process_closed
        close_mock.assert_called_once()


def test_c_dim16_06_rollback_compensation_reuses_d1_r6_evidence() -> None:
    """DIM16-06 — strict-session rollback path not duplicated; D1-R6 owns activation rollback."""
    evidence = repo_root() / _D1_R6_ROLLBACK_EVIDENCE
    assert evidence.is_file()
    text = evidence.read_text(encoding="utf-8")
    assert "physical" in text.lower()
    assert "admission" in text.lower()


def test_c_dim16_07_no_resurrection_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM16-07 — drain/stop deny unauthorized reopen without canonical lifecycle."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-07-no-resurrection",
    ) as fixture:
        revision, _request, unit_ref = _attestation_request_for_handoff(fixture.session)
        fixture.session.close()
        unit = fixture.adapter._units[unit_ref]
        assert unit.admission.admission_closed() is True
        fixture.adapter.resume_serving(revision, serving_unit_ref=unit_ref)
        assert unit.serving_admission_authorized is False
        assert unit.admission.admission_closed() is True
        fixture.adapter.stop(revision, serving_unit_ref=unit_ref)
        with pytest.raises(VenvBundleColocatedDeploymentError, match="stopped"):
            fixture.adapter.resume_serving(revision, serving_unit_ref=unit_ref)
        assert unit.lifecycle_state is DeploymentInstanceState.STOPPED
        inner_mock = AsyncMock(return_value=_completed_task_result())
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert response.status_code == 502
        inner_mock.assert_not_called()


def test_c_dim16_08_lifespan_and_repeated_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM16-08 — ASGI lifespan closes once; explicit second close is rejected."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-08-lifespan",
    ) as fixture:
        close_calls = 0
        session = fixture.session
        original_close = StrictProductionServingSession.close

        def _counting_close(self: StrictProductionServingSession) -> None:
            nonlocal close_calls
            close_calls += 1
            original_close(self)

        with patch.object(
            StrictProductionServingSession,
            "close",
            _counting_close,
        ):
            with TestClient(fixture.app):
                pass
        assert close_calls == 1
        assert session._closed is True
        with pytest.raises(StrictProductionServingLauncherError, match="already closed"):
            session.close()


def test_c_dim16_f01_missing_revision_physical_unit_still_revoked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM16-F01 — missing revision record; emergency path revokes bound physical unit."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-f01-missing-rev",
    ) as fixture:
        stores = fixture.session.lifecycle.composition.agent_platform_runtime.stores
        revision_store = stores.revision_store
        original_get = revision_store.get_revision

        def _revision_unavailable(runtime_revision_id: str):
            if runtime_revision_id == fixture.runtime_revision_id:
                return None
            return original_get(runtime_revision_id)

        monkeypatch.setattr(revision_store, "get_revision", _revision_unavailable)
        unit = fixture.adapter._units[fixture.unit_ref]
        assert unit.admission.admission_closed() is False
        inner_mock = AsyncMock(return_value=_completed_task_result())
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                with pytest.raises(
                    StrictProductionServingLauncherError,
                    match="missing runtime revision",
                ):
                    fixture.session.close()
                denied = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert unit.admission.admission_closed() is True
        assert unit.lifecycle_state is DeploymentInstanceState.DRAINING
        assert denied.status_code == 502
        inner_mock.assert_not_called()
        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0
        assert fixture.session._closed is False
        assert fixture.session.lifecycle._process_closed is True
        _assert_production_slot_released()


def test_c_dim16_f02_drain_and_emergency_revocation_both_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM16-F02 — drain and emergency revocation fail; lifecycle cleanup still runs once."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-f02-both-revoke-fail",
    ) as fixture:
        drain_fail = ColocatedServingAdmissionTopologyError("dim16-f02-drain-fail")
        revoke_fail = ColocatedServingAdmissionTopologyError("dim16-f02-revoke-fail")
        close_calls = 0
        original_close = ProductionProcessLifecycle.close_process_resources

        def _tracked_close(self: ProductionProcessLifecycle) -> None:
            nonlocal close_calls
            close_calls += 1
            original_close(self)

        unit = fixture.adapter._units[fixture.unit_ref]
        assert unit.admission.admission_closed() is False
        with patch(
            (
                "intergrax.applications._shared.strict_production_serving_launcher."
                "begin_colocated_serving_execution_drain_on_strict_session_close"
            ),
            side_effect=drain_fail,
        ), patch(
            (
                "intergrax.applications._shared.strict_production_serving_launcher."
                "revoke_colocated_serving_execution_admission_fail_closed_on_strict_session_close"
            ),
            side_effect=revoke_fail,
        ), patch.object(
            ProductionProcessLifecycle,
            "close_process_resources",
            _tracked_close,
        ):
            with pytest.raises(ExceptionGroup) as exc_info:
                fixture.session.close()
        errors = exc_info.value.exceptions
        assert len(errors) == 2
        assert isinstance(errors[0], StrictProductionServingLauncherError)
        assert isinstance(errors[1], StrictProductionServingLauncherError)
        assert "dim16-f02-drain-fail" in str(errors[0])
        assert "dim16-f02-revoke-fail" in str(errors[1])
        assert close_calls == 1
        assert unit.admission.admission_closed() is False
        assert fixture.session._closed is False
        assert fixture.session.lifecycle._process_closed is True
        _assert_production_slot_released()


def test_c_dim16_f02_post_close_01_execution_deny_after_dual_shutdown_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """F02-POST-CLOSE-01 — after dual drain/revoke failure + lifecycle cleanup, HTTP must not execute."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-f02-post-close-01",
    ) as fixture:
        drain_fail = ColocatedServingAdmissionTopologyError("dim16-f02-post-close-drain-fail")
        revoke_fail = ColocatedServingAdmissionTopologyError(
            "dim16-f02-post-close-revoke-fail",
        )
        inner_mock = AsyncMock(return_value=_completed_task_result())
        with patch(
            (
                "intergrax.applications._shared.strict_production_serving_launcher."
                "begin_colocated_serving_execution_drain_on_strict_session_close"
            ),
            side_effect=drain_fail,
        ), patch(
            (
                "intergrax.applications._shared.strict_production_serving_launcher."
                "revoke_colocated_serving_execution_admission_fail_closed_on_strict_session_close"
            ),
            side_effect=revoke_fail,
        ), patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                before = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
                assert before.status_code == 200
                assert inner_mock.call_count == 1
                inner_mock.reset_mock()
                with pytest.raises(ExceptionGroup) as exc_info:
                    fixture.session.close()
                errors = exc_info.value.exceptions
                assert len(errors) == 2
                assert "dim16-f02-post-close-drain-fail" in str(errors[0])
                assert "dim16-f02-post-close-revoke-fail" in str(errors[1])
                unit = fixture.adapter._units[fixture.unit_ref]
                assert unit.admission.admission_closed() is False
                assert fixture.session.lifecycle._process_closed is True
                denied = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert denied.status_code == 502
        inner_mock.assert_not_called()


def test_c_dim16_f03_drain_revoke_and_cleanup_all_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM16-F03 — preserve drain, revocation, and cleanup failures without duplicate cleanup."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-f03-triple-fail",
    ) as fixture:
        drain_fail = ColocatedServingAdmissionTopologyError("dim16-f03-drain-fail")
        revoke_fail = ColocatedServingAdmissionTopologyError("dim16-f03-revoke-fail")
        cleanup_fail = RuntimeError("dim16-f03-cleanup-fail")
        with patch(
            (
                "intergrax.applications._shared.strict_production_serving_launcher."
                "begin_colocated_serving_execution_drain_on_strict_session_close"
            ),
            side_effect=drain_fail,
        ), patch(
            (
                "intergrax.applications._shared.strict_production_serving_launcher."
                "revoke_colocated_serving_execution_admission_fail_closed_on_strict_session_close"
            ),
            side_effect=revoke_fail,
        ), patch.object(
            ProductionProcessLifecycle,
            "close_process_resources",
            side_effect=cleanup_fail,
        ) as close_mock:
            with pytest.raises(ExceptionGroup) as exc_info:
                fixture.session.close()
        errors = exc_info.value.exceptions
        assert len(errors) == 3
        assert isinstance(errors[0], StrictProductionServingLauncherError)
        assert isinstance(errors[1], StrictProductionServingLauncherError)
        assert errors[2] is cleanup_fail
        close_mock.assert_called_once()
        assert fixture.session._closed is False
        assert not fixture.session.lifecycle._process_closed


def test_c_dim16_f06_emergency_revocation_unit_isolation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DIM16-F06 — emergency revocation closes only the bound serving unit."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim16-f06-isolation",
    ) as fixture:
        bound = fixture.adapter._units[fixture.unit_ref]
        sibling_ref = f"{fixture.unit_ref}-sibling"
        sibling_gate = _UnitAdmissionGate()
        sibling_gate.open_admission()
        sibling = replace(
            bound,
            serving_unit_ref=sibling_ref,
            admission=sibling_gate,
            binding_key=f"{bound.binding_key}:sibling",
        )
        fixture.adapter._units[sibling_ref] = sibling
        assert bound.admission.admission_closed() is False
        assert sibling.admission.admission_closed() is False
        fixture.session.close()
        assert bound.admission.admission_closed() is True
        assert sibling.admission.admission_closed() is False
        assert fixture.session._closed is True


# Late import avoids circular patch targets in dim16-04/05.
from intergrax.applications._shared.production_process_lifecycle import (  # noqa: E402
    ProductionProcessLifecycle,
)
