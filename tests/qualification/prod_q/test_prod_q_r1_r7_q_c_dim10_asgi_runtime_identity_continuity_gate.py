# © Artur Czarnecki. All rights reserved.

"""R7-Q §C dimension 10 — ASGI runtime resolver identity continuity (strict production HTTP chain).

D4 lifecycle closure and P2-C HTTP admission identity proofs remain separate, top-level
qualification modules; DIM10 does not replay them via nested pytest subprocesses.
"""

from __future__ import annotations

import concurrent.futures
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace
from typing import Final, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.strict_production_serving import (
    create_governed_contractor_strict_production_asgi_app,
)
from intergrax.agent_distribution.activation_domain_coordinator import (
    ServingExecutionAdmissionRequest,
)
from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.colocated_serving_unit_execution_admission import (
    StoreBackedColocatedServingUnitExecutionAdmission,
)
from intergrax.applications._shared.serving_runtime_handoff import ServingRuntimeHandoffError
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingSession,
)
from intergrax.applications._shared.venv_bundle_runtime_agent_factory_resolver import (
    VenvBundleRuntimeAgentFactoryResolver,
)
from intergrax.applications._shared.wiring import (
    _index_manifest_bindings,
    factory_reference_for_roster_entry,
)
from intergrax.contracts.execution_identity import (
    mint_run_id,
    mint_task_id,
    require_active_execution_identity,
)
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from tests.qualification.prod_q._prod_q_strict_external_work_provider_qualification_config import (
    with_qualified_strict_external_work_provider_pin,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    assert_identity_continuity,
    load_d4_launcher_module,
    repo_root,
)
from tests.qualification.prod_q._prod_q_strict_http_caller_auth import (
    strict_http_auth_headers,
    with_strict_http_api_keys,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
    _INNER_DELEGATE,
    _RUN_ROUTE_SUFFIX,
    _gr6_production_policy_bundle,
    _patch_gca_strict_http_wiring,
    _physical_in_flight,
    _run_request_json,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "39fda3704c330f5eaac953558a541602b45321d0"


@dataclass(frozen=True, slots=True)
class _ResolveFactoryObservation:
    runtime_revision_id: str
    application_id: str
    application_environment_id: str
    materialization_artifact_digest: str | None
    package_digest: str


@dataclass(frozen=True, slots=True)
class _Dim10HttpFixture:
    session: StrictProductionServingSession
    app: FastAPI
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter
    unit_ref: str
    run_url: str
    tenant_id: str
    application_id: str
    application_environment_id: str
    runtime_revision_id: str
    resolver: VenvBundleRuntimeAgentFactoryResolver
    resolve_factory_observations: list[_ResolveFactoryObservation]


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


def test_c_dim10_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def _completed_task_result() -> TaskResult:
    return TaskResult(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        state=TaskState.COMPLETED,
        answer="c-dim10-resolver-continuity",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


def _attach_resolve_factory_observer(
    resolver: VenvBundleRuntimeAgentFactoryResolver,
    observations: list[_ResolveFactoryObservation],
) -> None:
    real_resolve = resolver.resolve_factory

    def _observed_resolve_factory(
        *,
        runtime_revision,
        package_digest: str,
        factory_reference,
    ):
        observations.append(
            _ResolveFactoryObservation(
                runtime_revision_id=runtime_revision.runtime_revision_id,
                application_id=runtime_revision.application_id,
                application_environment_id=runtime_revision.application_environment_id,
                materialization_artifact_digest=runtime_revision.materialization_artifact_digest,
                package_digest=package_digest,
            ),
        )
        return real_resolve(
            runtime_revision=runtime_revision,
            package_digest=package_digest,
            factory_reference=factory_reference,
        )

    resolver.resolve_factory = _observed_resolve_factory  # type: ignore[method-assign]


@contextmanager
def _dim10_http_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    revision_id: str,
) -> Iterator[_Dim10HttpFixture]:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    from testing_support.host_fixture_wiring import (
        install_diagnostic_cursor_secret,
        install_host_llm_stub,
    )

    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)
    _patch_gca_strict_http_wiring(monkeypatch, d4)
    session = d4._open_with_activation(
        tmp_path,
        environment=d4._ac3_environment(),
        revision_id=revision_id,
    )
    try:
        assert session.readiness.ready_for_traffic is True
        handoff = session.serving_runtime_handoff
        assert handoff is not None
        resolver = handoff.runtime_agent_factory_resolver
        assert isinstance(resolver, VenvBundleRuntimeAgentFactoryResolver)
        resolve_factory_observations: list[_ResolveFactoryObservation] = []
        _attach_resolve_factory_observer(resolver, resolve_factory_observations)
        lifecycle_services = session.lifecycle.lifecycle_services
        projection_input = lifecycle_services.projection_input_store.get(revision_id)
        assert projection_input is not None
        assert handoff.runtime_agent_factory_resolver is resolver
        store_resolver = projection_input.factory_resolver
        if (
            store_resolver is not None
            and isinstance(store_resolver, VenvBundleRuntimeAgentFactoryResolver)
            and store_resolver is not resolver
        ):
            _attach_resolve_factory_observer(store_resolver, resolve_factory_observations)

        admission = handoff.colocated_execution_admission
        assert admission is not None
        physical_gate = admission.physical_gate
        assert isinstance(physical_gate.adapter, VenvBundleColocatedRuntimeDeploymentAdapter)
        adapter = physical_gate.adapter
        unit_ref = handoff.identity.serving_unit_ref
        assert_identity_continuity(session, revision_id=revision_id)

        env = session.environment
        tenant_id = env.application_environment_id
        settings = with_qualified_strict_external_work_provider_pin(
            with_strict_http_api_keys(
                replace(
                    GovernedContractorBackendSettings.from_env(),
                    include_mcp=False,
                    include_scheduler=False,
                    include_interaction_routes=False,
                    include_task_control=False,
                    default_agent_id="search",
                    runtime_policy_bundle=_gr6_production_policy_bundle(),
                ),
                execution_tenant_id=tenant_id,
                wrong_tenant_id=f"{tenant_id}-unauthorized-caller",
            ),
        )
        host_runtime = d4._host_runtime_for_startup(settings)
        app = create_governed_contractor_strict_production_asgi_app(
            session,
            settings=settings,
            host_runtime=host_runtime,
        )
        _exercise_handoff_resolver_resolve_factory_observation(
            session=session,
            revision_id=revision_id,
            resolver=resolver,
        )
        yield _Dim10HttpFixture(
            session=session,
            app=app,
            adapter=adapter,
            unit_ref=unit_ref,
            run_url=f"{settings.route_prefix}{_RUN_ROUTE_SUFFIX}",
            tenant_id=tenant_id,
            application_id=env.application_id,
            application_environment_id=env.application_environment_id,
            runtime_revision_id=revision_id,
            resolver=resolver,
            resolve_factory_observations=resolve_factory_observations,
        )
    finally:
        if not session._closed and not session.lifecycle._process_closed:
            session.close()


def _exercise_handoff_resolver_resolve_factory_observation(
    *,
    session: StrictProductionServingSession,
    revision_id: str,
    resolver: VenvBundleRuntimeAgentFactoryResolver,
) -> None:
    """Exercise production resolve_factory on the handoff-bound resolver (activation parity)."""
    projection_input = session.lifecycle.lifecycle_services.projection_input_store.get(
        revision_id,
    )
    assert projection_input is not None
    enabled_entries = [
        entry
        for entry in projection_input.effective_roster.entries
        if entry.effective_enablement
    ]
    assert enabled_entries, "expected at least one enabled roster agent for resolver exercise"
    entry = enabled_entries[0]
    factory_reference = factory_reference_for_roster_entry(
        entry,
        _index_manifest_bindings(projection_input.manifest),
    )
    resolver.resolve_factory(
        runtime_revision=projection_input.runtime_revision,
        package_digest=entry.package_digest,
        factory_reference=factory_reference,
    )


def _assert_observations_match_handoff(
    observations: tuple[_ResolveFactoryObservation, ...],
    *,
    handoff_identity,
) -> None:
    assert observations, "expected resolve_factory during strict ASGI composition"
    for observed in observations:
        assert observed.application_id == handoff_identity.application_id
        assert observed.application_environment_id == handoff_identity.application_environment_id
        assert observed.runtime_revision_id == handoff_identity.runtime_revision_id
        assert (
            observed.materialization_artifact_digest
            == handoff_identity.materialization_artifact_digest
        )


def test_c_dim10_01_positive_resolver_continuity_through_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-01-positive",
    ) as fixture:
        handoff = fixture.session.serving_runtime_handoff
        assert handoff is not None
        authoritative = handoff.identity
        _assert_observations_match_handoff(
            tuple(fixture.resolve_factory_observations),
            handoff_identity=authoritative,
        )
        assert fixture.resolver.artifact_root.is_dir()

        inner_calls = 0

        async def _inner_once(*_args, **_kwargs) -> TaskResult:
            nonlocal inner_calls
            inner_calls += 1
            run_id, attempt_id = require_active_execution_identity()
            assert run_id
            assert attempt_id
            return _completed_task_result()

        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0
        with patch(_INNER_DELEGATE, side_effect=_inner_once):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert response.status_code == 200
        assert inner_calls == 1
        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0


def test_c_dim10_02_wrong_runtime_revision_fail_closed_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-02-wrong-revision",
    ) as fixture:
        handoff = fixture.session.serving_runtime_handoff
        assert handoff is not None
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
            new_revision_id="rev-c-dim10-02-superseded",
            prior_revision_id=fixture.runtime_revision_id,
            committed_at=scope_record.committed_at,
        )
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
        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0


def test_c_dim10_03_wrong_serving_unit_fail_closed_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-03-wrong-unit",
    ) as fixture:
        handoff = fixture.session.serving_runtime_handoff
        assert handoff is not None
        admission = handoff.colocated_execution_admission
        assert admission is not None
        assert isinstance(admission, StoreBackedColocatedServingUnitExecutionAdmission)
        wrong_unit = f"{fixture.unit_ref}-non-authoritative"
        with pytest.raises(ServingRuntimeHandoffError, match="disagrees with handoff identity"):
            admission.admit_execution(wrong_unit)

        inner_mock = AsyncMock(return_value=_completed_task_result())
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
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert response.status_code == 502
        inner_mock.assert_not_called()
        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0


def test_c_dim10_04_stable_deployment_identity_distinct_execution_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-04-repeat",
    ) as fixture:
        handoff = fixture.session.serving_runtime_handoff
        assert handoff is not None
        authoritative = handoff.identity
        composition_calls_before = len(fixture.resolve_factory_observations)
        execution_pairs: list[tuple[str, str]] = []

        async def _inner_capture(*_args, **_kwargs) -> TaskResult:
            run_id, attempt_id = require_active_execution_identity()
            assert run_id
            assert attempt_id
            execution_pairs.append((str(run_id), str(attempt_id)))
            return _completed_task_result()

        with patch(_INNER_DELEGATE, side_effect=_inner_capture):
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
        assert handoff.identity == authoritative
        assert fixture.resolver is handoff.runtime_agent_factory_resolver
        assert len(fixture.resolve_factory_observations) == composition_calls_before


def test_c_dim10_05_http_serving_unit_admission_matches_handoff_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-05-serving-unit",
    ) as fixture:
        handoff = fixture.session.serving_runtime_handoff
        assert handoff is not None
        authoritative = handoff.identity
        expected_unit_ref = fixture.session.serving_runtime_handoff.identity.serving_unit_ref
        assert expected_unit_ref == authoritative.serving_unit_ref
        assert expected_unit_ref == fixture.unit_ref

        admission = handoff.colocated_execution_admission
        assert admission is not None
        coordinator = admission.activation_domain_coordinator
        real_coordinate = coordinator.coordinate_physical_execution_reservation
        admit_execution_serving_unit_refs: list[str] = []
        real_admit = StoreBackedColocatedServingUnitExecutionAdmission.admit_execution

        def _recording_coordinate_physical_execution_reservation(
            request: ServingExecutionAdmissionRequest,
            *,
            physical_reserve,
        ):
            assert request.serving_unit_ref == authoritative.serving_unit_ref
            assert request.serving_unit_ref == expected_unit_ref
            return real_coordinate(request, physical_reserve=physical_reserve)

        def _recording_admit(
            self: StoreBackedColocatedServingUnitExecutionAdmission,
            serving_unit_ref: str,
        ) -> None:
            admit_execution_serving_unit_refs.append(serving_unit_ref)
            real_admit(self, serving_unit_ref)

        inner_calls = 0

        async def _inner_once(*_args, **_kwargs) -> TaskResult:
            nonlocal inner_calls
            inner_calls += 1
            return _completed_task_result()

        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0
        with patch.object(
            coordinator,
            "coordinate_physical_execution_reservation",
            side_effect=_recording_coordinate_physical_execution_reservation,
        ), patch.object(
            StoreBackedColocatedServingUnitExecutionAdmission,
            "admit_execution",
            _recording_admit,
        ), patch(_INNER_DELEGATE, side_effect=_inner_once):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert response.status_code == 200
        assert inner_calls == 1
        assert admit_execution_serving_unit_refs == [expected_unit_ref]
        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0


def test_c_dim10_close_01_http_after_explicit_session_close_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """DIM10-CLOSE-01 — close then request."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-close-01",
    ) as fixture:
        inner_mock = AsyncMock(return_value=_completed_task_result())
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                before_close = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
                assert before_close.status_code == 200
                assert inner_mock.call_count == 1
                inner_mock.reset_mock()
                fixture.session.close()
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert response.status_code == 502
        inner_mock.assert_not_called()
        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0
        assert fixture.session._closed is True


def test_c_dim10_close_02_repeated_http_after_session_close_stays_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """DIM10-CLOSE-02 — repeated request after close."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-close-02",
    ) as fixture:
        inner_mock = AsyncMock(return_value=_completed_task_result())
        fixture.session.close()
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                for _ in range(3):
                    response = client.post(
                        fixture.run_url,
                        json=_run_request_json(fixture.tenant_id),
                        headers=strict_http_auth_headers(),
                    )
                    assert response.status_code == 502
        inner_mock.assert_not_called()
        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0


def test_c_dim10_close_03_shutdown_during_in_flight_then_post_close_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """DIM10-CLOSE-03 — shutdown/admission ordering."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-close-03",
    ) as fixture:
        admitted = threading.Event()
        release_in_flight = threading.Event()

        async def _inner_hold_in_flight(*_args, **_kwargs) -> TaskResult:
            admitted.set()
            assert release_in_flight.wait(timeout=5.0)
            return _completed_task_result()

        inner_mock = AsyncMock(return_value=_completed_task_result())
        with patch(_INNER_DELEGATE, side_effect=_inner_hold_in_flight):
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
                with patch(_INNER_DELEGATE, inner_mock):
                    after_close = client.post(
                        fixture.run_url,
                        json=_run_request_json(fixture.tenant_id),
                        headers=strict_http_auth_headers(),
                    )
        assert after_close.status_code == 502
        inner_mock.assert_not_called()


def test_c_dim10_close_04_scope_single_bound_unit_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """DIM10-CLOSE-04 — strict production is one session / one bound unit per process slot."""
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-close-04",
    ) as fixture:
        other_unit_ref = f"{fixture.unit_ref}-unbound-peer"
        assert other_unit_ref not in fixture.adapter._units
        fixture.session.close()
        unit = fixture.adapter._units[fixture.unit_ref]
        assert unit.lifecycle_state is DeploymentInstanceState.DRAINING
        assert unit.admission.admission_closed() is True


def test_c_dim10_06_http_after_explicit_session_close_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    test_c_dim10_close_01_http_after_explicit_session_close_denied(
        monkeypatch,
        tmp_path,
    )
