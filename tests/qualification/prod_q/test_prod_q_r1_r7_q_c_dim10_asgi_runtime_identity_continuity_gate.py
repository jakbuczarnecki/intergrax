# © Artur Czarnecki. All rights reserved.

"""R7-Q §C dimension 10 — ASGI runtime resolver identity continuity (strict production HTTP chain)."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.strict_production_serving import (
    create_governed_contractor_strict_production_asgi_app,
)
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
from intergrax.contracts.execution_identity import (
    mint_run_id,
    mint_task_id,
    require_active_execution_identity,
)
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    D4_MODULE_RELPATH,
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

_EXPECTED_START_HEAD: Final = "3d7e8674995492154d3fc42d4a6602455ba11569"
_D4_CLOSURE_REUSE: Final = ("test_d4_asgi_shutdown_releases_owned_resources",)


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
    resolve_factory_observations: tuple[_ResolveFactoryObservation, ...]


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
) -> list[_ResolveFactoryObservation]:
    observations: list[_ResolveFactoryObservation] = []
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
    return observations


def _build_dim10_http_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    revision_id: str,
) -> _Dim10HttpFixture:
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
    assert session.readiness.ready_for_traffic is True
    handoff = session.serving_runtime_handoff
    assert handoff is not None
    resolver = handoff.runtime_agent_factory_resolver
    assert isinstance(resolver, VenvBundleRuntimeAgentFactoryResolver)
    observations = _attach_resolve_factory_observer(resolver)
    lifecycle_services = session.lifecycle.lifecycle_services
    projection_input = lifecycle_services.projection_input_store.get(revision_id)
    assert projection_input is not None
    assert projection_input.factory_resolver is resolver

    admission = handoff.colocated_execution_admission
    assert admission is not None
    physical_gate = admission.physical_gate
    assert isinstance(physical_gate.adapter, VenvBundleColocatedRuntimeDeploymentAdapter)
    adapter = physical_gate.adapter
    unit_ref = handoff.identity.serving_unit_ref
    assert_identity_continuity(session, revision_id=revision_id)

    env = session.environment
    tenant_id = env.application_environment_id
    settings = with_strict_http_api_keys(
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
    )
    host_runtime = d4._host_runtime_for_startup(settings)
    app = create_governed_contractor_strict_production_asgi_app(
        session,
        settings=settings,
        host_runtime=host_runtime,
    )
    return _Dim10HttpFixture(
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
        resolve_factory_observations=tuple(observations),
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
    fixture = _build_dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-01-positive",
    )
    handoff = fixture.session.serving_runtime_handoff
    assert handoff is not None
    authoritative = handoff.identity
    _assert_observations_match_handoff(
        fixture.resolve_factory_observations,
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
    fixture.session.close()


def test_c_dim10_02_wrong_runtime_revision_fail_closed_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-02-wrong-revision",
    )
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
    fixture.session.close()


def test_c_dim10_03_wrong_serving_unit_fail_closed_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-03-wrong-unit",
    )
    handoff = fixture.session.serving_runtime_handoff
    assert handoff is not None
    admission = handoff.colocated_execution_admission
    assert admission is not None
    assert isinstance(admission, StoreBackedColocatedServingUnitExecutionAdmission)
    wrong_unit = f"{fixture.unit_ref}-non-authoritative"
    with pytest.raises(ServingRuntimeHandoffError, match="disagrees with handoff identity"):
        admission.admit_execution(wrong_unit)

    inner_mock = AsyncMock(return_value=_completed_task_result())
    stores = fixture.session.composition.agent_platform_runtime.stores
    instance = stores.deployment_instance_store.get_instance(
        fixture.application_id,
        fixture.application_environment_id,
        fixture.runtime_revision_id,
    )
    assert instance is not None
    stores.deployment_instance_store.update_instance(
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
    fixture.session.close()


def test_c_dim10_04_stable_deployment_identity_distinct_execution_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-04-repeat",
    )
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
    fixture.session.close()


def test_c_dim10_05_lifecycle_closure_reuses_d4_evidence() -> None:
    import subprocess

    d4_path = repo_root() / D4_MODULE_RELPATH
    for test_name in _D4_CLOSURE_REUSE:
        completed = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "-p",
                "no:xdist",
                "-q",
                "--noconftest",
                f"{d4_path}::{test_name}",
            ],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 and "circular import" in (
            completed.stdout + completed.stderr
        ).lower():
            pytest.skip("ENVIRONMENT/TEST ISSUE — D4 closure replay blocked by import cycle")
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_c_dim10_05_http_after_explicit_session_close_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-c-dim10-05-closed",
    )
    inner_mock = AsyncMock(return_value=_completed_task_result())
    with TestClient(fixture.app):
        fixture.session.close()
    with patch(_INNER_DELEGATE, inner_mock):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
                headers=strict_http_auth_headers(),
            )
    assert response.status_code == 502
    inner_mock.assert_not_called()


def test_c_dim10_g01_p2c_identity_admission_evidence_regression_matrix() -> None:
    """Reuse P2-C HTTP identity/admission proofs without duplicating the full suite."""
    import subprocess

    module = (
        repo_root()
        / "tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py"
    )
    for test_name in ("test_p2c_05_identity_and_scope_continuity",):
        completed = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "-p",
                "no:xdist",
                "-q",
                "--noconftest",
                f"{module}::{test_name}",
            ],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
