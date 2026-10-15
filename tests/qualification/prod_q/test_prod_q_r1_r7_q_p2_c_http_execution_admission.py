# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-P2-C — strict HTTP → colocated execution admission continuity."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingSession,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle
from testing_support.host_fixture_wiring import (
    host_wiring_test_environment,
    install_diagnostic_cursor_secret,
    install_host_llm_stub,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    assert_identity_continuity,
    load_d4_launcher_module,
    repo_root,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "3d1a603f9454d17bd15f28c07a9e7731bae2b3de"
_RUN_ROUTE_SUFFIX: Final = "/run"
_INNER_DELEGATE = (
    "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute"
)


@dataclass(frozen=True, slots=True)
class _StrictHttpAdmissionFixture:
    session: StrictProductionServingSession
    app: FastAPI
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter
    unit_ref: str
    run_url: str
    tenant_id: str
    application_id: str
    application_environment_id: str
    runtime_revision_id: str


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


def test_p2c_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def _strict_http_environment_profile(
    _settings: GovernedContractorBackendSettings,
) -> ApplicationEnvironmentProfile:
    from intergrax.applications._shared.strict_host_dependency_concurrency_admission import (
        build_dependency_concurrency_admission_for_tool_ids,
    )
    from tests.unit.applications.test_production_registry_startup_ac3_final import _ENV

    profile = host_wiring_test_environment(
        ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV),
    )
    return profile.model_copy(
        update={
            "observability_profile": profile.observability_profile.model_copy(
                update={"otel_enabled": False},
            ),
            "reliability_profile": profile.reliability_profile.model_copy(
                update={
                    "dependency_concurrency_admission": (
                        build_dependency_concurrency_admission_for_tool_ids(
                            ("qualification-http-tool",),
                        )
                    ),
                },
            ),
        },
    )


def _ac3_serving_manifest_for_package_closure() -> ApplicationManifest:
    """GCA HTTP host manifest: declarative contract_id matches Echo AgentContract.id."""
    from echo.echo_agent import EchoAgent
    from tests.unit.applications.test_production_registry_startup_ac3_final import _APP

    return ApplicationManifest.lab(
        app_id=_APP,
        name="AC3 Final",
        agents=[
            AgentBinding.mount(
                EchoAgent,
                contract_id="echo",
                capabilities=["echo.basic"],
            ),
        ],
    )


def _gr6_production_policy_bundle() -> ImmutableRuntimePolicyBundle:
    from applications.governed_contractor_application.tests.host.test_gr6_wire_production_decision_governance import (
        _test_policy_bundle,
    )

    bundle = _test_policy_bundle()
    if not isinstance(bundle, ImmutableRuntimePolicyBundle):
        raise TypeError("expected ImmutableRuntimePolicyBundle from gr6 test helper")
    return bundle


def _patch_gca_strict_http_wiring(monkeypatch: pytest.MonkeyPatch, d4) -> None:
    """AC3-aligned manifest (D4) + wiring-safe env; canonical package closure unfiltered."""
    d4._patch_gca_manifest_for_ac3_serving(monkeypatch)
    for module in (
        "governed_contractor_application.host.main",
        "governed_contractor_application.host.factory",
    ):
        monkeypatch.setattr(
            f"{module}.build_governed_contractor_manifest",
            _ac3_serving_manifest_for_package_closure,
        )
        monkeypatch.setattr(
            f"{module}.build_governed_contractor_environment_profile",
            _strict_http_environment_profile,
        )


def _physical_in_flight(
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
    serving_unit_ref: str,
) -> int:
    unit = adapter._units[serving_unit_ref]
    return unit.admission.in_flight_count()


def _completed_task_result() -> TaskResult:
    return TaskResult(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        state=TaskState.COMPLETED,
        answer="p2c-admission-proof",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


def _run_request_json(tenant_id: str) -> dict[str, str]:
    return {
        "tenant_id": tenant_id,
        "user_id": "p2c-operator",
        "message": "http admission continuity",
        "capability": "external_contractor.adapt",
    }


def _build_strict_http_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    revision_id: str,
) -> _StrictHttpAdmissionFixture:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)

    _patch_gca_strict_http_wiring(monkeypatch, d4)
    session = d4._open_with_activation(
        tmp_path,
        environment=d4._ac3_environment(),
        revision_id=revision_id,
    )
    assert session.readiness.ready_for_traffic is True
    assert session.composition.serving_runtime_handoff is not None

    settings = replace(
        GovernedContractorBackendSettings.from_env(),
        include_mcp=False,
        include_scheduler=False,
        include_interaction_routes=False,
        include_task_control=False,
        default_agent_id="search",
        runtime_policy_bundle=_gr6_production_policy_bundle(),
    )
    host_runtime = d4._host_runtime_for_startup(settings)
    app = create_governed_contractor_strict_production_asgi_app(
        session,
        settings=settings,
        host_runtime=host_runtime,
    )

    handoff = session.serving_runtime_handoff
    assert handoff is not None
    admission = handoff.colocated_execution_admission
    assert admission is not None
    physical_gate = admission.physical_gate
    assert isinstance(physical_gate.adapter, VenvBundleColocatedRuntimeDeploymentAdapter)
    adapter = physical_gate.adapter
    unit_ref = handoff.identity.serving_unit_ref
    assert_identity_continuity(session, revision_id=revision_id)

    env = session.environment
    tenant_id = env.application_environment_id
    return _StrictHttpAdmissionFixture(
        session=session,
        app=app,
        adapter=adapter,
        unit_ref=unit_ref,
        run_url=f"{settings.route_prefix}{_RUN_ROUTE_SUFFIX}",
        tenant_id=tenant_id,
        application_id=env.application_id,
        application_environment_id=env.application_environment_id,
        runtime_revision_id=revision_id,
    )


def _close_serving_unit_admission(
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
    serving_unit_ref: str,
) -> None:
    # Physical admission seam (D1 R6 serving-admission proofs); authority port unchanged.
    unit = adapter._units[serving_unit_ref]
    unit.serving_admission_authorized = False


def test_p2c_01_positive_http_admission_continuity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p2c-01-positive",
    )
    inner_calls: list[int] = []

    async def _inner_with_in_flight_assertion(*_args, **_kwargs) -> TaskResult:
        assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 1
        inner_calls.append(1)
        return _completed_task_result()

    assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0
    with patch(
        _INNER_DELEGATE,
        new_callable=AsyncMock,
        side_effect=_inner_with_in_flight_assertion,
    ):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
            )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == TaskState.COMPLETED.value
    assert body["answer"] == "p2c-admission-proof"
    assert len(inner_calls) == 1
    assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0
    assert fixture.session.composition.serving_runtime_handoff is not None
    assert (
        fixture.session.composition.serving_runtime_handoff.colocated_execution_admission
        is not None
    )


def test_p2c_02_admission_deny_prevents_inner_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p2c-02-deny",
    )
    _close_serving_unit_admission(fixture.adapter, fixture.unit_ref)
    inner_mock = AsyncMock(return_value=_completed_task_result())

    with patch(_INNER_DELEGATE, inner_mock):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
            )

    assert response.status_code == 502
    assert response.json()["detail"].startswith("run_error:")
    inner_mock.assert_not_called()
    assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0


def test_p2c_03_release_after_successful_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p2c-03-release-ok",
    )
    with patch(
        _INNER_DELEGATE,
        new_callable=AsyncMock,
        return_value=_completed_task_result(),
    ):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
            )
    assert response.status_code == 200
    assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0


def test_p2c_04_release_after_inner_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p2c-04-release-fail",
    )
    inner_mock = AsyncMock(side_effect=RuntimeError("p2c-inner-failure"))

    with patch(_INNER_DELEGATE, inner_mock):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
            )

    assert response.status_code == 502
    assert "RuntimeError" in response.json()["detail"]
    inner_mock.assert_called_once()
    assert _physical_in_flight(fixture.adapter, fixture.unit_ref) == 0


def test_p2c_05_identity_and_scope_continuity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p2c-05-identity",
    )
    handoff = fixture.session.serving_runtime_handoff
    assert handoff is not None
    identity = handoff.identity
    assert identity.application_id == fixture.application_id
    assert identity.application_environment_id == fixture.application_environment_id
    assert identity.runtime_revision_id == fixture.runtime_revision_id
    assert identity.serving_unit_ref == fixture.unit_ref
    assert fixture.tenant_id == fixture.application_environment_id
    assert fixture.application_id != fixture.tenant_id

    with patch(
        _INNER_DELEGATE,
        new_callable=AsyncMock,
        return_value=_completed_task_result(),
    ):
        with TestClient(fixture.app) as client:
            response = client.post(
                fixture.run_url,
                json=_run_request_json(fixture.tenant_id),
            )
    assert response.status_code == 200
