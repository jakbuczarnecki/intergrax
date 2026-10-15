# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-P3-A — cross-tenant strict production activation and HTTP admission."""

from __future__ import annotations

from dataclasses import replace
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.strict_production_serving import (
    create_governed_contractor_strict_production_asgi_app,
)
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
    authorize_scoped_control_plane_mutation,
    build_admit_runtime_revision_mutation_request,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_reference_production_control_plane_mutation_boundary,
)
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications._shared.production_process_lifecycle import (
    open_production_process_lifecycle,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionActivationIntent,
    StrictProductionServingEnvironment,
    StrictProductionServingLauncherError,
    open_strict_production_serving_session,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from testing_support.host_fixture_wiring import (
    install_diagnostic_cursor_secret,
    install_host_llm_stub,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    load_d4_launcher_module,
    repo_root,
)
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _APP,
    _ENV,
    _ac3_revision_bound_host_manifest,
    _activation_request,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "5d6d688fb006807beca151599210fc14b5510956"
_TENANT_A: Final = "p3a-tenant-a"
_TENANT_B: Final = "p3a-tenant-b"
_RUN_ROUTE_SUFFIX: Final = "/run"
_INNER_DELEGATE = (
    "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute"
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


def test_p3a_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def _production_governance(
    *,
    principal_tenant_id: str,
    environment_profile_id: str,
    resolver_tenant_id: str,
) -> ProductionControlPlaneGovernance:
    env = ApplicationEnvironmentProfile(profile_id=environment_profile_id)
    principal = RequestIdentity(
        tenant_id=principal_tenant_id,
        user_id="p3a-production-launcher",
        principal_type=PrincipalType.SERVICE,
        auth_subject="p3a-production-launcher",
    )
    return ProductionControlPlaneGovernance(
        principal=principal,
        mutation_authorization_boundary=build_reference_production_control_plane_mutation_boundary(
            env,
        ),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            tenant_id=resolver_tenant_id,
        ),
    )


def _serving_environment() -> StrictProductionServingEnvironment:
    return StrictProductionServingEnvironment(
        application_id=_APP,
        application_environment_id=_ENV,
    )


def _traffic_revision_id(stores) -> str | None:
    record = stores.serving_store.get_serving_record(_APP, _ENV)
    if record is None:
        return None
    return record.traffic_serving_revision_id


def _physical_unit_count(adapter: VenvBundleColocatedRuntimeDeploymentAdapter) -> int:
    return len(adapter._units)


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
        answer="p3a-admission-positive",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


def _run_request_json(tenant_id: str) -> dict[str, str]:
    return {
        "tenant_id": tenant_id,
        "user_id": "p3a-operator",
        "message": "cross-tenant admission probe",
        "capability": "external_contractor.adapt",
    }


def _strict_http_environment_profile(
    _settings: GovernedContractorBackendSettings,
) -> ApplicationEnvironmentProfile:
    from intergrax.applications._shared.strict_host_dependency_concurrency_admission import (
        build_dependency_concurrency_admission_for_tool_ids,
    )

    profile = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
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


def _patch_gca_strict_http_wiring(monkeypatch: pytest.MonkeyPatch, d4) -> None:
    d4._patch_gca_manifest_for_ac3_serving(monkeypatch)
    for module in (
        "governed_contractor_application.host.main",
        "governed_contractor_application.host.factory",
    ):
        monkeypatch.setattr(
            f"{module}.build_governed_contractor_manifest",
            _ac3_revision_bound_host_manifest,
        )
        monkeypatch.setattr(
            f"{module}.build_governed_contractor_environment_profile",
            _strict_http_environment_profile,
        )


def test_p3t01_cross_tenant_activation_blocked_before_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Tenant A principal cannot activate tenant B environment via strict opener."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"

    environment = _serving_environment()
    storage, adapter = d4._storage_and_adapter(tmp_path)
    governance_a_on_b = _production_governance(
        principal_tenant_id=_TENANT_A,
        environment_profile_id=environment.application_environment_id,
        resolver_tenant_id=_TENANT_B,
    )
    units_before = _physical_unit_count(adapter)

    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance_a_on_b,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=d4._composition_builder(tmp_path),
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        bundle, locator, *_rest = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-p3a-t01-cross",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        stores = lifecycle.composition.agent_platform_runtime.stores
        traffic_before = _traffic_revision_id(stores)

        authz = authorize_scoped_control_plane_mutation(
            boundary=governance_a_on_b.mutation_authorization_boundary,
            tenant_resolver=governance_a_on_b.environment_tenant_resolver,
            request=build_admit_runtime_revision_mutation_request(
                principal=governance_a_on_b.principal,
                application_id=environment.application_id,
                application_environment_id=environment.application_environment_id,
                mutation_id="mut-p3a-t01",
                runtime_revision_id=bundle.runtime_revision.runtime_revision_id,
                identity_digest="digest-p3a-t01",
            ),
        )
        assert not authz.permitted
        assert authz.decision.action is PolicyAction.DENY
        assert authz.decision.policy_rule_id == "agent_distribution.tenant_scope"
        assert authz.decision.reason == "tenant_authority_mismatch"

    intent = StrictProductionActivationIntent(
        projection_input=bundle,
        activation_request=_activation_request(bundle, locator),
        admission_mutation_id=reference_admission_mutation_id(
            bundle.runtime_revision.runtime_revision_id,
        ),
    )
    session = None
    try:
        with pytest.raises(StrictProductionServingLauncherError):
            session = open_strict_production_serving_session(
                storage=storage,
                environment=environment,
                governance=governance_a_on_b,
                deployment_adapter=adapter,
                activation_intent=intent,
                composition_builder=d4._composition_builder(tmp_path),
                missing_continuation_file_policy="empty",
            )
    finally:
        if session is not None:
            session.close()

    traffic_after = _traffic_revision_id(stores)
    assert traffic_after == traffic_before
    assert _physical_unit_count(adapter) == units_before


def test_p3t01_legitimate_tenant_b_state_unchanged_after_cross_tenant_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"

    environment = _serving_environment()
    storage, adapter = d4._storage_and_adapter(tmp_path)
    governance_b = _production_governance(
        principal_tenant_id=_TENANT_B,
        environment_profile_id=environment.application_environment_id,
        resolver_tenant_id=_TENANT_B,
    )
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance_b,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=d4._composition_builder(tmp_path),
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        bundle_baseline, locator_baseline, *_rest = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-p3a-t01-baseline",
            marker="baseline",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        d4._activate_bundle(
            lifecycle.launcher,
            bundle_baseline,
            locator_baseline,
            principal=governance_b.principal,
        )
        stores = lifecycle.composition.agent_platform_runtime.stores

    baseline_session = open_strict_production_serving_session(
        storage=storage,
        environment=environment,
        governance=governance_b,
        deployment_adapter=adapter,
        composition_builder=d4._composition_builder(tmp_path),
        missing_continuation_file_policy="empty",
    )
    stores = baseline_session.composition.agent_platform_runtime.stores
    baseline_traffic = _traffic_revision_id(stores)
    assert baseline_traffic == "rev-p3a-t01-baseline"
    baseline_session.close()

    governance_a_on_b = _production_governance(
        principal_tenant_id=_TENANT_A,
        environment_profile_id=environment.application_environment_id,
        resolver_tenant_id=_TENANT_B,
    )
    bundle, locator, *_rest = d4._qualified_canonical_bundle(
        tmp_path,
        revision_id="rev-p3a-t01-adversarial",
        marker="adv",
        application_id=environment.application_id,
        application_environment_id=environment.application_environment_id,
        stores=stores,
    )
    intent = StrictProductionActivationIntent(
        projection_input=bundle,
        activation_request=_activation_request(bundle, locator),
        admission_mutation_id=reference_admission_mutation_id(
            bundle.runtime_revision.runtime_revision_id,
        ),
    )
    with pytest.raises(StrictProductionServingLauncherError):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance_a_on_b,
            deployment_adapter=adapter,
            activation_intent=intent,
            composition_builder=d4._composition_builder(tmp_path),
            missing_continuation_file_policy="empty",
        )

    assert _traffic_revision_id(stores) == baseline_traffic
    assert baseline_session.readiness.ready_for_traffic is True


def test_p3t02_wrong_tenant_http_does_not_invoke_inner_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)
    _patch_gca_strict_http_wiring(monkeypatch, d4)

    environment = _serving_environment()
    storage, adapter = d4._storage_and_adapter(tmp_path)
    governance_b = _production_governance(
        principal_tenant_id=_TENANT_B,
        environment_profile_id=environment.application_environment_id,
        resolver_tenant_id=_TENANT_B,
    )

    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance_b,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=d4._composition_builder(tmp_path),
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        bundle, locator, *_rest = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-p3a-t02-http",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        d4._activate_bundle(
            lifecycle.launcher,
            bundle,
            locator,
            principal=governance_b.principal,
        )

    session = open_strict_production_serving_session(
        storage=storage,
        environment=environment,
        governance=governance_b,
        deployment_adapter=adapter,
        composition_builder=d4._composition_builder(tmp_path),
        missing_continuation_file_policy="empty",
    )
    inner_mock = AsyncMock(return_value=_completed_task_result())
    try:
        from applications.governed_contractor_application.tests.host.test_gr6_wire_production_decision_governance import (
            _test_policy_bundle,
        )

        settings = replace(
            GovernedContractorBackendSettings.from_env(),
            include_mcp=False,
            include_scheduler=False,
            include_interaction_routes=False,
            include_task_control=False,
            default_agent_id="search",
            runtime_policy_bundle=_test_policy_bundle(),
        )
        host_runtime = d4._host_runtime_for_startup(settings)
        app = create_governed_contractor_strict_production_asgi_app(
            session,
            settings=settings,
            host_runtime=host_runtime,
        )
        handoff = session.serving_runtime_handoff
        assert handoff is not None
        unit_ref = handoff.identity.serving_unit_ref
        run_url = f"{settings.route_prefix}{_RUN_ROUTE_SUFFIX}"
        authoritative_host_tenant = environment.application_environment_id

        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(app) as client:
                wrong = client.post(run_url, json=_run_request_json(_TENANT_A))
                inner_mock.assert_not_called()
                assert _physical_in_flight(adapter, unit_ref) == 0

                positive = client.post(
                    run_url,
                    json=_run_request_json(authoritative_host_tenant),
                )

        assert positive.status_code == 200
        assert positive.json()["state"] == TaskState.COMPLETED.value
        assert inner_mock.call_count == 1
        assert _physical_in_flight(adapter, unit_ref) == 0
        assert wrong.status_code in {200, 502}
        if wrong.status_code == 200:
            assert wrong.json()["state"] != TaskState.COMPLETED.value
    finally:
        if not session._closed:
            session.close()
