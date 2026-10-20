# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D4 — strict production GCA launcher, handoff, ASGI lifecycle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from governed_contractor_application.host.agent_builders import (
    build_governed_contractor_agent_builders,
)
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    compose_governed_contractor_host_runtime,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.strict_gca_production_process_composition import (
    build_strict_durable_gca_composition_builder,
    open_owned_sqlite_governed_external_work_host_stores,
)
from governed_contractor_application.host.strict_production_serving import (
    create_governed_contractor_strict_production_asgi_app,
)
from governed_contractor_application.manifest import build_governed_contractor_manifest
from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceState,
    FakeInMemoryRuntimeDeploymentAdapter,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.production_process_lifecycle import (
    ProductionProcessLifecycle,
    ProductionProcessLifecycleError,
    open_production_process_lifecycle,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionActivationIntent,
    StrictProductionProcessStorage,
    StrictProductionServingEnvironment,
    StrictProductionServingLauncherError,
    StrictProductionServingReadiness,
    open_strict_production_serving_session,
)
from intergrax.applications._shared.deploy_activate_pipeline import (
    ReferenceProductionLifecycleGovernanceBlockedError,
)
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.runtime.execution.continuation.durable_serialized_seal_store import (
    SerializedSealExecutionContinuationStateStore,
)
from tests.unit.applications.test_prod_q_r1_r7_d2_production_process_lifecycle import (
    _governance,
)
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _APP,
    _ENV,
    _ac3_revision_bound_strict_external_work_host_manifest,
    _activate_bundle,
    _activation_request,
    _authority,
    _build_echo_artifact,
    _manifest,
    _revision,
    _roster,
    _roster_with_strict_external_work,
    _seed_canonical_authority,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from testing_support.host_fixture_wiring import (
    install_diagnostic_cursor_secret,
    install_host_llm_stub,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


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

_GCA_APP = build_governed_contractor_manifest().app_id
_GCA_ENV = build_governed_contractor_environment_profile(
    GovernedContractorBackendSettings.from_env(),
).profile_id


def _storage_and_adapter(
    tmp_path: Path,
) -> tuple[StrictProductionProcessStorage, VenvBundleColocatedRuntimeDeploymentAdapter]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    roots = (tmp_path,)
    storage = StrictProductionProcessStorage(
        agent_distribution_db=tmp_path / "ad.db",
        continuation_state_file=tmp_path / "continuation.json",
        provider_invocation_db=tmp_path / "invocation.db",
        governed_external_work_host_db=tmp_path / "gca_host.db",
        trusted_artifact_parent_roots=roots,
    )
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=roots,
        allow_qualification_locators=True,
    )
    return storage, adapter


def _gca_environment() -> StrictProductionServingEnvironment:
    return StrictProductionServingEnvironment(
        application_id=_GCA_APP,
        application_environment_id=_GCA_ENV,
    )


def _qualified_canonical_bundle(
    tmp_path: Path,
    *,
    revision_id: str = "rev-prod-a",
    marker: str = "artifact-a",
    application_id: str = _APP,
    application_environment_id: str = _ENV,
    stores=None,
    revision_bound_strict_external_work: bool = True,
):
    from intergrax.applications._shared.production_agent_platform_runtime import (
        build_production_agent_platform_runtime,
    )

    runtime = build_production_agent_platform_runtime()
    stores = stores or runtime.stores
    from intergrax.agent_distribution.runtime_context_staging import (
        directory_content_digest,
    )

    artifact_root, _digest_before_marker, lock = _build_echo_artifact(
        tmp_path,
        marker=marker,
        revision_bound_strict_external_work=revision_bound_strict_external_work,
    )
    (artifact_root / "site-packages" / "bundle_marker.py").write_text(
        'MARKER = "qualification"\n',
        encoding="utf-8",
    )
    digest = directory_content_digest(artifact_root)
    if revision_bound_strict_external_work:
        manifest = _ac3_revision_bound_strict_external_work_host_manifest()
        roster = _roster_with_strict_external_work().model_copy(
            update={
                "application_id": application_id,
                "application_environment_id": application_environment_id,
            },
        ).with_revision_id()
    else:
        manifest = _manifest()
        roster = _roster().model_copy(
            update={
                "application_id": application_id,
                "application_environment_id": application_environment_id,
            },
        ).with_revision_id()
    revision = _revision(revision_id, roster=roster, artifact_digest=digest, lock=lock)
    locator = _seed_canonical_authority(
        stores,
        revision=revision,
        roster=roster,
        lock=lock,
        artifact_root=artifact_root,
        digest=digest,
        manifest=manifest,
    )
    authority = _authority(stores)
    bundle = build_production_registry_projection_input_bundle_for_revision(
        application_id=application_id,
        application_environment_id=application_environment_id,
        runtime_revision_id=revision_id,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        authority=authority,
    )
    return bundle, locator, artifact_root, digest, lock, stores, authority


def _ac3_environment() -> StrictProductionServingEnvironment:
    return StrictProductionServingEnvironment(
        application_id=_APP,
        application_environment_id=_ENV,
    )


def _open_durable_gca_host_stores(storage: StrictProductionProcessStorage):
    return open_owned_sqlite_governed_external_work_host_stores(
        storage.governed_external_work_host_db,
    )


def _composition_builder_for_storage(storage: StrictProductionProcessStorage):
    owned = _open_durable_gca_host_stores(storage)
    return owned, build_strict_durable_gca_composition_builder(owned)


def _composition_builder(tmp_path: Path):
    """Qualification helper: durable GCA composition builder for ``tmp_path`` storage."""
    storage, _adapter = _storage_and_adapter(tmp_path)
    _owned, builder = _composition_builder_for_storage(storage)
    return builder


def _lifecycle_gca_kwargs(storage: StrictProductionProcessStorage) -> dict[str, object]:
    owned, builder = _composition_builder_for_storage(storage)
    return {
        "composition_builder": builder,
        "owned_governed_external_work_host_store_bundle": owned,
    }


def _patch_gca_manifest_for_ac3_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )

    monkeypatch.setattr(
        "governed_contractor_application.host.main.build_governed_contractor_manifest",
        _ac3_revision_bound_strict_external_work_host_manifest,
    )
    monkeypatch.setattr(
        "governed_contractor_application.host.factory.build_governed_contractor_manifest",
        _ac3_revision_bound_strict_external_work_host_manifest,
    )

    def _ac3_aligned_environment_profile(
        _settings: GovernedContractorBackendSettings,
    ) -> ApplicationEnvironmentProfile:
        return ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)

    monkeypatch.setattr(
        "governed_contractor_application.host.main.build_governed_contractor_environment_profile",
        _ac3_aligned_environment_profile,
    )
    monkeypatch.setattr(
        "governed_contractor_application.host.factory.build_governed_contractor_environment_profile",
        _ac3_aligned_environment_profile,
    )


def _host_runtime_for_startup(settings: GovernedContractorBackendSettings):
    from applications.governed_contractor_application.tests.host.test_gr6_wire_production_decision_governance import (
        _test_policy_bundle,
    )
    from external_contractor_adapter.tests.fakes.deterministic_external_work import (
        DeterministicExternalWorkFake,
    )
    from governed_contractor_application.host.collaborative_work_local_fixture import (
        build_in_memory_collaborative_work_repositories,
    )
    from intergrax.runtime.task.active_task_registry import ActiveTaskRegistryTaskScopeResolver

    fake = DeterministicExternalWorkFake()
    cw = build_in_memory_collaborative_work_repositories()
    from tests.qualification.prod_q._prod_q_strict_http_caller_auth import (
        with_strict_http_api_keys,
    )
    from tests.unit.applications.test_production_registry_startup_ac3_final import _ENV

    from tests.qualification.prod_q._prod_q_strict_external_work_provider_qualification_config import (
        with_qualified_strict_external_work_provider_pin,
    )

    settings = with_qualified_strict_external_work_provider_pin(
        with_strict_http_api_keys(
            replace(
                settings,
                include_mcp=False,
                include_scheduler=False,
                include_interaction_routes=False,
                include_task_control=False,
                runtime_policy_bundle=_test_policy_bundle(),  # type: ignore[arg-type]
            ),
            execution_tenant_id=_ENV,
        ),
    )
    return compose_governed_contractor_host_runtime(
        settings,
        integration=fake,
        task_scope=ActiveTaskRegistryTaskScopeResolver(),
        collaborative_work_repositories=cw,
    )


def _open_with_activation(
    tmp_path: Path,
    *,
    environment: StrictProductionServingEnvironment,
    revision_id: str = "rev-d4-activate",
):
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )

    storage, adapter = _storage_and_adapter(tmp_path)
    governance = _governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        **_lifecycle_gca_kwargs(storage),
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        bundle, locator, *_rest = _qualified_canonical_bundle(
            tmp_path,
            revision_id=revision_id,
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        _activate_bundle(
            lifecycle.launcher,
            bundle,
            locator,
            principal=governance.principal,
        )

    return open_strict_production_serving_session(
        storage=storage,
        environment=environment,
        governance=governance,
        deployment_adapter=adapter,
        **_lifecycle_gca_kwargs(storage),
        missing_continuation_file_policy="empty",
    )


def test_d4_authorized_production_opener_builds_valid_lifecycle(tmp_path: Path) -> None:
    storage, adapter = _storage_and_adapter(tmp_path)
    env = build_governed_contractor_environment_profile(
        GovernedContractorBackendSettings.from_env(),
    )
    governance = _governance(env)
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
    ) as lifecycle:
        assert lifecycle.launcher is not None
        assert lifecycle.governance is governance


def test_d4_active_state_served_after_reconciliation(tmp_path: Path) -> None:
    session = _open_with_activation(tmp_path, environment=_ac3_environment())
    assert session.readiness.ready_for_traffic is True
    assert session.serving_runtime_handoff is not None
    session.close()


def _assert_gca_host_store_operations_fail_closed(
    lifecycle: ProductionProcessLifecycle,
) -> None:
    from governed_contractor_application.host.sqlite_governed_external_work_host_store_bundle import (
        GovernedExternalWorkHostPersistenceError,
    )

    bundle = lifecycle.composition.governed_external_work_host_store_bundle
    assert bundle is not None
    with pytest.raises(GovernedExternalWorkHostPersistenceError, match="closed"):
        bundle.execution_store.get_create_follow_on_record("closed-probe")


def test_d4_gca_owned_host_store_closed_after_close_process_resources(
    tmp_path: Path,
) -> None:
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )

    storage, adapter = _storage_and_adapter(tmp_path)
    environment = _ac3_environment()
    governance = _governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    gca_kwargs = _lifecycle_gca_kwargs(storage)
    lifecycle = open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
        **gca_kwargs,
    )
    lifecycle.close_process_resources()
    _assert_gca_host_store_operations_fail_closed(lifecycle)


def test_d4_gca_owned_host_store_close_failure_propagates_and_releases_slot(
    tmp_path: Path,
) -> None:
    from intergrax.applications._shared import production_process_lifecycle as ppl
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )

    storage, adapter = _storage_and_adapter(tmp_path)
    environment = _ac3_environment()
    governance = _governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    gca_kwargs = _lifecycle_gca_kwargs(storage)
    lifecycle = open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
        **gca_kwargs,
    )
    owned = lifecycle._owned_governed_external_work_host_store_bundle
    assert owned is not None
    with patch.object(owned, "close", side_effect=RuntimeError("gca_owned_close_fail")):
        with pytest.raises(RuntimeError, match="gca_owned_close_fail"):
            lifecycle.close_process_resources()
    assert ppl._active_production_process_lifecycle is None
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db.parent / "ad-retry.db",
        continuation_state_file=storage.continuation_state_file.parent / "continuation-retry.json",
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db.parent / "invocation-retry.db",
        **_lifecycle_gca_kwargs(storage),
    ) as retry_lifecycle:
        assert retry_lifecycle.composition.agent_platform_runtime is not None


def test_d4_strict_session_close_closes_owned_gca_host_store(tmp_path: Path) -> None:
    session = _open_with_activation(
        tmp_path,
        environment=_ac3_environment(),
        revision_id="rev-d4-gca-owned-close",
    )
    session.close()
    _assert_gca_host_store_operations_fail_closed(session.lifecycle)


def test_d4_session_close_drains_colocated_execution_admission(tmp_path: Path) -> None:
    session = _open_with_activation(
        tmp_path,
        environment=_ac3_environment(),
        revision_id="rev-d4-close-drain",
    )
    handoff = session.serving_runtime_handoff
    assert handoff is not None
    adapter = session.lifecycle.deployment_adapter
    assert isinstance(adapter, VenvBundleColocatedRuntimeDeploymentAdapter)
    unit_ref = handoff.identity.serving_unit_ref
    unit = adapter._units[unit_ref]
    assert unit.serving_admission_authorized is True
    assert unit.admission.admission_closed() is False
    session.close()
    assert unit.lifecycle_state is DeploymentInstanceState.DRAINING
    assert unit.admission.admission_closed() is True


def test_d4_explicit_activation_uses_d2_launcher(tmp_path: Path) -> None:
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )

    storage, adapter = _storage_and_adapter(tmp_path)
    environment = _ac3_environment()
    governance = _governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        **_lifecycle_gca_kwargs(storage),
    ) as lifecycle:
        bundle, locator, *_rest = _qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-d4-explicit",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
    intent = StrictProductionActivationIntent(
        projection_input=bundle,
        activation_request=_activation_request(bundle, locator),
        admission_mutation_id=reference_admission_mutation_id(
            bundle.runtime_revision.runtime_revision_id,
        ),
    )
    session = open_strict_production_serving_session(
        storage=storage,
        environment=environment,
        governance=governance,
        deployment_adapter=adapter,
        activation_intent=intent,
        **_lifecycle_gca_kwargs(storage),
        missing_continuation_file_policy="empty",
    )
    assert session.composition.serving_runtime_handoff is not None
    session.close()


def test_d4_gca_factory_consumes_validated_handoff(tmp_path: Path) -> None:
    from fastapi import FastAPI

    session = _open_with_activation(
        tmp_path,
        environment=_ac3_environment(),
        revision_id="rev-d4-gca-handoff",
    )
    handoff = session.serving_runtime_handoff
    with patch(
        "governed_contractor_application.host.strict_production_serving.create_governed_contractor_process_app",
    ) as create_app:
        create_app.return_value = FastAPI()
        create_governed_contractor_strict_production_asgi_app(session)
        composition = create_app.call_args.kwargs["process_composition"]
        assert composition.serving_runtime_handoff is handoff
    session.close()


def test_d4_asgi_startup_only_after_preconditions(tmp_path: Path) -> None:
    session = _open_with_activation(tmp_path, environment=_ac3_environment())
    session.readiness = StrictProductionServingReadiness(
        process_lifecycle_open=True,
        production_dependencies_validated=True,
        serving_state_reconciled=False,
        serving_handoff_bound=False,
        execution_admission_eligible=False,
        ready_for_traffic=False,
    )
    with pytest.raises(ValueError, match="not ready for traffic"):
        create_governed_contractor_strict_production_asgi_app(session)
    session.close()


def test_d4_asgi_shutdown_releases_owned_resources(tmp_path: Path) -> None:
    from fastapi import FastAPI

    session = _open_with_activation(tmp_path, environment=_ac3_environment())
    with patch(
        "governed_contractor_application.host.strict_production_serving.create_governed_contractor_process_app",
        return_value=FastAPI(),
    ):
        app = create_governed_contractor_strict_production_asgi_app(session)
    with TestClient(app):
        pass
    with pytest.raises(StrictProductionServingLauncherError, match="already closed"):
        session.close()


def test_d4_missing_governance(tmp_path: Path) -> None:
    storage, adapter = _storage_and_adapter(tmp_path)
    with pytest.raises(StrictProductionServingLauncherError, match="Governance"):
        open_strict_production_serving_session(
            storage=storage,
            environment=_ac3_environment(),
            governance=None,  # type: ignore[arg-type]
            deployment_adapter=adapter,
        )


def test_d4_governance_deny_blocks_activation(tmp_path: Path) -> None:
    from intergrax.agent_distribution.control_plane_governance import (
        StaticApplicationEnvironmentTenantResolver,
    )
    from intergrax.applications._shared.production_control_plane_governance_wiring import (
        ProductionControlPlaneGovernance,
    )
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )
    from intergrax.runtime.governance.control_plane_mutation_authorization import (
        ControlPlaneMutationAuthorizationBoundary,
    )
    from tests.unit.applications.test_prod_q_r1_r7_d2_production_process_lifecycle import (
        _DenyEvaluator,
        _governance as d2_governance,
    )

    storage, adapter = _storage_and_adapter(tmp_path)
    environment = _ac3_environment()
    env = ApplicationEnvironmentProfile(profile_id=environment.application_environment_id)
    base = d2_governance(env)
    governance = ProductionControlPlaneGovernance(
        principal=base.principal,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=_DenyEvaluator(),
        ),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            tenant_id=env.profile_id,
        ),
    )
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        **_lifecycle_gca_kwargs(storage),
    ) as lifecycle:
        bundle, locator, *_rest = _qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-d4-deny",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
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
            governance=governance,
            deployment_adapter=adapter,
            activation_intent=intent,
            **_lifecycle_gca_kwargs(storage),
            missing_continuation_file_policy="empty",
        )


def test_d4_missing_serving_pointer_after_open(tmp_path: Path) -> None:
    storage, adapter = _storage_and_adapter(tmp_path)
    env = build_governed_contractor_environment_profile(
        GovernedContractorBackendSettings.from_env(),
    )
    governance = _governance(env)
    with pytest.raises(StrictProductionServingLauncherError):
        open_strict_production_serving_session(
            storage=storage,
            environment=_ac3_environment(),
            governance=governance,
            deployment_adapter=adapter,
            **_lifecycle_gca_kwargs(storage),
            missing_continuation_file_policy="empty",
        )


def test_d4_fake_adapter_rejected(tmp_path: Path) -> None:
    storage, adapter = _storage_and_adapter(tmp_path)
    env = build_governed_contractor_environment_profile(
        GovernedContractorBackendSettings.from_env(),
    )
    governance = _governance(env)
    with pytest.raises(
        StrictProductionServingLauncherError,
        match="FakeInMemoryRuntimeDeploymentAdapter",
    ):
        open_strict_production_serving_session(
            storage=storage,
            environment=_ac3_environment(),
            governance=governance,
            deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
        )


def test_d4_duplicate_process_lifecycle(tmp_path: Path) -> None:
    session = _open_with_activation(tmp_path, environment=_ac3_environment())
    storage, adapter = _storage_and_adapter(tmp_path)
    env = build_governed_contractor_environment_profile(
        GovernedContractorBackendSettings.from_env(),
    )
    governance = _governance(env)
    with pytest.raises(ProductionProcessLifecycleError, match="duplicate"):
        open_strict_production_serving_session(
            storage=storage,
            environment=_ac3_environment(),
            governance=governance,
            deployment_adapter=adapter,
            **_lifecycle_gca_kwargs(storage),
            missing_continuation_file_policy="empty",
        )
    session.close()


def test_d4_stale_physical_unit_after_serving(tmp_path: Path) -> None:
    storage, adapter = _storage_and_adapter(tmp_path)
    environment = _ac3_environment()
    env_profile = build_governed_contractor_environment_profile(
        GovernedContractorBackendSettings.from_env(),
    )
    governance = _governance(
        env_profile.model_copy(update={"profile_id": environment.application_environment_id}),
    )
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        **_lifecycle_gca_kwargs(storage),
    ) as lifecycle:
        bundle, locator, *_rest = _qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-d4-stale",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        _activate_bundle(
            lifecycle.launcher,
            bundle,
            locator,
            principal=governance.principal,
        )
    fresh_adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        allow_qualification_locators=True,
    )
    with pytest.raises(StrictProductionServingLauncherError):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=fresh_adapter,
            **_lifecycle_gca_kwargs(storage),
            missing_continuation_file_policy="empty",
        )


def test_d4_03_legacy_main_run_fail_closed(capsys: pytest.CaptureFixture[str]) -> None:
    from governed_contractor_application.host.main import run

    with patch(
        "intergrax.applications._shared.production_process_lifecycle.open_production_process_lifecycle",
    ) as open_lifecycle:
        with pytest.raises(SystemExit) as exc_info:
            run()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "open_strict_production_serving_session" in err
    assert "create_governed_contractor_strict_production_asgi_app" in err
    assert "Do not use main.run() for deployment." in err
    open_lifecycle.assert_not_called()


def test_d4_21_activation_intent_scope_mismatch_before_lifecycle(
    tmp_path: Path,
) -> None:
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )

    storage, adapter = _storage_and_adapter(tmp_path)
    bundle_env = _ac3_environment()
    mismatched_serving_env = _gca_environment()
    governance = _governance(
        ApplicationEnvironmentProfile(profile_id=bundle_env.application_environment_id),
    )
    bundle, locator, *_rest = _qualified_canonical_bundle(
        tmp_path,
        revision_id="rev-d4-21-scope-mismatch",
        application_id=bundle_env.application_id,
        application_environment_id=bundle_env.application_environment_id,
    )
    intent = StrictProductionActivationIntent(
        projection_input=bundle,
        activation_request=_activation_request(bundle, locator),
        admission_mutation_id=reference_admission_mutation_id(
            bundle.runtime_revision.runtime_revision_id,
        ),
    )
    with patch(
        "intergrax.applications._shared.strict_production_serving_launcher.open_production_process_lifecycle",
    ) as open_lifecycle:
        with pytest.raises(
            StrictProductionServingLauncherError,
            match="activation intent revision scope does not match serving environment",
        ):
            open_strict_production_serving_session(
                storage=storage,
                environment=mismatched_serving_env,
                governance=governance,
                deployment_adapter=adapter,
                activation_intent=intent,
                **_lifecycle_gca_kwargs(storage),
                missing_continuation_file_policy="empty",
            )
    open_lifecycle.assert_not_called()


def _open_session_missing_serving_prerequisites_kwargs(tmp_path: Path):
    storage, adapter = _storage_and_adapter(tmp_path)
    env = build_governed_contractor_environment_profile(
        GovernedContractorBackendSettings.from_env(),
    )
    governance = _governance(env)
    return dict(
        storage=storage,
        environment=_ac3_environment(),
        governance=governance,
        deployment_adapter=adapter,
        **_lifecycle_gca_kwargs(storage),
        missing_continuation_file_policy="empty",
    )


def test_d4_r2_startup_failure_cleanup_success_preserves_primary_and_releases_slot(
    tmp_path: Path,
) -> None:
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )

    kwargs = _open_session_missing_serving_prerequisites_kwargs(tmp_path)
    close_calls = 0
    original_close = ProductionProcessLifecycle.close_process_resources

    def _tracked_close(self: ProductionProcessLifecycle) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(self)

    with patch.object(
        ProductionProcessLifecycle,
        "close_process_resources",
        _tracked_close,
    ):
        with pytest.raises(StrictProductionServingLauncherError):
            open_strict_production_serving_session(**kwargs)
    assert close_calls == 1
    storage = kwargs["storage"]
    adapter = kwargs["deployment_adapter"]
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db.parent / "ad-retry.db",
        continuation_state_file=storage.continuation_state_file.parent / "continuation-retry.json",
        deployment_adapter=adapter,
        governance=_governance(
            ApplicationEnvironmentProfile(profile_id=kwargs["environment"].application_environment_id),
        ),
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db.parent / "invocation-retry.db",
        composition_builder=kwargs["composition_builder"],
    ) as lifecycle:
        assert lifecycle.composition.agent_platform_runtime is not None


def test_d4_r2_session_close_drain_failure_still_closes_process_resources(
    tmp_path: Path,
) -> None:
    from intergrax.applications._shared.colocated_serving_runtime_composition import (
        ColocatedServingAdmissionTopologyError,
    )

    session = _open_with_activation(
        tmp_path,
        environment=_ac3_environment(),
        revision_id="rev-d4-close-drain-fail",
    )
    drain_fail = ColocatedServingAdmissionTopologyError("d4_close_drain_fail")
    close_calls = 0
    original_close = ProductionProcessLifecycle.close_process_resources

    def _tracked_close(self: ProductionProcessLifecycle) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(self)

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
    ):
        with pytest.raises(StrictProductionServingLauncherError, match="d4_close_drain_fail"):
            session.close()
    assert close_calls == 1
    assert session._closed is False


def test_d4_r2_session_close_drain_and_cleanup_failure_preserves_both_errors(
    tmp_path: Path,
) -> None:
    from intergrax.applications._shared.colocated_serving_runtime_composition import (
        ColocatedServingAdmissionTopologyError,
    )

    session = _open_with_activation(
        tmp_path,
        environment=_ac3_environment(),
        revision_id="rev-d4-close-both-fail",
    )
    drain_fail = ColocatedServingAdmissionTopologyError("d4_close_drain_fail")
    cleanup_fail = RuntimeError("d4_close_cleanup_fail")
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
    ):
        with pytest.raises(ExceptionGroup) as exc_info:
            session.close()
    errors = exc_info.value.exceptions
    assert len(errors) == 2
    assert isinstance(errors[0], StrictProductionServingLauncherError)
    assert errors[1] is cleanup_fail
    assert session._closed is False


def test_d4_r2_startup_failure_cleanup_failure_preserves_both_errors(
    tmp_path: Path,
) -> None:
    kwargs = _open_session_missing_serving_prerequisites_kwargs(tmp_path)
    cleanup_fail = RuntimeError("d4_r2_cleanup_fail")
    with patch.object(
        ProductionProcessLifecycle,
        "close_process_resources",
        side_effect=cleanup_fail,
    ) as close_mock:
        with pytest.raises(ExceptionGroup) as exc_info:
            open_strict_production_serving_session(**kwargs)
    close_mock.assert_called_once()
    errors = exc_info.value.exceptions
    assert len(errors) == 2
    assert isinstance(errors[0], StrictProductionServingLauncherError)
    assert errors[1] is cleanup_fail
