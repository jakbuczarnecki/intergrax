# © Artur Czarnecki. All rights reserved.

"""R7-Q §C Package B — operator production profiles (DIM6 + DIM17).

Lab/regression gates exercise canonical strict-production boundaries with explicit
classification. Operator-approved production mutation policy and production
credential profiles are out-of-repo; see qualification evidence record.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from unittest.mock import patch

import pytest

from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
    authorize_scoped_control_plane_mutation,
    build_admit_runtime_revision_mutation_request,
)
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
    VenvBundleDeploymentArtifactError,
    resolve_venv_bundle_artifact_directory,
)
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_reference_production_lifecycle_policy_bundle,
)
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.production_lifecycle import (
    ProductionLifecycleConfigurationError,
    wire_production_lifecycle_services,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.production_process_lifecycle import (
    open_production_process_lifecycle,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionActivationIntent,
    StrictProductionServingLauncherError,
    open_strict_production_serving_session,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.durable_agent_platform_runtime import (
    build_durable_production_agent_platform_runtime,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    assert_identity_continuity,
    load_d4_launcher_module,
    repo_root,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_venv_bundle_colocated_deployment_adapter import (
    _bundle,
    _revision_for,
)
from tests.unit.applications.test_prod_q_r1_r7_d2_production_process_lifecycle import (
    _DenyEvaluator,
    _governance as d2_governance,
)
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _activation_request,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "fd10a76eb0a35c08441ca5e3fab5f48e393695ea"
_REFERENCE_LAB_POLICY_BUNDLE_ID: Final = "reference_production.control_plane"
_OPERATOR_MUTATION_POLICY_PROFILE_ENV: Final = (
    "INTERGRAX_OPERATOR_PRODUCTION_MUTATION_POLICY_BUNDLE_ID"
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


def test_c_pkg_b_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def test_c_pkg_b_a01_in_tree_policy_is_reference_lab_only() -> None:
    bundle = build_reference_production_lifecycle_policy_bundle()
    assert bundle.bundle_id == _REFERENCE_LAB_POLICY_BUNDLE_ID
    operator_profile = os.environ.get(_OPERATOR_MUTATION_POLICY_PROFILE_ENV, "").strip()
    assert not operator_profile, (
        "ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED: operator mutation policy profile "
        f"must be supplied via sanctioned out-of-repo mechanism (env "
        f"{_OPERATOR_MUTATION_POLICY_PROFILE_ENV} is set but has no in-repo loader)."
    )


@dataclass(slots=True)
class _GovernanceDenyEvidence:
    evaluator: _DenyEvaluator = field(default_factory=_DenyEvaluator)

    def build(self, environment) -> ProductionControlPlaneGovernance:
        env = ApplicationEnvironmentProfile(profile_id=environment.application_environment_id)
        base = d2_governance(env)
        return ProductionControlPlaneGovernance(
            principal=base.principal,
            mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
                evaluator=self.evaluator,
            ),
            environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
                tenant_id=env.profile_id,
            ),
        )


def test_c_pkg_b_b6_01_lab_authorized_strict_activation_after_governance(
    tmp_path: Path,
) -> None:
    """LAB_REFERENCE profile only — not operator production qualification."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    revision_id = "rev-c-pkg-b-b6-01-lab"
    session = d4._open_with_activation(
        tmp_path,
        environment=environment,
        revision_id=revision_id,
    )
    try:
        assert_identity_continuity(session, revision_id=revision_id)
        assert session.readiness.ready_for_traffic is True
        boundary = session.lifecycle.launcher._mutation_authorization_boundary  # noqa: SLF001
        assert boundary is not None
        governance = d4._governance(
            ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
        )
        request = build_admit_runtime_revision_mutation_request(
            principal=governance.principal,
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            mutation_id="mut-pkg-b-b6-01",
            runtime_revision_id=revision_id,
            identity_digest="digest",
        )
        permitted = authorize_scoped_control_plane_mutation(
            boundary=boundary,
            tenant_resolver=governance.environment_tenant_resolver,
            request=request,
        )
        assert permitted.permitted
    finally:
        session.close()


def test_c_pkg_b_b6_02_unauthorized_mutation_denied_before_physical_prepare(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    storage, adapter = d4._storage_and_adapter(tmp_path)
    deny_evidence = _GovernanceDenyEvidence()
    governance = deny_evidence.build(environment)
    composition_builder = d4._composition_builder(tmp_path)
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
    ) as lifecycle:
        bundle, locator, *_ = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-c-pkg-b-b6-02-deny",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        stores = lifecycle.composition.agent_platform_runtime.stores
    intent = StrictProductionActivationIntent(
        projection_input=bundle,
        activation_request=_activation_request(bundle, locator),
        admission_mutation_id=reference_admission_mutation_id(
            bundle.runtime_revision.runtime_revision_id,
        ),
    )
    with patch.object(adapter, "prepare", wraps=adapter.prepare) as physical_prepare:
        with pytest.raises(StrictProductionServingLauncherError) as exc_info:
            open_strict_production_serving_session(
                storage=storage,
                environment=environment,
                governance=governance,
                deployment_adapter=adapter,
                activation_intent=intent,
                composition_builder=composition_builder,
                missing_continuation_file_policy="empty",
            )
    assert physical_prepare.call_count == 0
    assert len(deny_evidence.evaluator.calls) >= 1
    assert exc_info.value.__cause__ is not None or str(exc_info.value)
    serving = stores.serving_store.get_serving_record(
        environment.application_id,
        environment.application_environment_id,
    )
    assert serving is None


def test_c_pkg_b_b6_03_missing_governance_and_deny_policy_fail_closed(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    storage, adapter = d4._storage_and_adapter(tmp_path)
    environment = d4._ac3_environment()
    with pytest.raises(StrictProductionServingLauncherError, match="Governance"):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=None,  # type: ignore[arg-type]
            deployment_adapter=adapter,
        )
    env = ApplicationEnvironmentProfile(profile_id=environment.application_environment_id)
    deny_governance = _GovernanceDenyEvidence().build(environment)
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=deny_governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=d4._composition_builder(tmp_path),
    ) as lifecycle:
        bundle, locator, *_ = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-c-pkg-b-b6-03-deny",
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
            governance=deny_governance,
            deployment_adapter=adapter,
            activation_intent=intent,
            composition_builder=d4._composition_builder(tmp_path),
            missing_continuation_file_policy="empty",
        )
    request = build_admit_runtime_revision_mutation_request(
        principal=deny_governance.principal,
        application_id=environment.application_id,
        application_environment_id=environment.application_environment_id,
        mutation_id="mut-deny-evidence",
        runtime_revision_id=bundle.runtime_revision.runtime_revision_id,
        identity_digest="digest",
    )
    scoped = authorize_scoped_control_plane_mutation(
        boundary=deny_governance.mutation_authorization_boundary,
        tenant_resolver=deny_governance.environment_tenant_resolver,
        request=request,
    )
    assert not scoped.permitted
    assert scoped.decision.reason == "explicit_deny"


def test_c_pkg_b_b6_04_tenant_resolver_blocks_cross_environment_mutation(
    tmp_path: Path,
) -> None:
    env = ApplicationEnvironmentProfile(profile_id="pkg-b-env-a")
    adapter_roots = (tmp_path / "artifacts",)
    adapter_roots[0].mkdir(parents=True, exist_ok=True)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=adapter_roots,
        allow_qualification_locators=True,
    )
    governance = ProductionControlPlaneGovernance(
        principal=d2_governance(env).principal,
        mutation_authorization_boundary=d2_governance(env).mutation_authorization_boundary,
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            tenant_id="pkg-b-env-b",
        ),
    )
    with open_production_process_lifecycle(
        agent_distribution_db=tmp_path / "ad.db",
        continuation_state_file=tmp_path / "continuation.json",
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=adapter_roots,
        missing_file_policy="empty",
    ):
        request = build_admit_runtime_revision_mutation_request(
            principal=governance.principal,
            application_id="app",
            application_environment_id=env.profile_id,
            mutation_id="mut-tenant",
            runtime_revision_id="rev-1",
            identity_digest="digest",
        )
        result = authorize_scoped_control_plane_mutation(
            boundary=governance.mutation_authorization_boundary,
            tenant_resolver=governance.environment_tenant_resolver,
            request=request,
        )
        assert not result.permitted


def test_c_pkg_b_b17_01_unsupported_topology_fail_closed(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
        allow_qualification_locators=True,
    )
    revision = _revision_for(
        "rev-pkg-b-oci",
        digest=digest,
        lock=lock,
        topology=MaterializationTopology.OCI_IMAGE,
    )
    with pytest.raises(VenvBundleColocatedDeploymentError, match="VENV_BUNDLE"):
        adapter.prepare(revision, artifact_locator=locator)


def test_c_pkg_b_b17_02_prod_backend_settings_reject_missing_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOVERNED_CONTRACTOR_BACKEND_ENV", "prod")
    monkeypatch.delenv("GOVERNED_CONTRACTOR_BACKEND_BOOTSTRAP_API_KEY", raising=False)
    monkeypatch.delenv("GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("GOVERNED_CONTRACTOR_BACKEND_ALLOW_UNAUTHENTICATED", raising=False)
    with pytest.raises(ValueError, match="requires API keys"):
        GovernedContractorBackendSettings.from_env()


def test_c_pkg_b_b17_03_invalid_production_lifecycle_configuration_fail_closed(
    tmp_path: Path,
) -> None:
    durable = build_durable_production_agent_platform_runtime(tmp_path / "ad.db")
    composition = ProductionProcessComposition(
        agent_platform_runtime=durable.agent_platform_runtime,
    )
    with pytest.raises(ProductionLifecycleConfigurationError, match="FakeInMemory"):
        wire_production_lifecycle_services(
            composition,
            durable_store_bundle=durable.distribution_store_bundle,
            deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
            trusted_artifact_parent_roots=(tmp_path,),
        )


def test_c_pkg_b_b17_04_trusted_artifact_roots_and_qualification_locator_gate(
    tmp_path: Path,
) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path / "allowed"],
        allow_qualification_locators=False,
    )
    revision = _revision_for("rev-pkg-b-trust", digest=digest, lock=lock)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="trusted"):
        adapter.prepare(revision, artifact_locator=locator)
    with pytest.raises(VenvBundleDeploymentArtifactError, match="test://"):
        resolve_venv_bundle_artifact_directory(
            f"test://{root.as_posix()}",
            allow_qualification_locators=False,
        )


def test_c_pkg_b_b17_05_deployment_adapter_replaceable_via_typed_wiring(
    tmp_path: Path,
) -> None:
    from tests.unit.applications.test_prod_q_r1_r7_d2_production_process_lifecycle import (
        _AltRuntimeDeploymentAdapter,
    )

    durable = build_durable_production_agent_platform_runtime(tmp_path / "ad.db")
    composition = ProductionProcessComposition(
        agent_platform_runtime=durable.agent_platform_runtime,
    )
    alt = _AltRuntimeDeploymentAdapter()
    services = wire_production_lifecycle_services(
        composition,
        durable_store_bundle=durable.distribution_store_bundle,
        deployment_adapter=alt,  # type: ignore[arg-type]
        trusted_artifact_parent_roots=(tmp_path / "artifacts",),
    )
    assert services.activation_service._deployment_adapter is alt  # noqa: SLF001
