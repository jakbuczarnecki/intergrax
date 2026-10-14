# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D2 — sanctioned production process lifecycle composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.agent_distribution.deployment import (
    FakeInMemoryRuntimeDeploymentAdapter,
    RuntimeDeploymentAdapter,
)
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_reference_production_control_plane_mutation_boundary,
)
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.production_lifecycle import (
    ProductionLifecycleConfigurationError,
    wire_production_lifecycle_services,
)
from intergrax.applications._shared.production_process_lifecycle import (
    ProductionProcessLifecycleError,
    open_production_process_lifecycle,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.durable_agent_platform_runtime import (
    build_durable_production_agent_platform_runtime,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationPolicyEvaluator,
    ControlPlaneMutationRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_STRICT_PRODUCTION_MODULE = (
    _REPO / "intergrax" / "applications" / "_shared" / "production_process_lifecycle.py"
)


def _env() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile(profile_id="tenant-a")


def _governance(env: ApplicationEnvironmentProfile) -> ProductionControlPlaneGovernance:
    tenant = env.profile_id
    principal = RequestIdentity(
        tenant_id=tenant,
        user_id="production-launcher",
        principal_type=PrincipalType.SERVICE,
        auth_subject="production-launcher",
    )
    return ProductionControlPlaneGovernance(
        principal=principal,
        mutation_authorization_boundary=build_reference_production_control_plane_mutation_boundary(
            env,
        ),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            tenant_id=tenant,
        ),
    )


def _venv_adapter(tmp_path: Path) -> VenvBundleColocatedRuntimeDeploymentAdapter:
    roots = (tmp_path / "artifacts",)
    roots[0].mkdir(parents=True, exist_ok=True)
    return VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=roots,
        allow_qualification_locators=True,
    )


def test_d2_production_process_lifecycle_module_has_no_fake_adapter_reachability() -> None:
    text = _STRICT_PRODUCTION_MODULE.read_text(encoding="utf-8")
    assert "FakeInMemoryRuntimeDeploymentAdapter" not in text


def test_d2_open_production_process_lifecycle_wires_durable_stores_and_adapter(
    tmp_path: Path,
) -> None:
    env = _env()
    adapter = _venv_adapter(tmp_path)
    roots = (tmp_path / "artifacts",)
    with open_production_process_lifecycle(
        agent_distribution_db=tmp_path / "ad.db",
        continuation_state_file=tmp_path / "continuation.json",
        deployment_adapter=adapter,
        governance=_governance(env),
        trusted_artifact_parent_roots=roots,
        provider_invocation_db=tmp_path / "invocation.db",
        missing_file_policy="empty",
    ) as lifecycle:
        services = lifecycle.lifecycle_services
        bundle_db = lifecycle.composition.agent_platform_runtime.stores
        assert (
            services.activation_service._deployment_adapter  # noqa: SLF001
            is adapter
        )
        assert services.deployment_instance_store is services.activation_service._deployment_instance_store  # noqa: SLF001
        assert (
            bundle_db.serving_store
            is services.activation_service._serving_store  # noqa: SLF001
        )
        assert lifecycle.launcher.services is services


def test_d2_rejects_fake_adapter(tmp_path: Path) -> None:
    env = _env()
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


def test_d2_rejects_missing_governance(tmp_path: Path) -> None:
    env = _env()
    adapter = _venv_adapter(tmp_path)
    with pytest.raises(ProductionLifecycleConfigurationError, match="ProductionControlPlaneGovernance"):
        open_production_process_lifecycle(
            agent_distribution_db=tmp_path / "ad.db",
            continuation_state_file=tmp_path / "continuation.json",
            deployment_adapter=adapter,
            governance=None,  # type: ignore[arg-type]
            trusted_artifact_parent_roots=(tmp_path / "artifacts",),
            missing_file_policy="empty",
        )


@dataclass
class _DenyEvaluator(ControlPlaneMutationPolicyEvaluator):
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="explicit_deny",
            policy_rule_id="test.deny",
        )


def test_d2_governance_deny_is_fail_closed(tmp_path: Path) -> None:
    env = _env()
    adapter = _venv_adapter(tmp_path)
    deny_governance = ProductionControlPlaneGovernance(
        principal=_governance(env).principal,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=_DenyEvaluator(),
        ),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            tenant_id=env.profile_id,
        ),
    )
    with open_production_process_lifecycle(
        agent_distribution_db=tmp_path / "ad.db",
        continuation_state_file=tmp_path / "continuation.json",
        deployment_adapter=adapter,
        governance=deny_governance,
        trusted_artifact_parent_roots=(tmp_path / "artifacts",),
        missing_file_policy="empty",
    ) as lifecycle:
        from intergrax.agent_distribution.control_plane_governance import (
            authorize_scoped_control_plane_mutation,
            build_admit_runtime_revision_mutation_request,
        )

        request = build_admit_runtime_revision_mutation_request(
            principal=deny_governance.principal,
            application_id="app",
            application_environment_id="env",
            mutation_id="mut-1",
            runtime_revision_id="rev-1",
            identity_digest="digest",
        )
        result = authorize_scoped_control_plane_mutation(
            boundary=deny_governance.mutation_authorization_boundary,
            tenant_resolver=deny_governance.environment_tenant_resolver,
            request=request,
        )
        assert not result.permitted
        assert lifecycle.launcher._mutation_authorization_boundary is deny_governance.mutation_authorization_boundary  # noqa: SLF001


def test_d2_wrong_tenant_resolver_blocks(tmp_path: Path) -> None:
    env = _env()
    adapter = _venv_adapter(tmp_path)
    governance = ProductionControlPlaneGovernance(
        principal=_governance(env).principal,
        mutation_authorization_boundary=build_reference_production_control_plane_mutation_boundary(
            env,
        ),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            tenant_id="other-tenant",
        ),
    )
    with open_production_process_lifecycle(
        agent_distribution_db=tmp_path / "ad.db",
        continuation_state_file=tmp_path / "continuation.json",
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=(tmp_path / "artifacts",),
        missing_file_policy="empty",
    ):
        from intergrax.agent_distribution.control_plane_governance import (
            authorize_scoped_control_plane_mutation,
            build_admit_runtime_revision_mutation_request,
        )

        request = build_admit_runtime_revision_mutation_request(
            principal=governance.principal,
            application_id="app",
            application_environment_id="env",
            mutation_id="mut-1",
            runtime_revision_id="rev-1",
            identity_digest="digest",
        )
        result = authorize_scoped_control_plane_mutation(
            boundary=governance.mutation_authorization_boundary,
            tenant_resolver=governance.environment_tenant_resolver,
            request=request,
        )
        assert not result.permitted


def test_d2_mismatched_store_bundle_rejected(tmp_path: Path) -> None:
    durable_a = build_durable_production_agent_platform_runtime(tmp_path / "a.db")
    durable_b = build_durable_production_agent_platform_runtime(tmp_path / "b.db")
    composition = ProductionProcessComposition(
        agent_platform_runtime=durable_a.agent_platform_runtime,
    )
    adapter = _venv_adapter(tmp_path)
    with pytest.raises(ProductionLifecycleConfigurationError, match="revision_store"):
        wire_production_lifecycle_services(
            composition,
            durable_store_bundle=durable_b.distribution_store_bundle,
            deployment_adapter=adapter,
            trusted_artifact_parent_roots=(tmp_path / "artifacts",),
        )


class _AltRuntimeDeploymentAdapter:
    """Structural stand-in proving replaceability through the same composition boundary."""

    def prepare_deployment(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError

    def check_readiness(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError

    def attest_serving_unit(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError

    def begin_drain(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError

    def await_drain(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError

    def stop_serving_unit(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError


def test_d2_typed_replaceable_adapter_wiring(tmp_path: Path) -> None:
    durable = build_durable_production_agent_platform_runtime(tmp_path / "ad.db")
    composition = ProductionProcessComposition(
        agent_platform_runtime=durable.agent_platform_runtime,
    )
    alt: RuntimeDeploymentAdapter = _AltRuntimeDeploymentAdapter()  # type: ignore[assignment]
    services = wire_production_lifecycle_services(
        composition,
        durable_store_bundle=durable.distribution_store_bundle,
        deployment_adapter=alt,
        trusted_artifact_parent_roots=(tmp_path / "artifacts",),
    )
    assert services.activation_service._deployment_adapter is alt  # noqa: SLF001


def test_d2_duplicate_process_lifecycle_rejected(tmp_path: Path) -> None:
    env = _env()
    adapter = _venv_adapter(tmp_path)
    first = open_production_process_lifecycle(
        agent_distribution_db=tmp_path / "ad.db",
        continuation_state_file=tmp_path / "continuation.json",
        deployment_adapter=adapter,
        governance=_governance(env),
        trusted_artifact_parent_roots=(tmp_path / "artifacts",),
        missing_file_policy="empty",
    )
    with pytest.raises(ProductionProcessLifecycleError, match="duplicate"):
        open_production_process_lifecycle(
            agent_distribution_db=tmp_path / "ad2.db",
            continuation_state_file=tmp_path / "continuation2.json",
            deployment_adapter=adapter,
            governance=_governance(env),
            trusted_artifact_parent_roots=(tmp_path / "artifacts",),
            missing_file_policy="empty",
        )
    first.close_process_resources()


def test_d2_close_is_idempotent_reject(tmp_path: Path) -> None:
    env = _env()
    adapter = _venv_adapter(tmp_path)
    lifecycle = open_production_process_lifecycle(
        agent_distribution_db=tmp_path / "ad.db",
        continuation_state_file=tmp_path / "continuation.json",
        deployment_adapter=adapter,
        governance=_governance(env),
        trusted_artifact_parent_roots=(tmp_path / "artifacts",),
        missing_file_policy="empty",
    )
    lifecycle.close_process_resources()
    with pytest.raises(ProductionProcessLifecycleError, match="already closed"):
        lifecycle.close_process_resources()
