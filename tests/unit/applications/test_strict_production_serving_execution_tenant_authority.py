# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-P3-A1 — strict serving HTTP execution tenant authority."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.strict_production_serving_execution_tenant_authority import (
    admit_strict_production_serving_root_governance_identity,
    assert_task_tenant_matches_strict_production_serving_authority,
    resolve_strict_production_serving_execution_tenant_binding,
)
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_reference_production_control_plane_mutation_boundary,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.applications._shared.serving_runtime_handoff import (
    ServingRuntimeHandoff,
    ServingUnitIdentity,
)
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _governance(environment_tenant: str) -> ProductionControlPlaneGovernance:
    env = ApplicationEnvironmentProfile(profile_id="env-prod")
    principal = RequestIdentity(
        tenant_id=environment_tenant,
        user_id="svc-principal",
        principal_type=PrincipalType.SERVICE,
        auth_subject="svc-principal",
    )
    return ProductionControlPlaneGovernance(
        principal=principal,
        mutation_authorization_boundary=build_reference_production_control_plane_mutation_boundary(
            env,
        ),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            tenant_id=environment_tenant,
        ),
    )


def _handoff() -> ServingRuntimeHandoff:
    identity = ServingUnitIdentity(
        application_id="app_a",
        application_environment_id="env-prod",
        runtime_revision_id="rev-1",
        serving_unit_ref="unit-1",
        materialization_artifact_digest="digest",
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
    )
    return ServingRuntimeHandoff(
        identity=identity,
        runtime_agent_factory_resolver=object(),  # type: ignore[arg-type]
    )


def test_resolve_binding_uses_application_environment_id_as_execution_tenant() -> None:
    handoff = _handoff()
    governance = _governance("tenant-b")
    binding = resolve_strict_production_serving_execution_tenant_binding(
        handoff=handoff,
        governance=governance,
    )
    assert binding.authoritative_execution_tenant_id == handoff.identity.application_environment_id
    assert binding.control_plane_environment_tenant_id == "tenant-b"


def test_mismatch_request_tenant_rejected() -> None:
    handoff = _handoff()
    governance = _governance("tenant-b")
    binding = resolve_strict_production_serving_execution_tenant_binding(
        handoff=handoff,
        governance=governance,
    )
    task = Task(
        tenant_id="forged-tenant",
        user_id="operator",
        session_id="s",
        agent_id="a",
        message="m",
    )
    with pytest.raises(ValueError, match="tenant authority mismatch"):
        assert_task_tenant_matches_strict_production_serving_authority(task, binding=binding)


def test_admit_uses_governance_principal_not_forged_user() -> None:
    handoff = _handoff()
    governance = _governance("tenant-b")
    binding = resolve_strict_production_serving_execution_tenant_binding(
        handoff=handoff,
        governance=governance,
    )
    task = Task(
        tenant_id=binding.authoritative_execution_tenant_id,
        user_id="forged-user",
        session_id="s",
        agent_id="a",
        message="m",
    )
    admitted = admit_strict_production_serving_root_governance_identity(
        task,
        binding=binding,
        principal=governance.principal,
    )
    assert admitted.tenant_id == binding.authoritative_execution_tenant_id
    assert admitted.principal_id == "svc-principal"
