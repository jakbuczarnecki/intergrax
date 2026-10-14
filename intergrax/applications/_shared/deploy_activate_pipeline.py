# © Artur Czarnecki. All rights reserved.

"""Shared deploy → project → activate orchestration for production and reference launchers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.agent_distribution.control_plane_governance import (
    ApplicationEnvironmentTenantResolver,
    authorize_scoped_control_plane_mutation,
    build_activation_mutation_request,
    build_admit_runtime_revision_mutation_request,
    runtime_revision_admission_identity_digest,
)
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationScope,
)
from intergrax.applications._shared.registry_projection_descriptor import (
    build_runtime_registry_projection_descriptor,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationAuthorizationResult
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

if TYPE_CHECKING:
    from intergrax.applications._shared.reference_production_lifecycle import (
        ReferenceProductionLifecycleServices,
    )


@dataclass(frozen=True, slots=True)
class ReferenceProductionLifecycleResult:
    """Evidence after successful deploy/project/activate on one environment."""

    runtime_revision_id: str
    application_id: str
    application_environment_id: str
    serving_pointer_revision: int
    resolved_projection: MaterializedRegistryProjection


class ReferenceProductionLifecycleError(ValueError):
    """Lifecycle launcher input or precondition violation."""


class ReferenceProductionLifecycleGovernanceBlockedError(ReferenceProductionLifecycleError):
    """Control-plane governance blocked deploy/activate."""

    def __init__(
        self,
        message: str,
        *,
        policy_action: str,
        authorization_evidence: ControlPlaneMutationAuthorizationEvidence | None = None,
        authorization_scope: ControlPlaneMutationAuthorizationScope | None = None,
    ) -> None:
        super().__init__(message)
        self.policy_action = policy_action
        self.authorization_evidence = authorization_evidence
        self.authorization_scope = authorization_scope


@dataclass(frozen=True, slots=True)
class DeployActivatePipelineContext:
    """Inputs required to run one deploy/activate cycle without reference-only defaults."""

    composition: ProductionProcessComposition
    services: ReferenceProductionLifecycleServices
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary
    environment_tenant_resolver: ApplicationEnvironmentTenantResolver


def run_deploy_and_activate(
    ctx: DeployActivatePipelineContext,
    projection_input: RegistryProjectionInputBundle,
    activation_request: ActivateRuntimeRevisionRequest,
    *,
    principal: RequestIdentity,
    admission_mutation_id: str,
) -> ReferenceProductionLifecycleResult:
    """Prepare registry projection and commit activation for one explicit revision."""
    revision = projection_input.runtime_revision
    application_id = revision.application_id
    application_environment_id = revision.application_environment_id
    runtime_revision_id = revision.runtime_revision_id
    normalized_admission_mutation_id = admission_mutation_id.strip()
    if not normalized_admission_mutation_id:
        raise ReferenceProductionLifecycleError(
            "production admission requires explicit admission_mutation_id",
        )

    if activation_request.runtime_revision_id != runtime_revision_id:
        raise ReferenceProductionLifecycleError(
            "activation request runtime_revision_id does not match projection input",
        )
    artifact_digest = projection_input.materialization_artifact_digest
    if artifact_digest is None:
        raise ReferenceProductionLifecycleError(
            "projection input requires materialization_artifact_digest",
        )
    if activation_request.expected_artifact_digest != artifact_digest:
        raise ReferenceProductionLifecycleError(
            "activation expected_artifact_digest does not match projection input",
        )
    if revision.materialization_artifact_digest is not None:
        if revision.materialization_artifact_digest != artifact_digest:
            raise ReferenceProductionLifecycleError(
                "runtime revision materialization_artifact_digest mismatch with projection input",
            )
    if revision.materialized_runtime_lock_digest is None:
        raise ReferenceProductionLifecycleError(
            "runtime revision requires materialized_runtime_lock_digest for admission",
        )
    if revision.runtime_graph_digest is None:
        raise ReferenceProductionLifecycleError(
            "runtime revision requires runtime_graph_digest for admission",
        )
    if revision.materialization_topology is None:
        raise ReferenceProductionLifecycleError(
            "runtime revision requires materialization_topology for admission",
        )

    admission_identity_digest = runtime_revision_admission_identity_digest(
        runtime_revision_id=runtime_revision_id,
        application_release_id=revision.application_release_id,
        platform_version=revision.platform_version,
        effective_roster_revision_id=revision.effective_roster_revision_id,
        lock_digest=revision.materialized_runtime_lock_digest,
        graph_digest=revision.runtime_graph_digest,
        materialization_topology=revision.materialization_topology.value,
        materialization_artifact_digest=artifact_digest,
        build_input_digest=revision.build_input_digest,
    )
    _authorize_admission(
        ctx=ctx,
        principal=principal,
        application_id=application_id,
        application_environment_id=application_environment_id,
        mutation_id=normalized_admission_mutation_id,
        runtime_revision_id=runtime_revision_id,
        identity_digest=admission_identity_digest,
    )

    revision_service = ctx.services.revision_service
    candidate = revision.model_copy(
        update={
            "revision_state": RuntimeRevisionState.CANDIDATE,
            "materialization_artifact_digest": artifact_digest,
        },
    )
    revision_service.persist_candidate_revision(candidate)
    validated = candidate.model_copy(
        update={"revision_state": RuntimeRevisionState.VALIDATED},
    )
    revision_service.mark_validated(runtime_revision_id, validated_revision=validated)

    ctx.services.projection_input_store.register(projection_input)

    activation = ctx.services.activation_service
    activation.prepare_candidate(
        application_id=application_id,
        application_environment_id=application_environment_id,
        runtime_revision_id=runtime_revision_id,
        artifact_locator=activation_request.artifact_locator,
    )
    serving_store = ctx.composition.agent_platform_runtime.stores.serving_store
    serving = serving_store.get_serving_record(application_id, application_environment_id)
    current_traffic_revision_id = (
        serving.traffic_serving_revision_id if serving is not None else None
    )
    current_pointer_revision = serving.serving_pointer_revision if serving is not None else 0
    _authorize_activation(
        ctx=ctx,
        principal=principal,
        application_id=application_id,
        application_environment_id=application_environment_id,
        mutation_id=activation_request.mutation_id,
        current_traffic_revision_id=current_traffic_revision_id,
        current_serving_pointer_revision=current_pointer_revision,
        target_runtime_revision_id=runtime_revision_id,
    )
    materialization_store = (
        ctx.composition.agent_platform_runtime.stores.materialization_store
    )
    materialization = materialization_store.get_by_revision(runtime_revision_id)
    if materialization is None:
        raise ReferenceProductionLifecycleError(
            f"missing canonical materialization record for {runtime_revision_id!r}",
        )
    topology = revision.materialization_topology
    if topology is None:
        raise ReferenceProductionLifecycleError(
            "runtime revision requires materialization_topology for descriptor",
        )
    descriptor = build_runtime_registry_projection_descriptor(
        projection_input,
        artifact_locator=materialization.artifact_locator,
        materialization_topology=topology,
    )
    ctx.services.descriptor_store.put(descriptor)
    committed = activation.commit_activation(
        application_id=application_id,
        application_environment_id=application_environment_id,
        runtime_revision_id=runtime_revision_id,
        expected_prior_traffic_revision_id=activation_request.expected_prior_traffic_revision_id,
        expected_serving_pointer_revision=activation_request.expected_serving_pointer_revision,
        expected_artifact_digest=activation_request.expected_artifact_digest,
    )

    resolved = bootstrap_production_registry_projection(
        application_id=application_id,
        application_environment_id=application_environment_id,
        stores=ctx.composition.agent_platform_runtime.stores,
    )
    if resolved.evidence.runtime_revision_id != runtime_revision_id:
        raise ReferenceProductionLifecycleError(
            "serving authority resolved unexpected runtime revision",
        )

    serving_record = committed.value.serving_record
    return ReferenceProductionLifecycleResult(
        runtime_revision_id=runtime_revision_id,
        application_id=application_id,
        application_environment_id=application_environment_id,
        serving_pointer_revision=serving_record.serving_pointer_revision,
        resolved_projection=resolved,
    )


def _authorize_admission(
    *,
    ctx: DeployActivatePipelineContext,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    runtime_revision_id: str,
    identity_digest: str,
) -> ControlPlaneMutationAuthorizationResult:
    request = build_admit_runtime_revision_mutation_request(
        principal=principal,
        application_id=application_id,
        application_environment_id=application_environment_id,
        mutation_id=mutation_id,
        runtime_revision_id=runtime_revision_id,
        identity_digest=identity_digest,
    )
    return _enforce_authorization_result(
        authorize_scoped_control_plane_mutation(
            boundary=ctx.mutation_authorization_boundary,
            tenant_resolver=ctx.environment_tenant_resolver,
            request=request,
        ),
    )


def _authorize_activation(
    *,
    ctx: DeployActivatePipelineContext,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    current_traffic_revision_id: str | None,
    current_serving_pointer_revision: int,
    target_runtime_revision_id: str,
) -> ControlPlaneMutationAuthorizationResult:
    request = build_activation_mutation_request(
        principal=principal,
        application_id=application_id,
        application_environment_id=application_environment_id,
        mutation_id=mutation_id,
        current_traffic_revision_id=current_traffic_revision_id,
        current_serving_pointer_revision=current_serving_pointer_revision,
        target_runtime_revision_id=target_runtime_revision_id,
    )
    return _enforce_authorization_result(
        authorize_scoped_control_plane_mutation(
            boundary=ctx.mutation_authorization_boundary,
            tenant_resolver=ctx.environment_tenant_resolver,
            request=request,
        ),
    )


def _enforce_authorization_result(
    result: ControlPlaneMutationAuthorizationResult,
) -> ControlPlaneMutationAuthorizationResult:
    if result.permitted:
        return result
    action = result.decision.action
    if action is PolicyAction.REQUIRE_HUMAN:
        raise ReferenceProductionLifecycleGovernanceBlockedError(
            "production activation requires governed human approval",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )
    if action is PolicyAction.ESCALATE:
        raise ReferenceProductionLifecycleGovernanceBlockedError(
            "production activation requires escalation",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )
    raise ReferenceProductionLifecycleGovernanceBlockedError(
        "production activation denied by control-plane governance",
        policy_action=action.value,
        authorization_evidence=result.evidence,
        authorization_scope=result.authorization_scope,
    )


__all__ = [
    "DeployActivatePipelineContext",
    "ReferenceProductionLifecycleError",
    "ReferenceProductionLifecycleGovernanceBlockedError",
    "ReferenceProductionLifecycleResult",
    "run_deploy_and_activate",
]
