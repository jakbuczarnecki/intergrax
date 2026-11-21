# © Artur Czarnecki. All rights reserved.

"""Shared deploy → project → activate orchestration for production and reference launchers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.agent_distribution.control_plane_governance import (
    ApplicationEnvironmentTenantResolver,
    authorize_scoped_control_plane_mutation,
    build_activation_mutation_request,
    build_admit_runtime_revision_mutation_request,
    runtime_revision_admission_identity_digest,
)
from intergrax.agent_distribution.runtime_revision import RuntimeRevision, RuntimeRevisionState
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
    RegistryProjectionInputBundle,
)
from intergrax.applications._shared.registry_projection_descriptor import (
    build_runtime_registry_projection_descriptor,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationResult,
    ControlPlaneMutationAuthorizationScope,
)
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
    trust_admission_state: RevisionTrustAdmissionState | None = None


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
    # Optional trust re-check of the revision (T-IGX-041-A, F-D1-02), e.g.
    # ``AgentPlatformAdminService.assert_revision_trust_admissible``. It must raise to refuse.
    # ``None`` resolves the check from the composed capability runtime's admin service; a
    # composition without a capability runtime is an explicit pass-through
    # (``RevisionTrustAdmissionState.NOT_COMPOSED``, T-IGX-051-B Part B/C).
    revision_trust_admission: Callable[[RuntimeRevision], None] | None = None


class RevisionTrustAdmissionState(StrEnum):
    """How the deploy/activate pipeline admitted the revision's packages (recorded on the result).

    ``COMPOSED_RUNTIME``: the composed admin service's ``assert_revision_trust_admissible`` re-evaluated
    the installation trust records of every enabled roster entry (presence, digest, revocation, policy,
    freshness) against the revocation, policy and time sources of the composition's ``package_trust``
    (T-IGX-051-N Part B). Attestation material is verified at install and upgrade (Part A), not here.
    A lab composition (``package_trust.is_lab``, ``lab_package_trust_composition``) has an empty
    revocation state and admits unsigned packages, so on a lab composition nothing is revoked here.

    ``NOT_COMPOSED``: no capability runtime is composed (for example the shipped GCA strict path); the
    pipeline passes through without a trust check.
    """

    CONTEXT_HOOK = "trust_admission_context_hook"
    COMPOSED_RUNTIME = "trust_admission_composed_runtime"
    NOT_COMPOSED = "trust_admission_not_composed"


def _resolve_revision_trust_admission(
    ctx: DeployActivatePipelineContext,
) -> tuple[Callable[[RuntimeRevision], None] | None, RevisionTrustAdmissionState]:
    if ctx.revision_trust_admission is not None:
        return ctx.revision_trust_admission, RevisionTrustAdmissionState.CONTEXT_HOOK
    capability_runtime = ctx.composition.agent_capability_runtime
    if capability_runtime is not None:
        return (
            capability_runtime.admin_service.assert_revision_trust_admissible,
            RevisionTrustAdmissionState.COMPOSED_RUNTIME,
        )
    return None, RevisionTrustAdmissionState.NOT_COMPOSED


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
    # A-01: the trust admission runs before any store write; a refused revision is not
    # persisted, validated, registered or prepared. One check, one place.
    trust_admission, trust_admission_state = _resolve_revision_trust_admission(ctx)
    if trust_admission is not None:
        trust_admission(candidate)
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
    current_traffic_revision_id = serving.traffic_serving_revision_id if serving is not None else None
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
    materialization_store = ctx.composition.agent_platform_runtime.stores.materialization_store
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
        trust_admission_state=trust_admission_state,
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
    "RevisionTrustAdmissionState",
    "run_deploy_and_activate",
]
