# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Control-plane mutation helpers for Agent Distribution activation/rollback (CLA-CPM)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from intergrax.agent_distribution._digest import content_digest_for_model
from intergrax.agent_distribution.deployment import DrainPolicy
from intergrax.agent_distribution._immutable_json import (
    DistributionJsonValue,
    distribution_json_to_plain,
)
from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.evaluated_policy_decision import request_digest_for_payload
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationResult,
    ControlPlaneMutationPolicyEvaluator,
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

AGENT_DISTRIBUTION_RESOURCE_TYPE = "agent_distribution.application_environment"
MUTATION_TYPE_ACTIVATE_RUNTIME_REVISION = "agent_distribution.activate_runtime_revision"
MUTATION_TYPE_ROLLBACK_RUNTIME_REVISION = "agent_distribution.rollback_runtime_revision"
MUTATION_TYPE_INSTALL_AGENT = "agent_distribution.install_agent"
MUTATION_TYPE_BIND_AGENT = "agent_distribution.bind_agent"
MUTATION_TYPE_UPDATE_BINDING_CONFIG = "agent_distribution.update_binding_config"
MUTATION_TYPE_ENABLE_BINDING = "agent_distribution.enable_binding"
MUTATION_TYPE_DISABLE_BINDING = "agent_distribution.disable_binding"
MUTATION_TYPE_BUILD_RUNTIME_REVISION = "agent_distribution.build_runtime_revision"
MUTATION_TYPE_ADMIT_RUNTIME_REVISION = "agent_distribution.admit_runtime_revision"
MUTATION_TYPE_COMPLETE_DRAIN = "agent_distribution.complete_drain"
MUTATION_TYPE_MARK_POST_CUTOVER_FAILURE = (
    "agent_distribution.mark_post_cutover_failure"
)

_INSTALL_ABSENT_TOKEN = "inst:__absent__"
_BINDING_ABSENT_TOKEN = "binding:__absent__"


class ApplicationEnvironmentTenantResolver(Protocol):
    """Resolve tenant authority scope for one application environment."""

    def resolve_tenant_id(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> str:
        """Return canonical tenant id owning ``application_id`` / ``application_environment_id``."""


@dataclass(frozen=True, slots=True)
class StaticApplicationEnvironmentTenantResolver:
    """Explicit single-tenant host mapping for control-plane mutation authority."""

    tenant_id: str

    def resolve_tenant_id(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> str:
        del application_id, application_environment_id
        return self.tenant_id


def application_environment_resource_id(
    *,
    application_id: str,
    application_environment_id: str,
) -> str:
    return f"{application_id}:{application_environment_id}"


def application_environment_resource_scope(
    *,
    application_id: str,
    application_environment_id: str,
) -> str:
    return (
        f"agent_distribution.application:{application_id}"
        f".environment:{application_environment_id}"
    )


def binding_config_digest(config: Mapping[str, DistributionJsonValue]) -> str:
    """Deterministic digest for non-secret binding config governance evidence."""
    payload = {"config": distribution_json_to_plain(dict(config))}
    return request_digest_for_payload(payload)


def installation_absent_token(*, installation_slot_id: str) -> str:
    return f"slot:{installation_slot_id}|{_INSTALL_ABSENT_TOKEN}"


def installation_state_token(
    *,
    installation_slot_id: str,
    installation_id: str,
    installation_state: str,
    package_digest: str,
) -> str:
    return (
        f"slot:{installation_slot_id}|inst:{installation_id}"
        f"|state:{installation_state}|digest:{package_digest}"
    )


def installation_target_token(
    *,
    installation_slot_id: str,
    installation_id: str,
    package_digest: str,
) -> str:
    return (
        f"slot:{installation_slot_id}|inst:{installation_id}"
        f"|digest:{package_digest}|state:installed_active"
    )


def binding_absent_token() -> str:
    return _BINDING_ABSENT_TOKEN


def binding_state_token(
    *,
    application_binding_id: str,
    binding_revision: int,
    enablement: bool,
) -> str:
    enabled = "true" if enablement else "false"
    return f"binding:{application_binding_id}|rev:{binding_revision}|enabled:{enabled}"


def binding_create_target_token(
    *,
    application_binding_id: str,
    logical_agent_id: str,
    installation_slot_id: str,
    enablement: bool,
) -> str:
    enabled = "true" if enablement else "false"
    return (
        f"binding:{application_binding_id}|agent:{logical_agent_id}"
        f"|slot:{installation_slot_id}|rev:0|enabled:{enabled}"
    )


def binding_config_target_token(
    *,
    application_binding_id: str,
    next_revision: int,
    config_digest_value: str,
) -> str:
    return (
        f"binding:{application_binding_id}|rev:{next_revision}"
        f"|config:{config_digest_value}"
    )


def binding_enablement_target_token(
    *,
    application_binding_id: str,
    next_revision: int,
    enablement: bool,
) -> str:
    enabled = "true" if enablement else "false"
    return (
        f"binding:{application_binding_id}|rev:{next_revision}|enabled:{enabled}"
    )


def runtime_revision_absent_token(runtime_revision_id: str) -> str:
    return f"runtime_revision:{runtime_revision_id}|state:absent"


def build_input_digest(
    *,
    application_release_id: str,
    platform_version: str,
    python_version: str,
    source_context_root: str,
    application_source_root: str,
    agent_source_roots: tuple[tuple[str, str], ...],
    materialization_topology: str,
    repository_declaration: RepositoryDependencyDeclaration,
    resolver_algorithm_id: str,
    resolver_algorithm_version: str,
) -> str:
    """Deterministic digest of semantic build-driving inputs.

    ``output_root`` and other pure destination paths are excluded because they do
    not change artifact semantics — only where artifacts are written.
    """
    payload = {
        "application_release_id": application_release_id,
        "platform_version": platform_version,
        "python_version": python_version,
        "source_context_root": source_context_root,
        "application_source_root": application_source_root,
        "agent_source_roots": list(agent_source_roots),
        "materialization_topology": materialization_topology,
        "repository_declaration_digest": content_digest_for_model(repository_declaration),
        "resolver_algorithm_id": resolver_algorithm_id,
        "resolver_algorithm_version": resolver_algorithm_version,
    }
    return request_digest_for_payload(payload)


def build_runtime_revision_identity_digest(
    *,
    runtime_revision_id: str,
    application_release_id: str,
    platform_version: str,
    effective_roster_revision_id: str,
    lock_digest: str,
    graph_digest: str,
    materialization_topology: str,
    build_input_digest: str,
) -> str:
    payload = {
        "runtime_revision_id": runtime_revision_id,
        "application_release_id": application_release_id,
        "platform_version": platform_version,
        "effective_roster_revision_id": effective_roster_revision_id,
        "lock_digest": lock_digest,
        "graph_digest": graph_digest,
        "materialization_topology": materialization_topology,
        "build_input_digest": build_input_digest,
    }
    return request_digest_for_payload(payload)


def build_runtime_revision_target_token(
    *,
    runtime_revision_id: str,
    identity_digest: str,
) -> str:
    return f"runtime_revision:{runtime_revision_id}|digest:{identity_digest}"


def runtime_revision_admission_identity_digest(
    *,
    runtime_revision_id: str,
    application_release_id: str,
    platform_version: str,
    effective_roster_revision_id: str,
    lock_digest: str,
    graph_digest: str,
    materialization_topology: str,
    materialization_artifact_digest: str,
    build_input_digest: str | None,
) -> str:
    payload = {
        "runtime_revision_id": runtime_revision_id,
        "application_release_id": application_release_id,
        "platform_version": platform_version,
        "effective_roster_revision_id": effective_roster_revision_id,
        "lock_digest": lock_digest,
        "graph_digest": graph_digest,
        "materialization_topology": materialization_topology,
        "materialization_artifact_digest": materialization_artifact_digest,
        "build_input_digest": build_input_digest or "__absent__",
    }
    return request_digest_for_payload(payload)


def build_admit_runtime_revision_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    runtime_revision_id: str,
    identity_digest: str,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_ADMIT_RUNTIME_REVISION,
        principal=principal,
        resource_scope=application_environment_resource_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
        resource_id=application_environment_resource_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        current_revision=runtime_revision_absent_token(runtime_revision_id),
        target_revision=build_runtime_revision_target_token(
            runtime_revision_id=runtime_revision_id,
            identity_digest=identity_digest,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def serving_revision_token(
    *,
    traffic_revision_id: str | None,
    serving_pointer_revision: int,
) -> str:
    revision_token = traffic_revision_id or "__none__"
    return f"rev:{revision_token}|ptr:{serving_pointer_revision}"


def build_activation_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    current_traffic_revision_id: str | None,
    current_serving_pointer_revision: int,
    target_runtime_revision_id: str,
) -> ControlPlaneMutationRequest:
    expected_pointer = current_serving_pointer_revision + 1
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_ACTIVATE_RUNTIME_REVISION,
        principal=principal,
        resource_scope=application_environment_resource_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
        resource_id=application_environment_resource_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        current_revision=serving_revision_token(
            traffic_revision_id=current_traffic_revision_id,
            serving_pointer_revision=current_serving_pointer_revision,
        ),
        target_revision=serving_revision_token(
            traffic_revision_id=target_runtime_revision_id,
            serving_pointer_revision=expected_pointer,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def build_install_agent_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    installation_slot_id: str,
    installation_id: str,
    package_digest: str,
    current_revision: str,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_INSTALL_AGENT,
        principal=principal,
        resource_scope=application_environment_resource_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
        resource_id=application_environment_resource_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        current_revision=current_revision,
        target_revision=installation_target_token(
            installation_slot_id=installation_slot_id,
            installation_id=installation_id,
            package_digest=package_digest,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def build_bind_agent_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    application_binding_id: str,
    logical_agent_id: str,
    installation_slot_id: str,
    enablement: bool,
    current_revision: str,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_BIND_AGENT,
        principal=principal,
        resource_scope=application_environment_resource_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
        resource_id=application_environment_resource_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        current_revision=current_revision,
        target_revision=binding_create_target_token(
            application_binding_id=application_binding_id,
            logical_agent_id=logical_agent_id,
            installation_slot_id=installation_slot_id,
            enablement=enablement,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def build_update_binding_config_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    application_binding_id: str,
    expected_revision: int,
    config_digest_value: str,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_UPDATE_BINDING_CONFIG,
        principal=principal,
        resource_scope=application_environment_resource_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
        resource_id=application_environment_resource_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        current_revision=f"binding:{application_binding_id}|rev:{expected_revision}",
        target_revision=binding_config_target_token(
            application_binding_id=application_binding_id,
            next_revision=expected_revision + 1,
            config_digest_value=config_digest_value,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def build_enable_binding_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    application_binding_id: str,
    expected_revision: int,
    current_enablement: bool,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_ENABLE_BINDING,
        principal=principal,
        resource_scope=application_environment_resource_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
        resource_id=application_environment_resource_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        current_revision=binding_state_token(
            application_binding_id=application_binding_id,
            binding_revision=expected_revision,
            enablement=current_enablement,
        ),
        target_revision=binding_enablement_target_token(
            application_binding_id=application_binding_id,
            next_revision=expected_revision + 1,
            enablement=True,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def build_disable_binding_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    application_binding_id: str,
    expected_revision: int,
    current_enablement: bool,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_DISABLE_BINDING,
        principal=principal,
        resource_scope=application_environment_resource_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
        resource_id=application_environment_resource_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        current_revision=binding_state_token(
            application_binding_id=application_binding_id,
            binding_revision=expected_revision,
            enablement=current_enablement,
        ),
        target_revision=binding_enablement_target_token(
            application_binding_id=application_binding_id,
            next_revision=expected_revision + 1,
            enablement=False,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def build_runtime_revision_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    runtime_revision_id: str,
    identity_digest: str,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_BUILD_RUNTIME_REVISION,
        principal=principal,
        resource_scope=application_environment_resource_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
        resource_id=application_environment_resource_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        current_revision=runtime_revision_absent_token(runtime_revision_id),
        target_revision=build_runtime_revision_target_token(
            runtime_revision_id=runtime_revision_id,
            identity_digest=identity_digest,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


@dataclass(frozen=True, slots=True)
class PostCutoverRecoveryAuthority:
    """Scoped recovery continuation derived from a governed activation attempt."""

    application_id: str
    application_environment_id: str
    failed_runtime_revision_id: str
    originating_activation_mutation_id: str
    permitted_recovery_operation: Literal["rollback"]
    target_rollback_revision_id: str


def drain_policy_digest(policy: DrainPolicy) -> str:
    """Deterministic digest for drain policy governance identity."""
    payload = {
        "timeout_seconds": policy.timeout_seconds,
        "action_on_timeout": policy.action_on_timeout.value,
    }
    return request_digest_for_payload(payload)


def deployment_instance_state_token(
    *,
    runtime_revision_id: str,
    record_revision: int,
    instance_state: str,
) -> str:
    return (
        f"deployment_instance:{runtime_revision_id}"
        f"|rev:{record_revision}|state:{instance_state}"
    )


def deployment_instance_draining_token(
    *,
    runtime_revision_id: str,
    record_revision: int,
    serving_unit_ref: str,
    policy_digest: str,
) -> str:
    return (
        f"deployment_instance:{runtime_revision_id}|rev:{record_revision}"
        f"|state:draining|unit:{serving_unit_ref}|policy:{policy_digest}"
    )


def deployment_instance_drain_target_token(
    *,
    runtime_revision_id: str,
    next_record_revision: int,
    policy_digest: str,
) -> str:
    return (
        f"deployment_instance:{runtime_revision_id}|rev:{next_record_revision}"
        f"|outcome:drain_policy_complete|policy:{policy_digest}"
    )


def build_complete_drain_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    runtime_revision_id: str,
    record_revision: int,
    serving_unit_ref: str,
    policy: DrainPolicy,
) -> ControlPlaneMutationRequest:
    policy_digest = drain_policy_digest(policy)
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_COMPLETE_DRAIN,
        principal=principal,
        resource_scope=application_environment_resource_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
        resource_id=application_environment_resource_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        current_revision=deployment_instance_draining_token(
            runtime_revision_id=runtime_revision_id,
            record_revision=record_revision,
            serving_unit_ref=serving_unit_ref,
            policy_digest=policy_digest,
        ),
        target_revision=deployment_instance_drain_target_token(
            runtime_revision_id=runtime_revision_id,
            next_record_revision=record_revision + 1,
            policy_digest=policy_digest,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def build_mark_post_cutover_failure_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    runtime_revision_id: str,
    record_revision: int,
    current_instance_state: str,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_MARK_POST_CUTOVER_FAILURE,
        principal=principal,
        resource_scope=application_environment_resource_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
        resource_id=application_environment_resource_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        current_revision=deployment_instance_state_token(
            runtime_revision_id=runtime_revision_id,
            record_revision=record_revision,
            instance_state=current_instance_state,
        ),
        target_revision=deployment_instance_state_token(
            runtime_revision_id=runtime_revision_id,
            record_revision=record_revision + 1,
            instance_state="failed",
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def build_rollback_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    current_traffic_revision_id: str,
    current_serving_pointer_revision: int,
    target_runtime_revision_id: str,
) -> ControlPlaneMutationRequest:
    expected_pointer = current_serving_pointer_revision + 1
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_ROLLBACK_RUNTIME_REVISION,
        principal=principal,
        resource_scope=application_environment_resource_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
        resource_id=application_environment_resource_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        ),
        current_revision=serving_revision_token(
            traffic_revision_id=current_traffic_revision_id,
            serving_pointer_revision=current_serving_pointer_revision,
        ),
        target_revision=serving_revision_token(
            traffic_revision_id=target_runtime_revision_id,
            serving_pointer_revision=expected_pointer,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


class TenantScopedControlPlaneMutationEvaluator:
    """Fail closed when tenant authority or permission policy is not explicitly configured."""

    def __init__(
        self,
        *,
        tenant_resolver: ApplicationEnvironmentTenantResolver | None = None,
        inner: ControlPlaneMutationPolicyEvaluator | None = None,
    ) -> None:
        self._tenant_resolver = tenant_resolver
        self._inner = inner

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        if self._tenant_resolver is None:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="tenant_authority_not_configured",
                policy_rule_id="agent_distribution.tenant_scope",
            )
        application_id, application_environment_id = _parse_resource_id(request.resource_id)
        environment_tenant = self._tenant_resolver.resolve_tenant_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        if environment_tenant != request.principal.tenant_id:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="tenant_authority_mismatch",
                policy_rule_id="agent_distribution.tenant_scope",
            )
        if self._inner is None:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="control_plane_policy_not_configured",
                policy_rule_id="agent_distribution.control_plane_policy",
            )
        return self._inner.evaluate(request)


def compose_tenant_scoped_mutation_boundary(
    *,
    policy_evaluator: ControlPlaneMutationPolicyEvaluator,
    tenant_resolver: ApplicationEnvironmentTenantResolver,
) -> ControlPlaneMutationAuthorizationBoundary:
    """Compose canonical tenant scope + permission policy for one mutation boundary."""
    return ControlPlaneMutationAuthorizationBoundary(
        evaluator=TenantScopedControlPlaneMutationEvaluator(
            tenant_resolver=tenant_resolver,
            inner=policy_evaluator,
        )
    )


def authorize_scoped_control_plane_mutation(
    *,
    boundary: ControlPlaneMutationAuthorizationBoundary,
    tenant_resolver: ApplicationEnvironmentTenantResolver,
    request: ControlPlaneMutationRequest,
) -> ControlPlaneMutationAuthorizationResult:
    """Authorize one mutation with explicit tenant authority and configured policy."""
    scoped_boundary = compose_tenant_scoped_mutation_boundary(
        policy_evaluator=boundary.evaluator,
        tenant_resolver=tenant_resolver,
    )
    return scoped_boundary.authorize(request)


def _parse_resource_id(resource_id: str) -> tuple[str, str]:
    application_id, separator, application_environment_id = resource_id.partition(":")
    if not separator:
        raise ValueError("invalid agent distribution resource_id")
    return application_id, application_environment_id
