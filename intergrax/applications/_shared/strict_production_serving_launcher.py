# © Artur Czarnecki. All rights reserved.

"""Sanctioned strict production serving opener (PROD-Q-R1-R7-D4 / ADR Decision C)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.agent_distribution.deployment import RuntimeDeploymentAdapter
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.colocated_serving_runtime_composition import (
    ColocatedServingActivationCommitResolutionError,
    bind_colocated_serving_runtime_after_activation,
    resolve_traffic_serving_activation_commit,
)
from intergrax.applications._shared.deploy_activate_pipeline import (
    ReferenceProductionLifecycleGovernanceBlockedError,
)
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.production_execution_continuation_store import (
    MissingContinuationFilePolicy,
)
from intergrax.applications._shared.production_lifecycle import (
    ProductionLifecycleConfigurationError,
    reject_reference_runtime_deployment_adapter,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.production_process_lifecycle import (
    ProductionProcessLifecycle,
    ProductionServingReconciliationError,
    open_production_process_lifecycle,
)
from intergrax.applications._shared.production_process_serving_reconciliation import (
    RECONCILIATION_OUTCOME_READY,
    ProductionServingReconciliationResult,
)
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications._shared.serving_runtime_handoff import ServingRuntimeHandoff
from intergrax.applications._shared.venv_bundle_colocated_serving_unit_runtime_accessor import (
    VenvBundleColocatedServingUnitRuntimeAccessor,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    resolve_production_artifact_root,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.runtime.execution.continuation.durable_serialized_seal_store import (
    SerializedSealExecutionContinuationStateStore,
)

if TYPE_CHECKING:
    from intergrax.applications._shared.production_agent_capability_runtime import (
        AgentCapabilityApplicationComposition,
    )


class StrictProductionServingLauncherError(ValueError):
    """Strict production serving opener precondition or binding failure."""


@dataclass(frozen=True, slots=True)
class StrictProductionServingEnvironment:
    """One traffic-serving application environment identity."""

    application_id: str
    application_environment_id: str


@dataclass(frozen=True, slots=True)
class StrictProductionProcessStorage:
    """Durable production storage paths required for one process."""

    agent_distribution_db: Path
    continuation_state_file: Path
    provider_invocation_db: Path
    trusted_artifact_parent_roots: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class StrictProductionActivationIntent:
    """Explicit operator-authorized deploy/activate for one environment."""

    projection_input: RegistryProjectionInputBundle
    activation_request: ActivateRuntimeRevisionRequest
    admission_mutation_id: str


@dataclass(frozen=True, slots=True)
class StrictProductionServingReadiness:
    """Derived readiness evidence — not execution permission."""

    process_lifecycle_open: bool
    production_dependencies_validated: bool
    serving_state_reconciled: bool
    serving_handoff_bound: bool
    execution_admission_eligible: bool
    ready_for_traffic: bool


@dataclass
class StrictProductionServingSession:
    """One opened strict production serving session owning the process lifecycle."""

    lifecycle: ProductionProcessLifecycle
    environment: StrictProductionServingEnvironment
    reconciliation: ProductionServingReconciliationResult
    readiness: StrictProductionServingReadiness
    _closed: bool = field(default=False, repr=False)

    @property
    def composition(self) -> ProductionProcessComposition:
        return self.lifecycle.composition

    @property
    def serving_runtime_handoff(self) -> ServingRuntimeHandoff:
        handoff = self.lifecycle.composition.serving_runtime_handoff
        if handoff is None:
            raise StrictProductionServingLauncherError(
                "strict production serving session has no bound serving_runtime_handoff",
            )
        return handoff

    def close(self) -> None:
        if self._closed:
            raise StrictProductionServingLauncherError(
                "strict production serving session already closed",
            )
        self.lifecycle.close_process_resources()
        self._closed = True


def _require_venv_production_adapter(
    deployment_adapter: RuntimeDeploymentAdapter,
) -> VenvBundleColocatedRuntimeDeploymentAdapter:
    try:
        reject_reference_runtime_deployment_adapter(deployment_adapter)
    except ProductionLifecycleConfigurationError as exc:
        raise StrictProductionServingLauncherError(str(exc)) from exc
    if not isinstance(deployment_adapter, VenvBundleColocatedRuntimeDeploymentAdapter):
        raise StrictProductionServingLauncherError(
            "strict production serving requires VenvBundleColocatedRuntimeDeploymentAdapter",
        )
    return deployment_adapter


def _bind_serving_handoff_for_environment(
    *,
    lifecycle: ProductionProcessLifecycle,
    physical_adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
    environment: StrictProductionServingEnvironment,
    reconciliation: ProductionServingReconciliationResult,
) -> ProductionProcessComposition:
    attestation = reconciliation.attestation
    if attestation is None:
        raise StrictProductionServingLauncherError(
            "serving reconciliation produced no attestation for traffic environment",
        )
    if reconciliation.evidence.outcome is not RECONCILIATION_OUTCOME_READY:
        raise StrictProductionServingLauncherError(
            "serving reconciliation did not reach ready outcome",
        )
    stores = lifecycle.composition.agent_platform_runtime.stores
    lifecycle_services = lifecycle.lifecycle_services
    revision_id = attestation.runtime_revision_id
    projection_input = lifecycle_services.projection_input_store.get(revision_id)
    if projection_input is None:
        raise StrictProductionServingLauncherError(
            f"missing projection input bundle for {revision_id!r}",
        )
    try:
        activation_commit = resolve_traffic_serving_activation_commit(
            serving_store=stores.serving_store,
            revision_store=stores.revision_store,
            deployment_instance_store=lifecycle_services.deployment_instance_store,
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
        )
    except ColocatedServingActivationCommitResolutionError as exc:
        raise StrictProductionServingLauncherError(str(exc)) from exc

    deployment_instance = activation_commit.candidate_instance
    topology = projection_input.runtime_revision.materialization_topology
    if topology is None:
        raise StrictProductionServingLauncherError(
            "projection input runtime revision lacks materialization_topology",
        )
    materialization = stores.materialization_store.get_by_revision(revision_id)
    if materialization is None:
        raise StrictProductionServingLauncherError(
            f"missing materialization record for {revision_id!r}",
        )
    artifact_root = resolve_production_artifact_root(materialization.artifact_locator)
    accessor = VenvBundleColocatedServingUnitRuntimeAccessor(physical_adapter)
    if topology is not MaterializationTopology.VENV_BUNDLE:
        raise StrictProductionServingLauncherError(
            "strict production serving supports VENV_BUNDLE topology only",
        )
    unit_resolver = accessor.runtime_agent_factory_resolver_for_unit(
        serving_unit_ref=attestation.serving_unit_ref,
        revision=projection_input.runtime_revision,
        artifact_root=artifact_root,
    )
    if projection_input.factory_resolver is not unit_resolver:
        projection_input = projection_input.__class__(
            runtime_revision=projection_input.runtime_revision,
            effective_roster=projection_input.effective_roster,
            manifest=projection_input.manifest,
            build_context=projection_input.build_context,
            composition=projection_input.composition,
            factory_resolver=unit_resolver,
            builders=projection_input.builders,
            materialization_artifact_digest=projection_input.materialization_artifact_digest,
            roster_contract_authority=projection_input.roster_contract_authority,
        )
    return bind_colocated_serving_runtime_after_activation(
        composition=lifecycle.composition,
        activation_commit=activation_commit,
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=projection_input,
        colocated_accessor=accessor,
        physical_adapter=physical_adapter,
        serving_store=stores.serving_store,
        deployment_instance_store=lifecycle_services.deployment_instance_store,
        revision_store=stores.revision_store,
        topology=topology,
        artifact_root=artifact_root,
    )


def open_strict_production_serving_session(
    *,
    storage: StrictProductionProcessStorage,
    environment: StrictProductionServingEnvironment,
    governance: ProductionControlPlaneGovernance,
    deployment_adapter: RuntimeDeploymentAdapter,
    activation_intent: StrictProductionActivationIntent | None = None,
    missing_continuation_file_policy: MissingContinuationFilePolicy = "fail",
    application_composition: AgentCapabilityApplicationComposition | None = None,
    composition_builder: Callable[
        [
            SerializedSealExecutionContinuationStateStore,
            ProviderInvocationStore,
            ProductionProcessComposition,
        ],
        ProductionProcessComposition,
    ]
    | None = None,
) -> StrictProductionServingSession:
    """
    Open one strict production serving session: lifecycle, reconciliation, handoff binding.

    When ``activation_intent`` is omitted, uses reconciled durable ACTIVE serving state only
    (no redundant activation mutation). When provided, runs ``deploy_and_activate`` first.
    """
    physical_adapter = _require_venv_production_adapter(deployment_adapter)
    if governance is None:
        raise StrictProductionServingLauncherError(
            "strict production serving requires ProductionControlPlaneGovernance",
        )
    if activation_intent is not None:
        revision = activation_intent.projection_input.runtime_revision
        if (
            revision.application_id != environment.application_id
            or revision.application_environment_id
            != environment.application_environment_id
        ):
            raise StrictProductionServingLauncherError(
                "activation intent revision scope does not match serving environment",
            )

    lifecycle: ProductionProcessLifecycle | None = None
    try:
        try:
            lifecycle = open_production_process_lifecycle(
                agent_distribution_db=storage.agent_distribution_db,
                continuation_state_file=storage.continuation_state_file,
                deployment_adapter=physical_adapter,
                governance=governance,
                trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
                missing_file_policy=missing_continuation_file_policy,
                provider_invocation_db=storage.provider_invocation_db,
                application_composition=application_composition,
                composition_builder=composition_builder,
                serving_reconciliation_targets=(
                    (environment.application_id, environment.application_environment_id),
                ),
            )
        except ProductionServingReconciliationError as reconcile_exc:
            raise StrictProductionServingLauncherError(str(reconcile_exc)) from reconcile_exc
        if activation_intent is not None:
            try:
                lifecycle.launcher.deploy_and_activate(
                    activation_intent.projection_input,
                    activation_intent.activation_request,
                    principal=governance.principal,
                    admission_mutation_id=activation_intent.admission_mutation_id,
                )
            except ReferenceProductionLifecycleGovernanceBlockedError as exc:
                raise StrictProductionServingLauncherError(str(exc)) from exc

        try:
            reconciliation = lifecycle.reconcile_serving_environment(
                environment.application_id,
                environment.application_environment_id,
            )
        except ProductionServingReconciliationError as exc:
            raise StrictProductionServingLauncherError(str(exc)) from exc

        updated = _bind_serving_handoff_for_environment(
            lifecycle=lifecycle,
            physical_adapter=physical_adapter,
            environment=environment,
            reconciliation=reconciliation,
        )
        lifecycle.composition = updated
        handoff = updated.serving_runtime_handoff
        admission_eligible = (
            handoff is not None and handoff.colocated_execution_admission is not None
        )
        readiness = StrictProductionServingReadiness(
            process_lifecycle_open=True,
            production_dependencies_validated=True,
            serving_state_reconciled=(
                reconciliation.evidence.outcome is RECONCILIATION_OUTCOME_READY
            ),
            serving_handoff_bound=handoff is not None,
            execution_admission_eligible=admission_eligible,
            ready_for_traffic=handoff is not None and admission_eligible,
        )
        if not readiness.ready_for_traffic:
            raise StrictProductionServingLauncherError(
                "strict production serving prerequisites incomplete",
            )
        return StrictProductionServingSession(
            lifecycle=lifecycle,
            environment=environment,
            reconciliation=reconciliation,
            readiness=readiness,
        )
    except BaseException:
        if lifecycle is not None:
            try:
                lifecycle.close_process_resources()
            except BaseException:
                pass
        raise


def require_strict_production_activation_intent_principal(
    governance: ProductionControlPlaneGovernance,
) -> RequestIdentity:
    if governance.principal is None:
        raise ProductionLifecycleConfigurationError(
            "production governance requires authentic principal identity",
        )
    return governance.principal


__all__ = [
    "StrictProductionActivationIntent",
    "StrictProductionProcessStorage",
    "StrictProductionServingEnvironment",
    "StrictProductionServingLauncherError",
    "StrictProductionServingReadiness",
    "StrictProductionServingSession",
    "open_strict_production_serving_session",
    "require_strict_production_activation_intent_principal",
]
