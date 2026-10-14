# © Artur Czarnecki. All rights reserved.

"""Sanctioned production lifecycle service wiring (PROD-Q-R1-R7-D2 / ADR DEC-03)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from intergrax.agent_distribution.activation import ActivationService
from intergrax.agent_distribution.deployment import (
    FakeInMemoryRuntimeDeploymentAdapter,
    RuntimeDeploymentAdapter,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.sqlite_stores import SqliteAgentDistributionStoreBundle
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.colocated_serving_runtime_composition import (
    validate_colocated_serving_admission_topology,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.registry_projection import (
    ApplicationRegistryProjectionCoordinator,
    InMemoryRegistryProjectionInputStore,
)
from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.agent_distribution.control_plane_governance import (
    ApplicationEnvironmentTenantResolver,
)
from intergrax.applications._shared.deploy_activate_pipeline import (
    DeployActivatePipelineContext,
    run_deploy_and_activate,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleResult,
    ReferenceProductionLifecycleServices,
)
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

if TYPE_CHECKING:
    from intergrax.applications._shared.production_control_plane_governance_wiring import (
        ProductionControlPlaneGovernance,
    )


class ProductionLifecycleConfigurationError(ValueError):
    """Invalid or incomplete production lifecycle dependency wiring."""


ProductionLifecycleServices = ReferenceProductionLifecycleServices


def reject_reference_runtime_deployment_adapter(
    deployment_adapter: RuntimeDeploymentAdapter,
) -> None:
    """Fail closed when production wiring would use reference/test deployment authority."""
    if isinstance(deployment_adapter, FakeInMemoryRuntimeDeploymentAdapter):
        raise ProductionLifecycleConfigurationError(
            "production lifecycle forbids FakeInMemoryRuntimeDeploymentAdapter",
        )


def validate_production_trusted_artifact_configuration(
    *,
    trusted_artifact_parent_roots: tuple[Path, ...],
    deployment_adapter: RuntimeDeploymentAdapter,
) -> None:
    """Require explicit artifact trust roots for supported A1 VENV production adapters."""
    if not trusted_artifact_parent_roots:
        raise ProductionLifecycleConfigurationError(
            "production lifecycle requires non-empty trusted_artifact_parent_roots",
        )
    if isinstance(deployment_adapter, VenvBundleColocatedRuntimeDeploymentAdapter):
        if deployment_adapter._trusted_roots is None:
            raise ProductionLifecycleConfigurationError(
                "VenvBundleColocatedRuntimeDeploymentAdapter requires "
                "trusted_artifact_parent_roots at construction for production wiring",
            )


def validate_production_durable_store_bundle_consistency(
    composition: ProductionProcessComposition,
    durable_store_bundle: SqliteAgentDistributionStoreBundle,
) -> None:
    """Reject split canonical store universes across composition and durable bundle."""
    stores = composition.agent_platform_runtime.stores
    if stores.revision_store is not durable_store_bundle.revision_store:
        raise ProductionLifecycleConfigurationError(
            "composition revision_store must be the durable Sqlite bundle revision_store",
        )
    if stores.serving_store is not durable_store_bundle.serving_store:
        raise ProductionLifecycleConfigurationError(
            "composition serving_store must be the durable Sqlite bundle serving_store",
        )
    if stores.materialization_store is not durable_store_bundle.materialization_store:
        raise ProductionLifecycleConfigurationError(
            "composition materialization_store must match durable Sqlite bundle",
        )
    validate_colocated_serving_admission_topology(
        serving_store=durable_store_bundle.serving_store,
        deployment_instance_store=durable_store_bundle.deployment_instance_store,
        revision_store=durable_store_bundle.revision_store,
    )


def wire_production_lifecycle_services(
    composition: ProductionProcessComposition,
    *,
    durable_store_bundle: SqliteAgentDistributionStoreBundle,
    deployment_adapter: RuntimeDeploymentAdapter,
    trusted_artifact_parent_roots: tuple[Path, ...],
) -> ProductionLifecycleServices:
    """Construct production AP lifecycle services with required physical adapter and durable stores."""
    reject_reference_runtime_deployment_adapter(deployment_adapter)
    validate_production_trusted_artifact_configuration(
        trusted_artifact_parent_roots=trusted_artifact_parent_roots,
        deployment_adapter=deployment_adapter,
    )
    validate_production_durable_store_bundle_consistency(
        composition,
        durable_store_bundle,
    )
    stores = composition.agent_platform_runtime.stores
    revision_store = durable_store_bundle.revision_store
    projection_input_store = InMemoryRegistryProjectionInputStore()
    descriptor_store = durable_store_bundle.projection_descriptor_store
    projection_coordinator = ApplicationRegistryProjectionCoordinator(
        revision_store=revision_store,
        input_store=projection_input_store,
        projection_store=stores.registry_projection_store,
    )
    deployment_instance_store = durable_store_bundle.deployment_instance_store
    activation_service = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=deployment_instance_store,
        serving_store=durable_store_bundle.serving_store,
        activation_store=durable_store_bundle.activation_store,
        deployment_adapter=deployment_adapter,
        projection_coordinator=projection_coordinator,
        materialization_store=durable_store_bundle.materialization_store,
    )
    return ProductionLifecycleServices(
        activation_service=activation_service,
        projection_coordinator=projection_coordinator,
        revision_service=RuntimeRevisionService(revision_store),
        projection_input_store=projection_input_store,
        descriptor_store=descriptor_store,
        deployment_instance_store=deployment_instance_store,
    )


class ProductionLifecycleLauncher:
    """Production deploy/project/activate — explicit services and governance only."""

    def __init__(
        self,
        composition: ProductionProcessComposition,
        *,
        services: ProductionLifecycleServices,
        mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary,
        environment_tenant_resolver: ApplicationEnvironmentTenantResolver,
    ) -> None:
        if mutation_authorization_boundary is None:
            raise ProductionLifecycleConfigurationError(
                "production lifecycle launcher requires mutation_authorization_boundary",
            )
        if environment_tenant_resolver is None:
            raise ProductionLifecycleConfigurationError(
                "production lifecycle launcher requires environment_tenant_resolver",
            )
        self._composition = composition
        self._services = services
        self._mutation_authorization_boundary = mutation_authorization_boundary
        self._environment_tenant_resolver = environment_tenant_resolver
        self._pipeline = DeployActivatePipelineContext(
            composition=composition,
            services=services,
            mutation_authorization_boundary=mutation_authorization_boundary,
            environment_tenant_resolver=environment_tenant_resolver,
        )

    @property
    def process_composition(self) -> ProductionProcessComposition:
        return self._composition

    @property
    def services(self) -> ProductionLifecycleServices:
        return self._services

    def deploy_and_activate(
        self,
        projection_input: RegistryProjectionInputBundle,
        activation_request: ActivateRuntimeRevisionRequest,
        *,
        principal: RequestIdentity,
        admission_mutation_id: str,
    ) -> ReferenceProductionLifecycleResult:
        return run_deploy_and_activate(
            self._pipeline,
            projection_input,
            activation_request,
            principal=principal,
            admission_mutation_id=admission_mutation_id,
        )


def require_production_control_plane_governance(
    governance: ProductionControlPlaneGovernance | None,
) -> ProductionControlPlaneGovernance:
    if governance is None:
        raise ProductionLifecycleConfigurationError(
            "production lifecycle requires ProductionControlPlaneGovernance",
        )
    if governance.mutation_authorization_boundary is None:
        raise ProductionLifecycleConfigurationError(
            "production lifecycle requires mutation_authorization_boundary",
        )
    if governance.environment_tenant_resolver is None:
        raise ProductionLifecycleConfigurationError(
            "production lifecycle requires environment_tenant_resolver",
        )
    return governance


__all__ = [
    "ProductionLifecycleConfigurationError",
    "ProductionLifecycleLauncher",
    "ProductionLifecycleServices",
    "reject_reference_runtime_deployment_adapter",
    "require_production_control_plane_governance",
    "validate_production_durable_store_bundle_consistency",
    "validate_production_trusted_artifact_configuration",
    "wire_production_lifecycle_services",
]
