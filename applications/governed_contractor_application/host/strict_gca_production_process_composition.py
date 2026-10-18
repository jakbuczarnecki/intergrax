# © Artur Czarnecki. All rights reserved.

"""Strict production composition helpers for durable GCA host stores."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from governed_contractor_application.host.governed_external_work_host_stores import (
    GovernedExternalWorkHostStoreBundle,
)
from governed_contractor_application.host.sqlite_governed_external_work_host_store_bundle import (
    SQLiteGovernedExternalWorkHostStoreBackend,
    open_sqlite_governed_external_work_host_store_bundle_at,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.production_execution_continuation_store import (
    MissingContinuationFilePolicy,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionActivationIntent,
    StrictProductionProcessStorage,
    StrictProductionServingEnvironment,
    StrictProductionServingSession,
    open_strict_production_serving_session,
)
from intergrax.agent_distribution.deployment import RuntimeDeploymentAdapter
from intergrax.applications._shared.production_agent_capability_runtime import (
    AgentCapabilityApplicationComposition,
)
from intergrax.runtime.execution.continuation.durable_serialized_seal_store import (
    SerializedSealExecutionContinuationStateStore,
)


def build_strict_durable_gca_composition_builder(
    owned_gca_host_stores: SQLiteGovernedExternalWorkHostStoreBackend,
) -> Callable[
    [
        SerializedSealExecutionContinuationStateStore,
        ProviderInvocationStore,
        ProductionProcessComposition,
    ],
    ProductionProcessComposition,
]:
    """Bind an opened durable GCA bundle into the sanctioned production composition builder."""

    bundle: GovernedExternalWorkHostStoreBundle = owned_gca_host_stores.bundle

    def _build(
        continuation_store: SerializedSealExecutionContinuationStateStore,
        provider_invocation_store: ProviderInvocationStore,
        base: ProductionProcessComposition,
    ) -> ProductionProcessComposition:
        return ProductionProcessComposition(
            agent_platform_runtime=base.agent_platform_runtime,
            agent_capability_runtime=base.agent_capability_runtime,
            provider_invocation_store=provider_invocation_store,
            execution_continuation_state_store=continuation_store,
            governed_external_work_host_store_bundle=bundle,
        )

    return _build


def open_owned_sqlite_governed_external_work_host_stores(
    db_path: Path,
) -> SQLiteGovernedExternalWorkHostStoreBackend:
    """Open process-scoped durable GCA host stores at a trusted configuration path."""
    return open_sqlite_governed_external_work_host_store_bundle_at(db_path)


def open_gca_strict_production_serving_session(
    *,
    storage: StrictProductionProcessStorage,
    environment: StrictProductionServingEnvironment,
    governance: ProductionControlPlaneGovernance,
    deployment_adapter: RuntimeDeploymentAdapter,
    activation_intent: StrictProductionActivationIntent | None = None,
    missing_continuation_file_policy: MissingContinuationFilePolicy = "fail",
    application_composition: AgentCapabilityApplicationComposition | None = None,
) -> tuple[StrictProductionServingSession, SQLiteGovernedExternalWorkHostStoreBackend]:
    """Open strict production serving with durable SQLite GCA host stores."""
    owned_gca = open_owned_sqlite_governed_external_work_host_stores(
        storage.governed_external_work_host_db,
    )
    session = open_strict_production_serving_session(
        storage=storage,
        environment=environment,
        governance=governance,
        deployment_adapter=deployment_adapter,
        activation_intent=activation_intent,
        missing_continuation_file_policy=missing_continuation_file_policy,
        application_composition=application_composition,
        composition_builder=build_strict_durable_gca_composition_builder(owned_gca),
        owned_governed_external_work_host_store_bundle=owned_gca,
    )
    return session, owned_gca


__all__ = [
    "build_strict_durable_gca_composition_builder",
    "open_gca_strict_production_serving_session",
    "open_owned_sqlite_governed_external_work_host_stores",
]
