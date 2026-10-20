# © Artur Czarnecki. All rights reserved.

"""Sanctioned GCA strict startup seam for single-node production profile v1 (OD-R2-03)."""

from __future__ import annotations

from intergrax.agent_distribution.deployment import RuntimeDeploymentAdapter
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.production_execution_continuation_store import (
    MissingContinuationFilePolicy,
)
from intergrax.applications._shared.production_agent_capability_runtime import (
    AgentCapabilityApplicationComposition,
)
from intergrax.applications._shared.settings_loader import EnvReader
from intergrax.applications._shared.single_node_production_profile_v1 import (
    SingleNodeProductionProfileV1Configuration,
    SingleNodeProductionProfileV1CrossProcessWriterHold,
    load_validate_and_admit_single_node_production_profile_v1_from_env,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionActivationIntent,
    StrictProductionProcessStorage,
    StrictProductionServingEnvironment,
    StrictProductionServingSession,
)
from governed_contractor_application.host.sqlite_governed_external_work_host_store_bundle import (
    SQLiteGovernedExternalWorkHostStoreBackend,
)
from governed_contractor_application.host.strict_gca_production_process_composition import (
    open_gca_strict_production_serving_session,
)


def open_gca_strict_production_serving_session_from_single_node_production_profile_v1(
    *,
    environment: StrictProductionServingEnvironment,
    governance: ProductionControlPlaneGovernance,
    deployment_adapter: RuntimeDeploymentAdapter,
    activation_intent: StrictProductionActivationIntent | None = None,
    missing_continuation_file_policy: MissingContinuationFilePolicy = "fail",
    application_composition: AgentCapabilityApplicationComposition | None = None,
    env: EnvReader | None = None,
) -> tuple[
    StrictProductionServingSession,
    SQLiteGovernedExternalWorkHostStoreBackend,
    SingleNodeProductionProfileV1Configuration,
    SingleNodeProductionProfileV1CrossProcessWriterHold,
]:
    """
    Load profile v1 from environment, admit CONFIGURED invariants + cross-process writer lock,
    then open the sanctioned GCA strict production serving session.
    """
    config, storage, writer_hold = load_validate_and_admit_single_node_production_profile_v1_from_env(
        env=env,
        acquire_cross_process_writer=True,
    )
    session, owned_gca = open_gca_strict_production_serving_session(
        storage=storage,
        environment=environment,
        governance=governance,
        deployment_adapter=deployment_adapter,
        activation_intent=activation_intent,
        missing_continuation_file_policy=missing_continuation_file_policy,
        application_composition=application_composition,
    )
    return session, owned_gca, config, writer_hold


def resolve_strict_production_storage_from_single_node_production_profile_v1(
    *,
    env: EnvReader | None = None,
    acquire_cross_process_writer: bool = False,
) -> tuple[
    SingleNodeProductionProfileV1Configuration,
    StrictProductionProcessStorage,
    SingleNodeProductionProfileV1CrossProcessWriterHold,
]:
    """CONFIGURED admission without opening a strict serving session (preflight / gates)."""
    return load_validate_and_admit_single_node_production_profile_v1_from_env(
        env=env,
        acquire_cross_process_writer=acquire_cross_process_writer,
    )


__all__ = [
    "open_gca_strict_production_serving_session_from_single_node_production_profile_v1",
    "resolve_strict_production_storage_from_single_node_production_profile_v1",
]
