# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical serving execution admission evidence (PROD-Q-R1-R7-D1-R6)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.agent_distribution.activation_domain_coordinator import (
    ServingExecutionAdmissionRequest,
    ServingExecutionEligibilityEvidence,
    verify_serving_execution_eligibility_from_state,
)
from intergrax.agent_distribution.errors import ServingExecutionAdmissionError
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentServingStore,
    run_under_activation_domain,
)
from intergrax.agent_distribution.stores import (
    ApplicationEnvironmentServingRecord,
    ApplicationEnvironmentServingStore,
    DeploymentInstanceStore,
    RuntimeRevisionStore,
)

def shared_activation_domain_state_from_serving_store(
    serving_store: ApplicationEnvironmentServingStore,
) -> AgentDistributionStoreState | None:
    if isinstance(serving_store, InMemoryApplicationEnvironmentServingStore):
        return serving_store.state
    return None


def require_shared_activation_domain_state(
    serving_store: ApplicationEnvironmentServingStore,
) -> AgentDistributionStoreState:
    state = shared_activation_domain_state_from_serving_store(serving_store)
    if state is None:
        raise ServingExecutionAdmissionError(
            "serving execution admission requires shared activation domain store state; "
            "use ActivationDomainCoordinator for sqlite-backed stores"
        )
    return state


def run_serving_execution_admission_critical_section[T](
    serving_store: ApplicationEnvironmentServingStore,
    fn: Callable[[], T],
) -> T:
    """Serialize execution admission against activation pointer transitions on one bundle."""
    state = require_shared_activation_domain_state(serving_store)
    return run_under_activation_domain(state, fn)


def verify_current_serving_execution_admission_eligibility(
    *,
    serving_store: ApplicationEnvironmentServingStore,
    deployment_instance_store: DeploymentInstanceStore,
    revision_store: RuntimeRevisionStore,
    application_id: str,
    application_environment_id: str,
    runtime_revision_id: str,
    serving_unit_ref: str,
    materialization_artifact_digest: str,
) -> ServingExecutionEligibilityEvidence:
    """Read authoritative in-memory domain state and fail closed unless execution may be admitted."""
    del deployment_instance_store, revision_store
    state = require_shared_activation_domain_state(serving_store)
    request = ServingExecutionAdmissionRequest(
        application_id=application_id,
        application_environment_id=application_environment_id,
        runtime_revision_id=runtime_revision_id,
        serving_unit_ref=serving_unit_ref,
        materialization_artifact_digest=materialization_artifact_digest,
    )

    def _verify() -> ServingExecutionEligibilityEvidence:
        return verify_serving_execution_eligibility_from_state(state, request)

    return run_under_activation_domain(state, _verify)


def assert_activation_commit_matches_current_serving(
    *,
    activation_commit_serving: ApplicationEnvironmentServingRecord,
    current_serving: ApplicationEnvironmentServingRecord,
) -> None:
    """Reject stale activation commit evidence when durable pointer has advanced."""
    if activation_commit_serving.application_id != current_serving.application_id:
        raise ServingExecutionAdmissionError(
            "activation commit serving application disagrees with canonical record"
        )
    if (
        activation_commit_serving.application_environment_id
        != current_serving.application_environment_id
    ):
        raise ServingExecutionAdmissionError(
            "activation commit serving environment disagrees with canonical record"
        )
    if (
        activation_commit_serving.traffic_serving_revision_id
        != current_serving.traffic_serving_revision_id
    ):
        raise ServingExecutionAdmissionError(
            "activation commit serving revision disagrees with canonical pointer"
        )
    if (
        activation_commit_serving.serving_pointer_revision
        != current_serving.serving_pointer_revision
    ):
        raise ServingExecutionAdmissionError("activation commit serving pointer is stale")


__all__ = [
    "ServingExecutionAdmissionError",
    "ServingExecutionEligibilityEvidence",
    "assert_activation_commit_matches_current_serving",
    "require_shared_activation_domain_state",
    "run_serving_execution_admission_critical_section",
    "shared_activation_domain_state_from_serving_store",
    "verify_current_serving_execution_admission_eligibility",
]
