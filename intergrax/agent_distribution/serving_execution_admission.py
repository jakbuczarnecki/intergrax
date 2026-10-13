# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical serving execution admission evidence (PROD-Q-R1-R7-D1-R3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.errors import AgentDistributionError
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.stores import (
    ApplicationEnvironmentServingRecord,
    ApplicationEnvironmentServingStore,
    DeploymentInstanceStore,
    RuntimeRevisionStore,
)

_NON_EMPTY = Field(min_length=1)


class ServingExecutionAdmissionError(AgentDistributionError):
    """Current canonical serving state does not permit execution admission."""


class ServingExecutionAdmissionGrant(BaseModel):
    """Typed physical admission token bound to one canonical serving snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    runtime_revision_id: str = _NON_EMPTY
    serving_unit_ref: str = _NON_EMPTY
    materialization_artifact_digest: str = _NON_EMPTY
    serving_pointer_revision: int = Field(ge=0)


def _require_serving_record(
    serving_store: ApplicationEnvironmentServingStore,
    *,
    application_id: str,
    application_environment_id: str,
) -> ApplicationEnvironmentServingRecord:
    serving = serving_store.get_serving_record(application_id, application_environment_id)
    if serving is None:
        raise ServingExecutionAdmissionError("canonical serving record is missing")
    return serving


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
) -> ServingExecutionAdmissionGrant:
    """Read authoritative stores and fail closed unless execution may be admitted now."""
    serving = _require_serving_record(
        serving_store,
        application_id=application_id,
        application_environment_id=application_environment_id,
    )
    if serving.application_id != application_id:
        raise ServingExecutionAdmissionError("serving record application mismatch")
    if serving.application_environment_id != application_environment_id:
        raise ServingExecutionAdmissionError("serving record environment mismatch")
    if serving.traffic_serving_revision_id != runtime_revision_id:
        raise ServingExecutionAdmissionError(
            "canonical serving pointer does not target requested runtime revision"
        )

    revision = revision_store.get_revision(runtime_revision_id)
    if revision is None:
        raise ServingExecutionAdmissionError("runtime revision was not found")
    if revision.application_id != application_id:
        raise ServingExecutionAdmissionError("runtime revision application mismatch")
    if revision.application_environment_id != application_environment_id:
        raise ServingExecutionAdmissionError("runtime revision environment mismatch")
    if revision.revision_state is not RuntimeRevisionState.ACTIVE:
        raise ServingExecutionAdmissionError(
            "runtime revision is not active for execution admission"
        )
    if revision.materialization_artifact_digest != materialization_artifact_digest:
        raise ServingExecutionAdmissionError(
            "runtime revision artifact digest mismatch for execution admission"
        )

    instance = deployment_instance_store.get_instance(
        application_id,
        application_environment_id,
        runtime_revision_id,
    )
    if instance is None:
        raise ServingExecutionAdmissionError("canonical deployment instance is missing")
    if instance.instance_state is not DeploymentInstanceState.SERVING:
        raise ServingExecutionAdmissionError(
            "deployment instance is not in serving state for execution admission"
        )
    if instance.serving_unit_ref != serving_unit_ref:
        raise ServingExecutionAdmissionError(
            "deployment instance serving unit does not match requested unit"
        )

    return ServingExecutionAdmissionGrant(
        application_id=application_id,
        application_environment_id=application_environment_id,
        runtime_revision_id=runtime_revision_id,
        serving_unit_ref=serving_unit_ref,
        materialization_artifact_digest=materialization_artifact_digest,
        serving_pointer_revision=serving.serving_pointer_revision,
    )


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
    "ServingExecutionAdmissionGrant",
    "assert_activation_commit_matches_current_serving",
    "verify_current_serving_execution_admission_eligibility",
]
