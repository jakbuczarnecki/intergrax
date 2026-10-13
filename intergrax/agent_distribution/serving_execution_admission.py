# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical serving execution admission evidence (PROD-Q-R1-R7-D1-R3)."""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.errors import AgentDistributionError
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
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

_NON_EMPTY = Field(min_length=1)

_PENDING_GRANT_ISSUANCES: dict[str, tuple[str, str, str, str, str, int]] = {}
_ACTIVATION_CRITICAL_DEPTH = threading.local()


class ServingExecutionAdmissionError(AgentDistributionError):
    """Current canonical serving state does not permit execution admission."""


class ServingExecutionAdmissionGrant(BaseModel):
    """Typed snapshot of canonical serving eligibility — not authority without issuance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    runtime_revision_id: str = _NON_EMPTY
    serving_unit_ref: str = _NON_EMPTY
    materialization_artifact_digest: str = _NON_EMPTY
    serving_pointer_revision: int = Field(ge=0)
    _issuance_nonce: str | None = PrivateAttr(default=None)


def _grant_fingerprint(
    grant: ServingExecutionAdmissionGrant,
) -> tuple[str, str, str, str, str, int]:
    return (
        grant.application_id,
        grant.application_environment_id,
        grant.runtime_revision_id,
        grant.serving_unit_ref,
        grant.materialization_artifact_digest,
        grant.serving_pointer_revision,
    )


def _register_canonical_grant_issuance(grant: ServingExecutionAdmissionGrant) -> None:
    nonce = grant._issuance_nonce
    if nonce is None:
        raise ServingExecutionAdmissionError("canonical grant issuance is missing")
    _PENDING_GRANT_ISSUANCES[nonce] = _grant_fingerprint(grant)


def consume_serving_execution_admission_grant(
    grant: ServingExecutionAdmissionGrant,
) -> None:
    """Consume one canonical grant before physical admission may open (fail closed)."""
    nonce = grant._issuance_nonce
    if nonce is None:
        raise ServingExecutionAdmissionError(
            "serving execution admission grant was not issued by canonical authority"
        )
    if getattr(_ACTIVATION_CRITICAL_DEPTH, "depth", 0) < 1:
        raise ServingExecutionAdmissionError(
            "serving execution admission grant consumption requires activation critical section"
        )
    fingerprint = _PENDING_GRANT_ISSUANCES.pop(nonce, None)
    if fingerprint is None or fingerprint != _grant_fingerprint(grant):
        raise ServingExecutionAdmissionError(
            "serving execution admission grant issuance is invalid or already consumed"
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
            "serving execution admission requires shared activation domain store state"
        )
    return state


def run_serving_execution_admission_critical_section[T](
    serving_store: ApplicationEnvironmentServingStore,
    fn: Callable[[], T],
) -> T:
    """Serialize execution admission against activation pointer transitions on one bundle."""
    state = require_shared_activation_domain_state(serving_store)
    prior_depth = getattr(_ACTIVATION_CRITICAL_DEPTH, "depth", 0)
    _ACTIVATION_CRITICAL_DEPTH.depth = prior_depth + 1
    try:
        return run_under_activation_domain(state, fn)
    finally:
        _ACTIVATION_CRITICAL_DEPTH.depth = prior_depth


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
    issue_canonical_grant: bool = True,
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

    grant = ServingExecutionAdmissionGrant(
        application_id=application_id,
        application_environment_id=application_environment_id,
        runtime_revision_id=runtime_revision_id,
        serving_unit_ref=serving_unit_ref,
        materialization_artifact_digest=materialization_artifact_digest,
        serving_pointer_revision=serving.serving_pointer_revision,
    )
    if issue_canonical_grant:
        object.__setattr__(grant, "_issuance_nonce", secrets.token_hex(16))
        _register_canonical_grant_issuance(grant)
    return grant


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
    "consume_serving_execution_admission_grant",
    "require_shared_activation_domain_state",
    "run_serving_execution_admission_critical_section",
    "shared_activation_domain_state_from_serving_store",
    "verify_current_serving_execution_admission_eligibility",
]
