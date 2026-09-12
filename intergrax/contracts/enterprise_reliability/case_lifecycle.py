# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ERL reliability case lifecycle — platform-neutral states and explicit transitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_RELIABILITY_CASE_LIFECYCLE_V1: Final = "reliability_case_lifecycle.v1"


class ReliabilityCaseLifecycleState(StrEnum):
    """Where a reliability case sits in the ERL journey — not domain business states."""

    UNKNOWN_DETECTED = "unknown_detected"
    RECONCILIATION_RUNNING = "reconciliation_running"
    EVIDENCE_AVAILABLE = "evidence_available"
    RESOLUTION_PENDING = "resolution_pending"
    COMPENSATION_PENDING = "compensation_pending"
    RECOVERY_PENDING = "recovery_pending"
    GOVERNANCE_PENDING = "governance_pending"
    HANDOFF_READY = "handoff_ready"
    CLOSED = "closed"


class ReliabilityCaseLifecycleTransitionError(RuntimeError):
    """Illegal reliability case lifecycle transition."""


class ReliabilityCaseLifecycleContextError(ValueError):
    """Required lifecycle references missing for the target state."""


class ReliabilityCaseLifecycleRefs(BaseModel):
    """Opaque references to capability artifacts — no evidence/resolution/compensation payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str = Field(min_length=1, max_length=256)
    uncertainty_state_ref: str | None = Field(default=None, max_length=512)
    evidence_ref: str | None = Field(default=None, max_length=512)
    resolution_context_ref: str | None = Field(default=None, max_length=512)
    compensation_context_ref: str | None = Field(default=None, max_length=512)
    recovery_context_ref: str | None = Field(default=None, max_length=512)
    governance_result_ref: str | None = Field(default=None, max_length=512)
    handoff_request_ref: str | None = Field(default=None, max_length=512)
    execution_ref: str | None = Field(default=None, max_length=512)


class ReliabilityCaseLifecycleRecord(BaseModel):
    """Immutable view of one reliability case lifecycle position."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=256)
    correlation_id: str = Field(min_length=1, max_length=256)
    lifecycle_state: ReliabilityCaseLifecycleState
    refs: ReliabilityCaseLifecycleRefs


class ReliabilityCaseLifecycleTransitionRequest(BaseModel):
    """Request to move a reliability case to the next explicit lifecycle state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: ReliabilityCaseLifecycleRecord
    target_state: ReliabilityCaseLifecycleState
    refs: ReliabilityCaseLifecycleRefs


class ReliabilityCaseLifecycleTransitionResult(BaseModel):
    """Outcome of a validated lifecycle transition — state ownership only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    previous_state: ReliabilityCaseLifecycleState
    record: ReliabilityCaseLifecycleRecord


_ALLOWED_TRANSITIONS: dict[
    ReliabilityCaseLifecycleState, frozenset[ReliabilityCaseLifecycleState]
] = {
    ReliabilityCaseLifecycleState.UNKNOWN_DETECTED: frozenset(
        {ReliabilityCaseLifecycleState.RECONCILIATION_RUNNING}
    ),
    ReliabilityCaseLifecycleState.RECONCILIATION_RUNNING: frozenset(
        {ReliabilityCaseLifecycleState.EVIDENCE_AVAILABLE}
    ),
    ReliabilityCaseLifecycleState.EVIDENCE_AVAILABLE: frozenset(
        {ReliabilityCaseLifecycleState.RESOLUTION_PENDING}
    ),
    ReliabilityCaseLifecycleState.RESOLUTION_PENDING: frozenset(
        {
            ReliabilityCaseLifecycleState.COMPENSATION_PENDING,
            ReliabilityCaseLifecycleState.RECOVERY_PENDING,
        }
    ),
    ReliabilityCaseLifecycleState.COMPENSATION_PENDING: frozenset(
        {ReliabilityCaseLifecycleState.RECOVERY_PENDING}
    ),
    ReliabilityCaseLifecycleState.RECOVERY_PENDING: frozenset(
        {ReliabilityCaseLifecycleState.GOVERNANCE_PENDING}
    ),
    ReliabilityCaseLifecycleState.GOVERNANCE_PENDING: frozenset(
        {ReliabilityCaseLifecycleState.HANDOFF_READY}
    ),
    ReliabilityCaseLifecycleState.HANDOFF_READY: frozenset(
        {ReliabilityCaseLifecycleState.CLOSED}
    ),
    ReliabilityCaseLifecycleState.CLOSED: frozenset(),
}

_REQUIRED_REF_FIELDS_AT_STATE: dict[
    ReliabilityCaseLifecycleState, frozenset[str]
] = {
    ReliabilityCaseLifecycleState.UNKNOWN_DETECTED: frozenset(
        {"contract_id", "uncertainty_state_ref"}
    ),
    ReliabilityCaseLifecycleState.RECONCILIATION_RUNNING: frozenset(
        {"contract_id", "uncertainty_state_ref"}
    ),
    ReliabilityCaseLifecycleState.EVIDENCE_AVAILABLE: frozenset(
        {"contract_id", "uncertainty_state_ref", "evidence_ref"}
    ),
    ReliabilityCaseLifecycleState.RESOLUTION_PENDING: frozenset(
        {
            "contract_id",
            "uncertainty_state_ref",
            "evidence_ref",
            "resolution_context_ref",
        }
    ),
    ReliabilityCaseLifecycleState.COMPENSATION_PENDING: frozenset(
        {
            "contract_id",
            "uncertainty_state_ref",
            "evidence_ref",
            "resolution_context_ref",
            "compensation_context_ref",
        }
    ),
    ReliabilityCaseLifecycleState.RECOVERY_PENDING: frozenset(
        {
            "contract_id",
            "uncertainty_state_ref",
            "evidence_ref",
            "resolution_context_ref",
            "recovery_context_ref",
        }
    ),
    ReliabilityCaseLifecycleState.GOVERNANCE_PENDING: frozenset(
        {
            "contract_id",
            "uncertainty_state_ref",
            "evidence_ref",
            "resolution_context_ref",
            "recovery_context_ref",
        }
    ),
    ReliabilityCaseLifecycleState.HANDOFF_READY: frozenset(
        {
            "contract_id",
            "uncertainty_state_ref",
            "evidence_ref",
            "resolution_context_ref",
            "recovery_context_ref",
            "governance_result_ref",
        }
    ),
    ReliabilityCaseLifecycleState.CLOSED: frozenset(
        {
            "contract_id",
            "uncertainty_state_ref",
            "evidence_ref",
            "resolution_context_ref",
            "recovery_context_ref",
            "governance_result_ref",
            "handoff_request_ref",
        }
    ),
}


def assert_reliability_case_lifecycle_transition(
    current: ReliabilityCaseLifecycleState,
    target: ReliabilityCaseLifecycleState,
) -> None:
    """Fail closed on illegal lifecycle moves."""
    allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise ReliabilityCaseLifecycleTransitionError(
            f"illegal reliability case transition: {current.value} -> {target.value}",
        )


def _lifecycle_ref_value(
    refs: ReliabilityCaseLifecycleRefs,
    field_name: str,
) -> str | None:
    if field_name == "contract_id":
        return refs.contract_id
    if field_name == "uncertainty_state_ref":
        return refs.uncertainty_state_ref
    if field_name == "evidence_ref":
        return refs.evidence_ref
    if field_name == "resolution_context_ref":
        return refs.resolution_context_ref
    if field_name == "compensation_context_ref":
        return refs.compensation_context_ref
    if field_name == "recovery_context_ref":
        return refs.recovery_context_ref
    if field_name == "governance_result_ref":
        return refs.governance_result_ref
    if field_name == "handoff_request_ref":
        return refs.handoff_request_ref
    if field_name == "execution_ref":
        return refs.execution_ref
    return None


def assert_reliability_case_lifecycle_refs_for_state(
    state: ReliabilityCaseLifecycleState,
    refs: ReliabilityCaseLifecycleRefs,
) -> None:
    """Fail closed when required references for a lifecycle state are absent."""
    required = _REQUIRED_REF_FIELDS_AT_STATE.get(state, frozenset())
    for field_name in required:
        value = _lifecycle_ref_value(refs, field_name)
        if value is None or not value.strip():
            raise ReliabilityCaseLifecycleContextError(
                f"missing lifecycle reference '{field_name}' for state {state.value}",
            )


def initial_reliability_case_lifecycle(
    *,
    case_id: str,
    correlation_id: str,
    contract_id: str,
    uncertainty_state_ref: str,
) -> ReliabilityCaseLifecycleRecord:
    """Create a reliability case at UNKNOWN detection."""
    refs = ReliabilityCaseLifecycleRefs(
        contract_id=contract_id,
        uncertainty_state_ref=uncertainty_state_ref,
    )
    assert_reliability_case_lifecycle_refs_for_state(
        ReliabilityCaseLifecycleState.UNKNOWN_DETECTED,
        refs,
    )
    return ReliabilityCaseLifecycleRecord(
        case_id=case_id,
        correlation_id=correlation_id,
        lifecycle_state=ReliabilityCaseLifecycleState.UNKNOWN_DETECTED,
        refs=refs,
    )


__all__ = [
    "ReliabilityCaseLifecycleContextError",
    "ReliabilityCaseLifecycleRecord",
    "ReliabilityCaseLifecycleRefs",
    "ReliabilityCaseLifecycleState",
    "ReliabilityCaseLifecycleTransitionError",
    "ReliabilityCaseLifecycleTransitionRequest",
    "ReliabilityCaseLifecycleTransitionResult",
    "SCHEMA_RELIABILITY_CASE_LIFECYCLE_V1",
    "assert_reliability_case_lifecycle_refs_for_state",
    "assert_reliability_case_lifecycle_transition",
    "initial_reliability_case_lifecycle",
]
