# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision Lifecycle observability via canonical RuntimeEvent spine (DS-OBS-01)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar

from pydantic import model_validator

from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage, DecisionLifecycleState
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.runtime.decision_observability_common import (
    DecisionObservabilityIdentity,
    decision_observability_identity_from_decision_identity,
    validate_decision_version_value,
    validate_emit_context_lineage_for_identity,
    validate_positive_int,
)
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_kind_registry import register_event_kind
from intergrax.runtime.events.payload_registry import register_payload_schema
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.signals import emit_domain_signal
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionDurableFinalizationDisposition,
    DecisionDurableFinalizationResult,
)

DECISION_LIFECYCLE_SIGNAL_PAYLOAD_SCHEMA_ID = "intergrax.decision.lifecycle.signal.v1"

DECISION_LIFECYCLE_STARTED_EVENT_KIND = "intergrax.decision.lifecycle.started"
DECISION_LIFECYCLE_TRANSITIONED_EVENT_KIND = "intergrax.decision.lifecycle.transitioned"
DECISION_LIFECYCLE_RESOLVED_EVENT_KIND = "intergrax.decision.lifecycle.resolved"
DECISION_LIFECYCLE_FINALIZED_EVENT_KIND = "intergrax.decision.lifecycle.finalized"
DECISION_LIFECYCLE_TERMINAL_EVENT_KIND = "intergrax.decision.lifecycle.terminal"

_DECISION_LIFECYCLE_EVENT_KINDS: tuple[str, ...] = (
    DECISION_LIFECYCLE_STARTED_EVENT_KIND,
    DECISION_LIFECYCLE_TRANSITIONED_EVENT_KIND,
    DECISION_LIFECYCLE_RESOLVED_EVENT_KIND,
    DECISION_LIFECYCLE_FINALIZED_EVENT_KIND,
    DECISION_LIFECYCLE_TERMINAL_EVENT_KIND,
)

T = TypeVar("T")


class DecisionLifecycleSignalPhase(StrEnum):
    """Semantic phase carried in Decision Lifecycle domain signals."""

    STARTED = "started"
    TRANSITIONED = "transitioned"
    RESOLVED = "resolved"
    FINALIZED = "finalized"
    TERMINAL = "terminal"


def _validate_lifecycle_stage_value(value: str) -> str:
    if value not in {item.value for item in DecisionLifecycleStage}:
        raise ValueError(
            f"lifecycle stage must be a DecisionLifecycleStage value, got {value!r}",
        )
    return value


def _validate_resolution_outcome_value(value: str) -> str:
    if value not in {item.value for item in DecisionResolution}:
        raise ValueError(
            f"resolution_outcome must be a DecisionResolution value, got {value!r}",
        )
    return value


def _validate_finalization_disposition_value(value: str) -> str:
    if value not in {item.value for item in DecisionDurableFinalizationDisposition}:
        raise ValueError(
            "finalization_disposition must be a DecisionDurableFinalizationDisposition "
            f"value, got {value!r}",
        )
    return value


class DecisionLifecycleSignalPayloadV1(RuntimeEventPayload):
    """Typed redaction-safe payload for Decision Lifecycle domain signals."""

    schema_id = DECISION_LIFECYCLE_SIGNAL_PAYLOAD_SCHEMA_ID

    phase: str
    decision_id: str
    decision_version: int
    tenant_id: str
    scope_namespace: str
    task_id: str
    run_id: str
    attempt_id: str
    execution_id: str | None = None
    transition_index: int
    from_stage: str | None = None
    to_stage: str | None = None
    resolution_outcome: str | None = None
    proposal_branch_id: str | None = None
    finalization_disposition: str | None = None

    @model_validator(mode="after")
    def _validate_semantic_shape(self) -> DecisionLifecycleSignalPayloadV1:
        validate_decision_version_value(self.decision_version)
        validate_positive_int(self.transition_index, "transition_index")
        if self.phase not in {item.value for item in DecisionLifecycleSignalPhase}:
            raise ValueError(
                f"phase must be a DecisionLifecycleSignalPhase value, got {self.phase!r}",
            )
        if self.from_stage is not None:
            _validate_lifecycle_stage_value(self.from_stage)
        if self.to_stage is not None:
            _validate_lifecycle_stage_value(self.to_stage)
        if self.resolution_outcome is not None:
            _validate_resolution_outcome_value(self.resolution_outcome)
        if self.finalization_disposition is not None:
            _validate_finalization_disposition_value(self.finalization_disposition)
        phase = DecisionLifecycleSignalPhase(self.phase)
        if phase is DecisionLifecycleSignalPhase.STARTED:
            if self.from_stage is not None:
                raise ValueError("started phase must not include from_stage")
            if self.to_stage != DecisionLifecycleStage.PROPOSAL.value:
                raise ValueError("started phase requires to_stage proposal")
            if self.transition_index != 0:
                raise ValueError("started phase requires transition_index 0")
            if self.resolution_outcome is not None or self.finalization_disposition is not None:
                raise ValueError("started phase must not include outcome fields")
        if phase is DecisionLifecycleSignalPhase.TRANSITIONED:
            if self.from_stage is None or self.to_stage is None:
                raise ValueError("transitioned phase requires from_stage and to_stage")
            if self.transition_index < 1:
                raise ValueError("transitioned phase requires transition_index >= 1")
            if self.resolution_outcome is not None or self.finalization_disposition is not None:
                raise ValueError("transitioned phase must not include outcome fields")
        if phase is DecisionLifecycleSignalPhase.RESOLVED:
            if self.resolution_outcome is None:
                raise ValueError("resolved phase requires resolution_outcome")
            if self.finalization_disposition is not None:
                raise ValueError("resolved phase must not include finalization_disposition")
        if phase is DecisionLifecycleSignalPhase.FINALIZED:
            if self.finalization_disposition is None:
                raise ValueError("finalized phase requires finalization_disposition")
            if self.resolution_outcome is not None:
                raise ValueError("finalized phase must not include resolution_outcome")
        if phase is DecisionLifecycleSignalPhase.TERMINAL:
            if self.to_stage != DecisionLifecycleStage.TERMINAL.value:
                raise ValueError("terminal phase requires to_stage terminal")
            if self.resolution_outcome is not None or self.finalization_disposition is not None:
                raise ValueError("terminal phase must not include outcome fields")
        return self

    def redact(self) -> DecisionLifecycleSignalPayloadV1:
        """Return a production-safe copy; fields are identifiers and codes only."""
        return self


def register_decision_lifecycle_domain_signals() -> None:
    """Register Decision Lifecycle payload schema and domain event kinds (idempotent)."""
    register_payload_schema(DecisionLifecycleSignalPayloadV1, extension=True)
    for kind in _DECISION_LIFECYCLE_EVENT_KINDS:
        register_event_kind(kind, DECISION_LIFECYCLE_SIGNAL_PAYLOAD_SCHEMA_ID)


def _payload_from_observability_identity(
    *,
    phase: DecisionLifecycleSignalPhase,
    observability_identity: DecisionObservabilityIdentity,
    transition_index: int,
    from_stage: DecisionLifecycleStage | None = None,
    to_stage: DecisionLifecycleStage | None = None,
    resolution_outcome: DecisionResolution | None = None,
    finalization_disposition: DecisionDurableFinalizationDisposition | None = None,
) -> DecisionLifecycleSignalPayloadV1:
    return DecisionLifecycleSignalPayloadV1(
        phase=phase.value,
        decision_id=observability_identity.decision_id,
        decision_version=observability_identity.decision_version,
        tenant_id=observability_identity.tenant_id,
        scope_namespace=observability_identity.scope_namespace,
        task_id=observability_identity.task_id,
        run_id=observability_identity.run_id,
        attempt_id=observability_identity.attempt_id,
        execution_id=observability_identity.execution_id,
        transition_index=transition_index,
        from_stage=from_stage.value if from_stage is not None else None,
        to_stage=to_stage.value if to_stage is not None else None,
        resolution_outcome=(
            resolution_outcome.value if resolution_outcome is not None else None
        ),
        proposal_branch_id=observability_identity.proposal_branch_id,
        finalization_disposition=(
            finalization_disposition.value
            if finalization_disposition is not None
            else None
        ),
    )


def _emit_decision_lifecycle_signal(
    ctx: EmitContext,
    *,
    kind: str,
    payload: DecisionLifecycleSignalPayloadV1,
) -> RuntimeEvent:
    register_decision_lifecycle_domain_signals()
    return emit_domain_signal(ctx, kind=kind, payload=payload)


class DecisionLifecycleObserver(Protocol):
    """Optional observability seam for canonical Decision Lifecycle operations."""

    def lifecycle_started(self, state: DecisionLifecycleState) -> None: ...

    def lifecycle_transitioned(
        self,
        *,
        previous_state: DecisionLifecycleState,
        new_state: DecisionLifecycleState,
    ) -> None: ...

    def lifecycle_resolved(
        self,
        *,
        lifecycle_state: DecisionLifecycleState,
        resolution_outcome: DecisionResolution,
        proposal_branch_id: str | None = None,
    ) -> None: ...

    def lifecycle_finalized(
        self,
        *,
        lifecycle_state: DecisionLifecycleState,
        finalization_disposition: DecisionDurableFinalizationDisposition,
    ) -> None: ...

    def lifecycle_terminal(self, state: DecisionLifecycleState) -> None: ...


@dataclass(frozen=True, slots=True)
class CanonicalRuntimeEventDecisionLifecycleObserver:
    """Emit Decision Lifecycle signals through the canonical RuntimeEvent spine."""

    ctx: EmitContext

    def lifecycle_started(self, state: DecisionLifecycleState) -> None:
        if type(state) is not DecisionLifecycleState:
            raise TypeError("state must be DecisionLifecycleState")
        validate_emit_context_lineage_for_identity(state.identity, self.ctx)
        observability_identity = decision_observability_identity_from_decision_identity(
            state.identity,
        )
        payload = _payload_from_observability_identity(
            phase=DecisionLifecycleSignalPhase.STARTED,
            observability_identity=observability_identity,
            transition_index=0,
            to_stage=DecisionLifecycleStage.PROPOSAL,
        )
        _emit_decision_lifecycle_signal(
            self.ctx,
            kind=DECISION_LIFECYCLE_STARTED_EVENT_KIND,
            payload=payload,
        )

    def lifecycle_transitioned(
        self,
        *,
        previous_state: DecisionLifecycleState,
        new_state: DecisionLifecycleState,
    ) -> None:
        if type(previous_state) is not DecisionLifecycleState:
            raise TypeError("previous_state must be DecisionLifecycleState")
        if type(new_state) is not DecisionLifecycleState:
            raise TypeError("new_state must be DecisionLifecycleState")
        validate_emit_context_lineage_for_identity(new_state.identity, self.ctx)
        observability_identity = decision_observability_identity_from_decision_identity(
            new_state.identity,
        )
        payload = _payload_from_observability_identity(
            phase=DecisionLifecycleSignalPhase.TRANSITIONED,
            observability_identity=observability_identity,
            transition_index=new_state.transition_index,
            from_stage=previous_state.stage,
            to_stage=new_state.stage,
        )
        _emit_decision_lifecycle_signal(
            self.ctx,
            kind=DECISION_LIFECYCLE_TRANSITIONED_EVENT_KIND,
            payload=payload,
        )

    def lifecycle_resolved(
        self,
        *,
        lifecycle_state: DecisionLifecycleState,
        resolution_outcome: DecisionResolution,
        proposal_branch_id: str | None = None,
    ) -> None:
        if type(lifecycle_state) is not DecisionLifecycleState:
            raise TypeError("lifecycle_state must be DecisionLifecycleState")
        if type(resolution_outcome) is not DecisionResolution:
            raise TypeError("resolution_outcome must be DecisionResolution")
        validate_emit_context_lineage_for_identity(lifecycle_state.identity, self.ctx)
        observability_identity = decision_observability_identity_from_decision_identity(
            lifecycle_state.identity,
            proposal_branch_id=proposal_branch_id,
        )
        payload = _payload_from_observability_identity(
            phase=DecisionLifecycleSignalPhase.RESOLVED,
            observability_identity=observability_identity,
            transition_index=lifecycle_state.transition_index,
            to_stage=lifecycle_state.stage,
            resolution_outcome=resolution_outcome,
        )
        _emit_decision_lifecycle_signal(
            self.ctx,
            kind=DECISION_LIFECYCLE_RESOLVED_EVENT_KIND,
            payload=payload,
        )

    def lifecycle_finalized(
        self,
        *,
        lifecycle_state: DecisionLifecycleState,
        finalization_disposition: DecisionDurableFinalizationDisposition,
    ) -> None:
        if type(lifecycle_state) is not DecisionLifecycleState:
            raise TypeError("lifecycle_state must be DecisionLifecycleState")
        if type(finalization_disposition) is not DecisionDurableFinalizationDisposition:
            raise TypeError(
                "finalization_disposition must be DecisionDurableFinalizationDisposition",
            )
        validate_emit_context_lineage_for_identity(lifecycle_state.identity, self.ctx)
        observability_identity = decision_observability_identity_from_decision_identity(
            lifecycle_state.identity,
        )
        payload = _payload_from_observability_identity(
            phase=DecisionLifecycleSignalPhase.FINALIZED,
            observability_identity=observability_identity,
            transition_index=lifecycle_state.transition_index,
            to_stage=lifecycle_state.stage,
            finalization_disposition=finalization_disposition,
        )
        _emit_decision_lifecycle_signal(
            self.ctx,
            kind=DECISION_LIFECYCLE_FINALIZED_EVENT_KIND,
            payload=payload,
        )

    def lifecycle_terminal(self, state: DecisionLifecycleState) -> None:
        if type(state) is not DecisionLifecycleState:
            raise TypeError("state must be DecisionLifecycleState")
        if state.stage is not DecisionLifecycleStage.TERMINAL:
            raise ValueError("lifecycle_terminal requires current stage terminal")
        validate_emit_context_lineage_for_identity(state.identity, self.ctx)
        observability_identity = decision_observability_identity_from_decision_identity(
            state.identity,
        )
        payload = _payload_from_observability_identity(
            phase=DecisionLifecycleSignalPhase.TERMINAL,
            observability_identity=observability_identity,
            transition_index=state.transition_index,
            to_stage=DecisionLifecycleStage.TERMINAL,
        )
        _emit_decision_lifecycle_signal(
            self.ctx,
            kind=DECISION_LIFECYCLE_TERMINAL_EVENT_KIND,
            payload=payload,
        )


def observe_decision_resolution(
    observer: DecisionLifecycleObserver,
    *,
    lifecycle_state: DecisionLifecycleState,
    outcome: AuthoritativeAcceptedDecision[T] | AuthoritativeResolutionRecord,
    proposal_branch_id: str | None = None,
) -> None:
    """Observe one exact-version resolution without moving lifecycle authority."""
    if type(outcome) is AuthoritativeAcceptedDecision:
        branch_id = proposal_branch_id
        if branch_id is None:
            branch_id = str(outcome.lineage.current.branch_id)
        observer.lifecycle_resolved(
            lifecycle_state=lifecycle_state,
            resolution_outcome=DecisionResolution.ACCEPTED,
            proposal_branch_id=branch_id,
        )
        return
    if type(outcome) is AuthoritativeResolutionRecord:
        observer.lifecycle_resolved(
            lifecycle_state=lifecycle_state,
            resolution_outcome=outcome.resolution,
            proposal_branch_id=proposal_branch_id,
        )
        return
    raise TypeError(
        "outcome must be AuthoritativeAcceptedDecision or AuthoritativeResolutionRecord",
    )


def observe_durable_decision_finalization(
    observer: DecisionLifecycleObserver,
    *,
    lifecycle_state: DecisionLifecycleState,
    result: DecisionDurableFinalizationResult[T],
) -> None:
    """Observe durable finalization only after persistence returns one result."""
    if type(result) is not DecisionDurableFinalizationResult:
        raise TypeError("result must be DecisionDurableFinalizationResult")
    observer.lifecycle_finalized(
        lifecycle_state=lifecycle_state,
        finalization_disposition=result.disposition,
    )
