# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Required audit evidence admission semantics (BG-EXEC-3)."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import TypeVar

from intergrax.contracts.execution_identity import EventId, mint_event_id
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)

_T = TypeVar("_T")


class EvidenceDurabilityClass(StrEnum):
    """Platform-owned durability semantics for observability facts."""

    OPTIONAL_OBSERVABILITY = "optional_observability"
    REQUIRED_AUDIT_EVIDENCE = "required_audit_evidence"


REQUIRED_BACKGROUND_CAUSAL_RELATIONS: frozenset[CausalRelationKind] = frozenset(
    {CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION}
)


class RequiredAuditEvidencePersistenceError(RuntimeError):
    """
    Required causal audit evidence could not be persisted.

    Classified as ``FailureClass.DEPENDENCY_ERROR`` / ``RuntimeErrorCode.DEPENDENCY_ERROR``.
    """


def build_transport_triggered_execution_evidence(
    transport_ref: BackgroundTransportExecutionRef,
    execution_identity: BackgroundExecutionIdentity,
    *,
    evidence_id: EventId | None = None,
) -> PlatformCausalEvidence:
    """
    Build the required transport→execution causal fact for one attempt.

    Mint ``evidence_id`` once per attempt and reuse the returned object for
    persistence retries on the same attempt. A worker retry with a new
    ``AttemptId`` must build a new evidence object.
    """
    return PlatformCausalEvidence(
        evidence_id=evidence_id or mint_event_id(),
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=execution_identity.tenant_id,
        source=MessageBusTaskRef(
            provider=transport_ref.provider,
            task_id=transport_ref.transport_task_id,
            tenant_id=execution_identity.tenant_id,
        ),
        target=RuntimeExecutionRef(
            task_id=execution_identity.task_id,
            run_id=execution_identity.run_id,
            attempt_id=execution_identity.attempt_id,
            tenant_id=execution_identity.tenant_id,
        ),
    )


def persist_required_audit_evidence(
    persistence: CausalEvidencePersistence,
    evidence: PlatformCausalEvidence,
) -> PlatformCausalEvidence:
    """Persist required audit evidence; failures propagate (fail-closed)."""
    if evidence.relation_kind not in REQUIRED_BACKGROUND_CAUSAL_RELATIONS:
        raise ValueError(
            f"relation_kind {evidence.relation_kind!r} is not required audit evidence"
        )
    try:
        return persistence.append(evidence)
    except RequiredAuditEvidencePersistenceError:
        raise
    except Exception as exc:
        raise RequiredAuditEvidencePersistenceError(
            "required audit evidence persistence failed"
        ) from exc


def admit_background_execution_handler(
    *,
    transport_ref: BackgroundTransportExecutionRef,
    execution_identity: BackgroundExecutionIdentity,
    causal_evidence_persistence: CausalEvidencePersistence,
    handler: Callable[[], _T],
    evidence: PlatformCausalEvidence | None = None,
) -> _T:
    """
    Required audit admission gate: persist causal evidence, then invoke handler.

    Ordering invariant (DIAG-1I writer will call this before ``execute_logical_task``):

    ``transport task received → identity established → required evidence persisted
    → handler``
    """
    resolved_evidence = evidence or build_transport_triggered_execution_evidence(
        transport_ref,
        execution_identity,
    )
    persist_required_audit_evidence(causal_evidence_persistence, resolved_evidence)
    return handler()
