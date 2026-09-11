# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Diagnostic extension evidence for external operations (R2)."""

from __future__ import annotations

from dataclasses import dataclass
from intergrax.contracts.diagnostic_extension_evidence import (
    DiagnosticEvidenceContext,
    DiagnosticExtensionEvidence,
)
from intergrax.contracts.external_operations.attempt import ExternalOperationAttemptLifecycle
from intergrax.contracts.external_operations.evidence import ExternalOperationEvidence
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEventType


@dataclass(frozen=True, slots=True)
class ExternalOperationContributorSnapshot:
    operation_attempt_id: str
    provider_id: str
    admission_id: str
    execution_status: ExternalOperationAttemptLifecycle
    correlation_refs: tuple[str, ...]


class ExternalOperationEvidenceContributor:
    """Evidence producer — never mints Problems."""

    contributor_id = "platform.external_operation"
    evidence_namespace = "platform.external_operation"
    priority = 200

    def __init__(
        self,
        *,
        runtime_events: RuntimeEventPersistence,
        snapshots: tuple[ExternalOperationContributorSnapshot, ...] = (),
    ) -> None:
        self._runtime_events = runtime_events
        self._snapshots = snapshots

    def collect(
        self,
        context: DiagnosticEvidenceContext,
    ) -> tuple[DiagnosticExtensionEvidence, ...]:
        evidence: list[DiagnosticExtensionEvidence] = []
        for snapshot in self._snapshots:
            if snapshot.correlation_refs and context.tenant_id != snapshot.correlation_refs[0]:
                continue
            scope = context.to_evidence_scope()
            evidence.append(
                DiagnosticExtensionEvidence.mint(
                    scope=scope,
                    evidence_namespace=self.evidence_namespace,
                    kind=f"{self.evidence_namespace}.attempt",
                    summary=(
                        f"external operation {snapshot.operation_attempt_id} "
                        f"provider={snapshot.provider_id} "
                        f"status={snapshot.execution_status.value}"
                    ),
                    detail_ref=snapshot.admission_id,
                ),
            )
        if context.run_id is None:
            return tuple(evidence)
        for event in self._runtime_events.list_for_run(
            context.run_id,
            tenant_id=context.tenant_id,
        ):
            if event.event_type is not RuntimeEventType.EXTERNAL_OPERATION_FAILED:
                continue
            scope = context.to_evidence_scope()
            evidence.append(
                DiagnosticExtensionEvidence.mint(
                    scope=scope,
                    evidence_namespace=self.evidence_namespace,
                    kind=f"{self.evidence_namespace}.failure",
                    summary="external operation failure runtime evidence",
                    detail_ref=str(event.event_id),
                ),
            )
        return tuple(evidence)


def external_operation_evidence_from_record(
    *,
    evidence: ExternalOperationEvidence,
    admission_id: str,
    execution_status: ExternalOperationAttemptLifecycle,
    correlation_refs: tuple[str, ...],
) -> ExternalOperationContributorSnapshot:
    return ExternalOperationContributorSnapshot(
        operation_attempt_id=evidence.attempt_id,
        provider_id=correlation_refs[-1] if correlation_refs else "unknown",
        admission_id=admission_id,
        execution_status=execution_status,
        correlation_refs=correlation_refs,
    )


__all__ = [
    "ExternalOperationContributorSnapshot",
    "ExternalOperationEvidenceContributor",
    "external_operation_evidence_from_record",
]
