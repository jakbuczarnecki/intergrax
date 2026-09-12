# © Artur Czarnecki. All rights reserved.

"""Pluggable evolution evidence validators (DS-E2E-15J-L18)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.contracts import (
    EvolutionAssuranceContext,
    EvolutionAssuranceFinding,
    EvolutionAssuranceFindingSeverity,
    EvolutionAssuranceLifecycleStage,
)

_SOURCE_LINEAGE_ID = "source_lineage_evidence_validator"
_SOURCE_LINEAGE_VERSION = "1"
_AUDIT_TRACE_ID = "audit_trace_evidence_validator"
_AUDIT_TRACE_VERSION = "1"


@dataclass(frozen=True, slots=True)
class SourceLineageEvidenceValidator:
    @property
    def provider_id(self) -> str:
        return _SOURCE_LINEAGE_ID

    @property
    def provider_version(self) -> str:
        return _SOURCE_LINEAGE_VERSION

    def validate(
        self,
        context: EvolutionAssuranceContext,
    ) -> tuple[EvolutionAssuranceFinding, ...]:
        findings: list[EvolutionAssuranceFinding] = []
        for index, execution in enumerate(context.execution_results):
            source = execution.source_reference
            if not source.controlled_evolution_record_id:
                findings.append(
                    EvolutionAssuranceFinding(
                        finding_id=f"evidence-missing-evolution-record-{index}",
                        severity=EvolutionAssuranceFindingSeverity.REVIEW,
                        finding_code="missing_controlled_evolution_record",
                        summary=(
                            f"Adaptation {execution.adaptation_id} lacks "
                            "controlled evolution record reference."
                        ),
                        lifecycle_stage=EvolutionAssuranceLifecycleStage.ADAPTATION,
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                    )
                )
            if not source.proposal_id:
                findings.append(
                    EvolutionAssuranceFinding(
                        finding_id=f"evidence-missing-proposal-{index}",
                        severity=EvolutionAssuranceFindingSeverity.REVIEW,
                        finding_code="missing_proposal_reference",
                        summary=(
                            f"Adaptation {execution.adaptation_id} lacks proposal reference."
                        ),
                        lifecycle_stage=EvolutionAssuranceLifecycleStage.ADAPTATION,
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                    )
                )
        governance = context.governance_reference
        if governance is not None and not governance.governance_decision_reference:
            findings.append(
                EvolutionAssuranceFinding(
                    finding_id="evidence-empty-governance-ref",
                    severity=EvolutionAssuranceFindingSeverity.REVIEW,
                    finding_code="empty_governance_decision_reference",
                    summary="Governance reference is present but decision reference is empty.",
                    lifecycle_stage=EvolutionAssuranceLifecycleStage.GOVERNANCE,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        return tuple(findings)


@dataclass(frozen=True, slots=True)
class AuditTraceEvidenceValidator:
    @property
    def provider_id(self) -> str:
        return _AUDIT_TRACE_ID

    @property
    def provider_version(self) -> str:
        return _AUDIT_TRACE_VERSION

    def validate(
        self,
        context: EvolutionAssuranceContext,
    ) -> tuple[EvolutionAssuranceFinding, ...]:
        findings: list[EvolutionAssuranceFinding] = []
        for record in context.operation_records:
            if not record.record_id:
                findings.append(
                    EvolutionAssuranceFinding(
                        finding_id="evidence-operation-missing-id",
                        severity=EvolutionAssuranceFindingSeverity.WARNING,
                        finding_code="operation_record_missing_id",
                        summary="An operation record is missing record_id.",
                        lifecycle_stage=EvolutionAssuranceLifecycleStage.OPERATIONS,
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                    )
                )
        for index, execution in enumerate(context.execution_results):
            audit = execution.audit_metadata
            if audit.provider_id != execution.provider_id:
                findings.append(
                    EvolutionAssuranceFinding(
                        finding_id=f"evidence-audit-provider-mismatch-{index}",
                        severity=EvolutionAssuranceFindingSeverity.REVIEW,
                        finding_code="audit_provider_mismatch",
                        summary=(
                            f"Adaptation {execution.adaptation_id} audit provider "
                            "does not match execution provider."
                        ),
                        lifecycle_stage=EvolutionAssuranceLifecycleStage.ADAPTATION,
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                        evidence_ref=f"adaptation:{execution.adaptation_id}",
                    )
                )
        return tuple(findings)


def default_evolution_evidence_validator_providers() -> tuple[
    SourceLineageEvidenceValidator,
    AuditTraceEvidenceValidator,
]:
    return (
        SourceLineageEvidenceValidator(),
        AuditTraceEvidenceValidator(),
    )


__all__ = [
    "AuditTraceEvidenceValidator",
    "SourceLineageEvidenceValidator",
    "default_evolution_evidence_validator_providers",
]
