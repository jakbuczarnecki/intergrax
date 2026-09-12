# © Artur Czarnecki. All rights reserved.

"""Enterprise evolution assurance orchestration via injected plugins (DS-E2E-15J-L18)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.contracts import (
    ENTERPRISE_EVOLUTION_ASSURANCE_TASK_ID,
    EvolutionAssuranceContext,
    EvolutionAssuranceFinding,
    EvolutionAssuranceFindingSeverity,
    EvolutionAssuranceResult,
    EvolutionAssuranceStatus,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.protocol import (
    EnterpriseEvolutionAssuranceProvider,
    EvolutionAssuranceAuditProvider,
    EvolutionComplianceValidatorProvider,
    EvolutionEvidenceValidatorProvider,
    EvolutionQualityValidatorProvider,
)


def _validator_ids(
    providers: tuple[
        EvolutionQualityValidatorProvider
        | EvolutionComplianceValidatorProvider
        | EvolutionEvidenceValidatorProvider,
        ...,
    ],
) -> tuple[str, ...]:
    return tuple(item.provider_id for item in providers)


def _validator_versions(
    providers: tuple[
        EvolutionQualityValidatorProvider
        | EvolutionComplianceValidatorProvider
        | EvolutionEvidenceValidatorProvider,
        ...,
    ],
) -> tuple[str, ...]:
    return tuple(item.provider_version for item in providers)


def _assurance_provider_ids(
    providers: tuple[EnterpriseEvolutionAssuranceProvider, ...],
) -> tuple[str, ...]:
    return tuple(item.provider_id for item in providers)


def _assurance_provider_versions(
    providers: tuple[EnterpriseEvolutionAssuranceProvider, ...],
) -> tuple[str, ...]:
    return tuple(item.provider_version for item in providers)


def _aggregate_status(
    findings: tuple[EvolutionAssuranceFinding, ...],
) -> EvolutionAssuranceStatus:
    if any(
        item.severity is EvolutionAssuranceFindingSeverity.FAILED for item in findings
    ):
        return EvolutionAssuranceStatus.FAILED
    if any(
        item.severity is EvolutionAssuranceFindingSeverity.REVIEW for item in findings
    ):
        return EvolutionAssuranceStatus.REVIEW_REQUIRED
    if any(
        item.severity is EvolutionAssuranceFindingSeverity.WARNING for item in findings
    ):
        return EvolutionAssuranceStatus.WARNING
    return EvolutionAssuranceStatus.PASSED


def _collect_evidence_refs(context: EvolutionAssuranceContext) -> tuple[str, ...]:
    refs: list[str] = []
    if context.process_reference is not None:
        refs.append(
            f"process:{context.process_reference.process_id}:"
            f"v{context.process_reference.version}"
        )
    governance = context.governance_reference
    if governance is not None:
        refs.append(f"governance:{governance.governance_decision_reference}")
    framework = context.governance_framework_result
    if framework is not None:
        refs.append(f"framework:{framework.framework_task_id}")
    for execution in context.execution_results:
        if execution.applied_change_reference is not None:
            refs.append(f"adaptation:{execution.applied_change_reference}")
        refs.append(f"adaptation-audit:{execution.audit_metadata.adaptation_id}")
    for record in context.operation_records:
        refs.append(f"operation:{record.record_id}")
    return tuple(dict.fromkeys(refs))


def run_evolution_assurance_assessment(
    context: EvolutionAssuranceContext,
    *,
    quality_validator_providers: tuple[EvolutionQualityValidatorProvider, ...],
    compliance_validator_providers: tuple[EvolutionComplianceValidatorProvider, ...],
    evidence_validator_providers: tuple[EvolutionEvidenceValidatorProvider, ...],
    audit_provider: EvolutionAssuranceAuditProvider,
    assurance_providers: tuple[EnterpriseEvolutionAssuranceProvider, ...] = (),
    assessed_at: datetime | None = None,
) -> EvolutionAssuranceResult:
    stamp = assessed_at or datetime.now(tz=UTC)
    findings: list[EvolutionAssuranceFinding] = []
    for validator in quality_validator_providers:
        findings.extend(validator.validate(context))
    for validator in compliance_validator_providers:
        findings.extend(validator.validate(context))
    for validator in evidence_validator_providers:
        findings.extend(validator.validate(context))

    for provider in assurance_providers:
        supplemental = provider.assess(context, assessed_at=stamp)
        findings.extend(supplemental.findings)

    all_findings = tuple(findings)
    status = _aggregate_status(all_findings)
    evidence_refs = _collect_evidence_refs(context)
    scope_summary = (
        f"Evolution assurance assessment for scope {context.scope_id} "
        f"v{context.version} recorded {len(all_findings)} finding(s) "
        f"with status {status.value}."
    )
    audit = audit_provider.build_audit(
        context,
        quality_validator_ids=_validator_ids(quality_validator_providers),
        quality_validator_versions=_validator_versions(quality_validator_providers),
        compliance_validator_ids=_validator_ids(compliance_validator_providers),
        compliance_validator_versions=_validator_versions(
            compliance_validator_providers
        ),
        evidence_validator_ids=_validator_ids(evidence_validator_providers),
        evidence_validator_versions=_validator_versions(evidence_validator_providers),
        assurance_provider_ids=_assurance_provider_ids(assurance_providers),
        assurance_provider_versions=_assurance_provider_versions(assurance_providers),
        evidence_refs=evidence_refs,
        finding_ids=tuple(item.finding_id for item in all_findings),
        assessed_at=stamp,
        assessment_scope_summary=scope_summary,
    )
    return EvolutionAssuranceResult(
        assurance_task_id=ENTERPRISE_EVOLUTION_ASSURANCE_TASK_ID,
        status=status,
        findings=all_findings,
        evidence_refs=evidence_refs,
        audit=audit,
    )


@dataclass(frozen=True, slots=True)
class EnterpriseEvolutionAssuranceEngine:
    quality_validator_providers: tuple[EvolutionQualityValidatorProvider, ...]
    compliance_validator_providers: tuple[EvolutionComplianceValidatorProvider, ...]
    evidence_validator_providers: tuple[EvolutionEvidenceValidatorProvider, ...]
    assurance_providers: tuple[EnterpriseEvolutionAssuranceProvider, ...]
    audit_provider: EvolutionAssuranceAuditProvider

    def assess(
        self,
        context: EvolutionAssuranceContext,
        *,
        assessed_at: datetime | None = None,
    ) -> EvolutionAssuranceResult:
        return run_evolution_assurance_assessment(
            context,
            quality_validator_providers=self.quality_validator_providers,
            compliance_validator_providers=self.compliance_validator_providers,
            evidence_validator_providers=self.evidence_validator_providers,
            audit_provider=self.audit_provider,
            assurance_providers=self.assurance_providers,
            assessed_at=assessed_at,
        )


__all__ = [
    "EnterpriseEvolutionAssuranceEngine",
    "run_evolution_assurance_assessment",
]
