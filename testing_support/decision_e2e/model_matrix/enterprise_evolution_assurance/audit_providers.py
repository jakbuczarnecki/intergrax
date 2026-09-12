# © Artur Czarnecki. All rights reserved.

"""Evolution assurance audit metadata providers (DS-E2E-15J-L18)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.contracts import (
    ENTERPRISE_EVOLUTION_ASSURANCE_TASK_ID,
    ENTERPRISE_EVOLUTION_ASSURANCE_VERSION,
    EvolutionAssuranceAuditMetadata,
    EvolutionAssuranceContext,
)

_STANDARD_ASSURANCE_AUDIT_PROVIDER_ID = "standard_evolution_assurance_audit"
_STANDARD_ASSURANCE_AUDIT_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class StandardEvolutionAssuranceAuditProvider:
    @property
    def provider_id(self) -> str:
        return _STANDARD_ASSURANCE_AUDIT_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _STANDARD_ASSURANCE_AUDIT_PROVIDER_VERSION

    def build_audit(
        self,
        context: EvolutionAssuranceContext,
        *,
        quality_validator_ids: tuple[str, ...],
        quality_validator_versions: tuple[str, ...],
        compliance_validator_ids: tuple[str, ...],
        compliance_validator_versions: tuple[str, ...],
        evidence_validator_ids: tuple[str, ...],
        evidence_validator_versions: tuple[str, ...],
        assurance_provider_ids: tuple[str, ...],
        assurance_provider_versions: tuple[str, ...],
        evidence_refs: tuple[str, ...],
        finding_ids: tuple[str, ...],
        assessed_at: datetime,
        assessment_scope_summary: str,
    ) -> EvolutionAssuranceAuditMetadata:
        return EvolutionAssuranceAuditMetadata(
            assurance_task_id=ENTERPRISE_EVOLUTION_ASSURANCE_TASK_ID,
            assurance_layer_version=ENTERPRISE_EVOLUTION_ASSURANCE_VERSION,
            scope_id=context.scope_id,
            scope_version=context.version,
            quality_validator_ids=quality_validator_ids,
            quality_validator_versions=quality_validator_versions,
            compliance_validator_ids=compliance_validator_ids,
            compliance_validator_versions=compliance_validator_versions,
            evidence_validator_ids=evidence_validator_ids,
            evidence_validator_versions=evidence_validator_versions,
            assurance_provider_ids=assurance_provider_ids,
            assurance_provider_versions=assurance_provider_versions,
            process_reference=context.process_reference,
            evidence_refs=evidence_refs,
            finding_ids=finding_ids,
            assessment_scope_summary=assessment_scope_summary,
            assessed_at=assessed_at,
        )


def default_evolution_assurance_audit_provider() -> (
    StandardEvolutionAssuranceAuditProvider
):
    return StandardEvolutionAssuranceAuditProvider()


__all__ = [
    "StandardEvolutionAssuranceAuditProvider",
    "default_evolution_assurance_audit_provider",
]
