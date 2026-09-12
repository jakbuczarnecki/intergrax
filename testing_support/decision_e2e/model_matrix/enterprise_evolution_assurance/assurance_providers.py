# © Artur Czarnecki. All rights reserved.

"""Pluggable top-level evolution assurance providers (DS-E2E-15J-L18)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.audit_providers import (
    default_evolution_assurance_audit_provider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.compliance_validator_providers import (
    default_evolution_compliance_validator_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.contracts import (
    EvolutionAssuranceContext,
    EvolutionAssuranceResult,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.engine import (
    run_evolution_assurance_assessment,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.evidence_validator_providers import (
    default_evolution_evidence_validator_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.protocol import (
    EvolutionAssuranceAuditProvider,
    EvolutionComplianceValidatorProvider,
    EvolutionEvidenceValidatorProvider,
    EvolutionQualityValidatorProvider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.quality_validator_providers import (
    default_evolution_quality_validator_providers,
)

_DEFAULT_ASSURANCE_PROVIDER_ID = "default_enterprise_evolution_assurance"
_DEFAULT_ASSURANCE_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DefaultEnterpriseEvolutionAssuranceProvider:
    quality_validator_providers: tuple[EvolutionQualityValidatorProvider, ...]
    compliance_validator_providers: tuple[EvolutionComplianceValidatorProvider, ...]
    evidence_validator_providers: tuple[EvolutionEvidenceValidatorProvider, ...]
    audit_provider: EvolutionAssuranceAuditProvider

    @property
    def provider_id(self) -> str:
        return _DEFAULT_ASSURANCE_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _DEFAULT_ASSURANCE_PROVIDER_VERSION

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
            assurance_providers=(),
            assessed_at=assessed_at or datetime.now(tz=UTC),
        )


def default_enterprise_evolution_assurance_provider() -> (
    DefaultEnterpriseEvolutionAssuranceProvider
):
    return DefaultEnterpriseEvolutionAssuranceProvider(
        quality_validator_providers=default_evolution_quality_validator_providers(),
        compliance_validator_providers=default_evolution_compliance_validator_providers(),
        evidence_validator_providers=default_evolution_evidence_validator_providers(),
        audit_provider=default_evolution_assurance_audit_provider(),
    )


__all__ = [
    "DefaultEnterpriseEvolutionAssuranceProvider",
    "default_enterprise_evolution_assurance_provider",
]
