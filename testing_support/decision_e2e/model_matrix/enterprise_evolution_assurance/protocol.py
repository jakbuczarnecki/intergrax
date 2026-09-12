# © Artur Czarnecki. All rights reserved.

"""Pluggable enterprise evolution assurance contracts (DS-E2E-15J-L18)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.contracts import (
    EvolutionAssuranceAuditMetadata,
    EvolutionAssuranceContext,
    EvolutionAssuranceFinding,
    EvolutionAssuranceResult,
)


class EnterpriseEvolutionAssuranceProvider(Protocol):
    """Top-level assurance plugin — validates evolution process quality, never mutates."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def assess(
        self,
        context: EvolutionAssuranceContext,
        *,
        assessed_at: datetime | None = None,
    ) -> EvolutionAssuranceResult: ...


class EvolutionQualityValidatorProvider(Protocol):
    """Pluggable quality check — new validation equals new plugin."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def validate(
        self,
        context: EvolutionAssuranceContext,
    ) -> tuple[EvolutionAssuranceFinding, ...]: ...


class EvolutionComplianceValidatorProvider(Protocol):
    """Process contract compliance — does not execute compliance actions."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def validate(
        self,
        context: EvolutionAssuranceContext,
    ) -> tuple[EvolutionAssuranceFinding, ...]: ...


class EvolutionEvidenceValidatorProvider(Protocol):
    """Evidence lineage and audit trace checks."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def validate(
        self,
        context: EvolutionAssuranceContext,
    ) -> tuple[EvolutionAssuranceFinding, ...]: ...


class EvolutionAssuranceAuditProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

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
    ) -> EvolutionAssuranceAuditMetadata: ...


__all__ = [
    "EnterpriseEvolutionAssuranceProvider",
    "EvolutionAssuranceAuditProvider",
    "EvolutionComplianceValidatorProvider",
    "EvolutionEvidenceValidatorProvider",
    "EvolutionQualityValidatorProvider",
]
