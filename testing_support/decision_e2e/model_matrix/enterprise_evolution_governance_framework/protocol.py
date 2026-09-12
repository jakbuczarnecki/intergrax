# © Artur Czarnecki. All rights reserved.

"""Pluggable enterprise evolution governance framework contracts (DS-E2E-15J-L17)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.contracts import (
    EvolutionGovernanceFrameworkAuditMetadata,
    EvolutionGovernanceFrameworkContext,
    EvolutionGovernanceFrameworkResult,
    EvolutionGovernanceIssue,
    EvolutionGovernanceLifecycleStage,
)


class EnterpriseEvolutionGovernanceProvider(Protocol):
    """Coordinates governance metadata evaluation — never approves or mutates evolution."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def evaluate(
        self,
        context: EvolutionGovernanceFrameworkContext,
        *,
        evaluated_at: datetime | None = None,
    ) -> EvolutionGovernanceFrameworkResult: ...


class EvolutionLifecycleGovernanceProvider(Protocol):
    """Checks lifecycle stage completeness — does not decide approval."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def assess_lifecycle(
        self,
        context: EvolutionGovernanceFrameworkContext,
    ) -> tuple[EvolutionGovernanceIssue, ...]: ...

    def stages_present(
        self,
        context: EvolutionGovernanceFrameworkContext,
    ) -> tuple[EvolutionGovernanceLifecycleStage, ...]: ...


class EvolutionGovernancePolicyProvider(Protocol):
    """Pluggable governance policy check — new policy equals new plugin."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def assess(
        self,
        context: EvolutionGovernanceFrameworkContext,
    ) -> tuple[EvolutionGovernanceIssue, ...]: ...


class EvolutionGovernanceControlProvider(Protocol):
    """Technical process consistency checks — not business approval."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def validate(
        self,
        context: EvolutionGovernanceFrameworkContext,
    ) -> tuple[EvolutionGovernanceIssue, ...]: ...


class EvolutionGovernanceFrameworkAuditProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def build_audit(
        self,
        context: EvolutionGovernanceFrameworkContext,
        *,
        lifecycle_provider_id: str,
        lifecycle_provider_version: str,
        policy_provider_ids: tuple[str, ...],
        policy_provider_versions: tuple[str, ...],
        control_provider_ids: tuple[str, ...],
        control_provider_versions: tuple[str, ...],
        governance_provider_ids: tuple[str, ...],
        governance_provider_versions: tuple[str, ...],
        data_source_refs: tuple[str, ...],
        evaluated_at: datetime,
        evaluation_scope_summary: str,
    ) -> EvolutionGovernanceFrameworkAuditMetadata: ...


__all__ = [
    "EnterpriseEvolutionGovernanceProvider",
    "EvolutionGovernanceControlProvider",
    "EvolutionGovernanceFrameworkAuditProvider",
    "EvolutionGovernancePolicyProvider",
    "EvolutionLifecycleGovernanceProvider",
]
