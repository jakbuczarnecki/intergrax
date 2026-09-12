# © Artur Czarnecki. All rights reserved.

"""Evolution governance framework audit metadata providers (DS-E2E-15J-L17)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.contracts import (
    ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_TASK_ID,
    ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_VERSION,
    EvolutionGovernanceFrameworkAuditMetadata,
    EvolutionGovernanceFrameworkContext,
)

_STANDARD_FRAMEWORK_AUDIT_PROVIDER_ID = "standard_evolution_governance_framework_audit"
_STANDARD_FRAMEWORK_AUDIT_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class StandardEvolutionGovernanceFrameworkAuditProvider:
    @property
    def provider_id(self) -> str:
        return _STANDARD_FRAMEWORK_AUDIT_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _STANDARD_FRAMEWORK_AUDIT_PROVIDER_VERSION

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
    ) -> EvolutionGovernanceFrameworkAuditMetadata:
        return EvolutionGovernanceFrameworkAuditMetadata(
            framework_task_id=ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_TASK_ID,
            framework_layer_version=ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_VERSION,
            scope_id=context.scope_id,
            scope_version=context.version,
            lifecycle_provider_id=lifecycle_provider_id,
            lifecycle_provider_version=lifecycle_provider_version,
            policy_provider_ids=policy_provider_ids,
            policy_provider_versions=policy_provider_versions,
            control_provider_ids=control_provider_ids,
            control_provider_versions=control_provider_versions,
            governance_provider_ids=governance_provider_ids,
            governance_provider_versions=governance_provider_versions,
            process_reference=context.process_reference,
            data_source_refs=data_source_refs,
            evaluation_scope_summary=evaluation_scope_summary,
            evaluated_at=evaluated_at,
        )


def default_evolution_governance_framework_audit_provider() -> (
    StandardEvolutionGovernanceFrameworkAuditProvider
):
    return StandardEvolutionGovernanceFrameworkAuditProvider()


__all__ = [
    "StandardEvolutionGovernanceFrameworkAuditProvider",
    "default_evolution_governance_framework_audit_provider",
]
