# © Artur Czarnecki. All rights reserved.

"""Pluggable top-level evolution governance providers (DS-E2E-15J-L17)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.audit_providers import (
    default_evolution_governance_framework_audit_provider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.contracts import (
    EvolutionGovernanceFrameworkContext,
    EvolutionGovernanceFrameworkResult,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.control_providers import (
    default_evolution_governance_control_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.engine import (
    run_governance_framework_evaluation,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.lifecycle_providers import (
    default_evolution_lifecycle_governance_provider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.policy_providers import (
    default_evolution_governance_policy_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.protocol import (
    EvolutionGovernanceControlProvider,
    EvolutionGovernanceFrameworkAuditProvider,
    EvolutionGovernancePolicyProvider,
    EvolutionLifecycleGovernanceProvider,
)

_DEFAULT_GOVERNANCE_PROVIDER_ID = "default_enterprise_evolution_governance"
_DEFAULT_GOVERNANCE_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DefaultEnterpriseEvolutionGovernanceProvider:
    lifecycle_provider: EvolutionLifecycleGovernanceProvider
    policy_providers: tuple[EvolutionGovernancePolicyProvider, ...]
    control_providers: tuple[EvolutionGovernanceControlProvider, ...]
    audit_provider: EvolutionGovernanceFrameworkAuditProvider

    @property
    def provider_id(self) -> str:
        return _DEFAULT_GOVERNANCE_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _DEFAULT_GOVERNANCE_PROVIDER_VERSION

    def evaluate(
        self,
        context: EvolutionGovernanceFrameworkContext,
        *,
        evaluated_at: datetime | None = None,
    ) -> EvolutionGovernanceFrameworkResult:
        return run_governance_framework_evaluation(
            context,
            lifecycle_provider=self.lifecycle_provider,
            policy_providers=self.policy_providers,
            control_providers=self.control_providers,
            audit_provider=self.audit_provider,
            governance_providers=(),
            evaluated_at=evaluated_at or datetime.now(tz=UTC),
        )


def default_enterprise_evolution_governance_provider() -> (
    DefaultEnterpriseEvolutionGovernanceProvider
):
    return DefaultEnterpriseEvolutionGovernanceProvider(
        lifecycle_provider=default_evolution_lifecycle_governance_provider(),
        policy_providers=default_evolution_governance_policy_providers(),
        control_providers=default_evolution_governance_control_providers(),
        audit_provider=default_evolution_governance_framework_audit_provider(),
    )


__all__ = [
    "DefaultEnterpriseEvolutionGovernanceProvider",
    "default_enterprise_evolution_governance_provider",
]
