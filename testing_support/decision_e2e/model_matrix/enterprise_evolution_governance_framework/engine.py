# © Artur Czarnecki. All rights reserved.

"""Enterprise evolution governance framework orchestration via injected plugins (DS-E2E-15J-L17)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.contracts import (
    ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_TASK_ID,
    EvolutionGovernanceFrameworkContext,
    EvolutionGovernanceFrameworkResult,
    EvolutionGovernanceFrameworkStatus,
    EvolutionGovernanceIssue,
    EvolutionGovernanceIssueSeverity,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.protocol import (
    EnterpriseEvolutionGovernanceProvider,
    EvolutionGovernanceControlProvider,
    EvolutionGovernanceFrameworkAuditProvider,
    EvolutionGovernancePolicyProvider,
    EvolutionLifecycleGovernanceProvider,
)


def _provider_ids(
    providers: tuple[
        EvolutionGovernancePolicyProvider | EvolutionGovernanceControlProvider, ...
    ],
) -> tuple[str, ...]:
    return tuple(item.provider_id for item in providers)


def _provider_versions(
    providers: tuple[
        EvolutionGovernancePolicyProvider | EvolutionGovernanceControlProvider, ...
    ],
) -> tuple[str, ...]:
    return tuple(item.provider_version for item in providers)


def _governance_provider_ids(
    providers: tuple[EnterpriseEvolutionGovernanceProvider, ...],
) -> tuple[str, ...]:
    return tuple(item.provider_id for item in providers)


def _governance_provider_versions(
    providers: tuple[EnterpriseEvolutionGovernanceProvider, ...],
) -> tuple[str, ...]:
    return tuple(item.provider_version for item in providers)


def _aggregate_status(
    issues: tuple[EvolutionGovernanceIssue, ...],
) -> EvolutionGovernanceFrameworkStatus:
    if any(
        item.severity is EvolutionGovernanceIssueSeverity.BLOCKED for item in issues
    ):
        return EvolutionGovernanceFrameworkStatus.BLOCKED
    if any(
        item.severity is EvolutionGovernanceIssueSeverity.INCOMPLETE for item in issues
    ):
        return EvolutionGovernanceFrameworkStatus.INCOMPLETE
    if issues:
        return EvolutionGovernanceFrameworkStatus.REVIEW_REQUIRED
    return EvolutionGovernanceFrameworkStatus.CONSISTENT


def _collect_data_source_refs(
    context: EvolutionGovernanceFrameworkContext,
) -> tuple[str, ...]:
    refs: list[str] = []
    if context.process_reference is not None:
        refs.append(
            f"process:{context.process_reference.process_id}:"
            f"v{context.process_reference.version}"
        )
    governance = context.governance_reference
    if governance is not None:
        refs.append(f"governance:{governance.governance_decision_reference}")
    for execution in context.execution_results:
        if execution.applied_change_reference is not None:
            refs.append(f"adaptation:{execution.applied_change_reference}")
    for record in context.operation_records:
        refs.append(f"operation:{record.record_id}")
    return tuple(dict.fromkeys(refs))


def run_governance_framework_evaluation(
    context: EvolutionGovernanceFrameworkContext,
    *,
    lifecycle_provider: EvolutionLifecycleGovernanceProvider,
    policy_providers: tuple[EvolutionGovernancePolicyProvider, ...],
    control_providers: tuple[EvolutionGovernanceControlProvider, ...],
    audit_provider: EvolutionGovernanceFrameworkAuditProvider,
    governance_providers: tuple[EnterpriseEvolutionGovernanceProvider, ...] = (),
    evaluated_at: datetime | None = None,
) -> EvolutionGovernanceFrameworkResult:
    stamp = evaluated_at or datetime.now(tz=UTC)
    lifecycle_issues = lifecycle_provider.assess_lifecycle(context)
    policy_issues: list[EvolutionGovernanceIssue] = []
    for policy in policy_providers:
        policy_issues.extend(policy.assess(context))
    control_issues: list[EvolutionGovernanceIssue] = []
    for control in control_providers:
        control_issues.extend(control.validate(context))

    supplemental_issues: list[EvolutionGovernanceIssue] = []
    for provider in governance_providers:
        partial = provider.evaluate(context, evaluated_at=stamp)
        supplemental_issues.extend(partial.issues)

    all_issues = (
        *lifecycle_issues,
        *tuple(policy_issues),
        *tuple(control_issues),
        *tuple(supplemental_issues),
    )
    status = _aggregate_status(all_issues)
    stages = lifecycle_provider.stages_present(context)
    scope_summary = (
        f"Governance framework evaluation for scope {context.scope_id} "
        f"v{context.version} with {len(stages)} lifecycle stages present "
        f"and {len(all_issues)} issue(s) recorded."
    )
    audit = audit_provider.build_audit(
        context,
        lifecycle_provider_id=lifecycle_provider.provider_id,
        lifecycle_provider_version=lifecycle_provider.provider_version,
        policy_provider_ids=_provider_ids(policy_providers),
        policy_provider_versions=_provider_versions(policy_providers),
        control_provider_ids=_provider_ids(control_providers),
        control_provider_versions=_provider_versions(control_providers),
        governance_provider_ids=_governance_provider_ids(governance_providers),
        governance_provider_versions=_governance_provider_versions(
            governance_providers
        ),
        data_source_refs=_collect_data_source_refs(context),
        evaluated_at=stamp,
        evaluation_scope_summary=scope_summary,
    )
    return EvolutionGovernanceFrameworkResult(
        framework_task_id=ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_TASK_ID,
        status=status,
        issues=all_issues,
        lifecycle_stages_present=stages,
        audit=audit,
    )


@dataclass(frozen=True, slots=True)
class EnterpriseEvolutionGovernanceFrameworkEngine:
    governance_providers: tuple[EnterpriseEvolutionGovernanceProvider, ...]
    lifecycle_provider: EvolutionLifecycleGovernanceProvider
    policy_providers: tuple[EvolutionGovernancePolicyProvider, ...]
    control_providers: tuple[EvolutionGovernanceControlProvider, ...]
    audit_provider: EvolutionGovernanceFrameworkAuditProvider

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
            governance_providers=self.governance_providers,
            evaluated_at=evaluated_at,
        )


__all__ = [
    "EnterpriseEvolutionGovernanceFrameworkEngine",
    "run_governance_framework_evaluation",
]
