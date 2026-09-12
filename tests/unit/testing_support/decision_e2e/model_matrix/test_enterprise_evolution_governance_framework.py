# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework import (
    EnterpriseEvolutionGovernanceFrameworkEngine,
    EvolutionGovernanceFrameworkContext,
    EvolutionGovernanceFrameworkStatus,
    EvolutionGovernanceIssue,
    EvolutionGovernanceIssueSeverity,
    EvolutionGovernanceProcessReference,
    default_enterprise_evolution_governance_framework_engine,
    default_enterprise_evolution_governance_provider,
    default_evolution_governance_framework_audit_provider,
    default_evolution_lifecycle_governance_provider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.control_providers import (
    default_evolution_governance_control_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy import (
    EvolutionStrategyRunStatus,
    default_enterprise_evolution_strategy_engine,
)
from tests.unit.testing_support.decision_e2e.model_matrix.test_enterprise_evolution_strategy import (
    _strategy_context,
)
from tests.unit.testing_support.decision_e2e.model_matrix.test_enterprise_evolution_intelligence import (
    _stamp,
)


def _complete_governance_context() -> EvolutionGovernanceFrameworkContext:
    strategy_ctx = _strategy_context()
    strategy = default_enterprise_evolution_strategy_engine().analyze(
        strategy_ctx,
        analyzed_at=_stamp(),
    )
    assert strategy.status is EvolutionStrategyRunStatus.COMPLETE
    return EvolutionGovernanceFrameworkContext(
        scope_id=strategy_ctx.scope_id,
        version=strategy_ctx.version,
        process_reference=EvolutionGovernanceProcessReference(
            process_id="evo-process-1",
            version=strategy_ctx.version,
            description="Controlled enterprise evolution process.",
        ),
        intelligence_result=strategy_ctx.intelligence_result,
        strategy_result=strategy,
        governance_decision=strategy_ctx.governance_decision,
        governance_reference=strategy_ctx.governance_reference,
        execution_results=strategy_ctx.execution_results,
        operation_records=strategy_ctx.operation_records,
    )


def test_default_governance_provider_complete_evolution_process() -> None:
    provider = default_enterprise_evolution_governance_provider()
    result = provider.evaluate(_complete_governance_context(), evaluated_at=_stamp())
    assert result.status is EvolutionGovernanceFrameworkStatus.CONSISTENT
    assert not result.issues
    assert len(result.lifecycle_stages_present) == 5


@dataclass(frozen=True, slots=True)
class CustomEvolutionGovernancePolicyProvider:
    @property
    def provider_id(self) -> str:
        return "custom_evolution_governance_policy"

    @property
    def provider_version(self) -> str:
        return "99"

    def assess(
        self,
        context: EvolutionGovernanceFrameworkContext,
    ) -> tuple[EvolutionGovernanceIssue, ...]:
        return (
            EvolutionGovernanceIssue(
                issue_id="custom-policy-signal",
                severity=EvolutionGovernanceIssueSeverity.REVIEW,
                issue_code="custom_policy_signal",
                summary="Custom policy plugin attached.",
                lifecycle_stage=None,
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            ),
        )


def test_policy_plugin_swap_engine_unchanged() -> None:
    engine = EnterpriseEvolutionGovernanceFrameworkEngine(
        governance_providers=(),
        lifecycle_provider=default_evolution_lifecycle_governance_provider(),
        policy_providers=(CustomEvolutionGovernancePolicyProvider(),),
        control_providers=default_evolution_governance_control_providers(),
        audit_provider=default_evolution_governance_framework_audit_provider(),
    )
    result = engine.evaluate(_complete_governance_context(), evaluated_at=_stamp())
    assert result.status is EvolutionGovernanceFrameworkStatus.REVIEW_REQUIRED
    assert any(
        item.provider_id == "custom_evolution_governance_policy"
        for item in result.issues
    )
    assert "custom_evolution_governance_policy" in result.audit.policy_provider_ids


def test_missing_governance_reference_lifecycle_incomplete() -> None:
    base = _complete_governance_context()
    context = EvolutionGovernanceFrameworkContext(
        scope_id=base.scope_id,
        version=base.version,
        process_reference=base.process_reference,
        intelligence_result=base.intelligence_result,
        strategy_result=base.strategy_result,
        governance_decision=None,
        governance_reference=None,
        execution_results=base.execution_results,
        operation_records=base.operation_records,
    )
    result = default_enterprise_evolution_governance_framework_engine().evaluate(
        context,
        evaluated_at=_stamp(),
    )
    assert result.status in (
        EvolutionGovernanceFrameworkStatus.INCOMPLETE,
        EvolutionGovernanceFrameworkStatus.REVIEW_REQUIRED,
    )
    assert any(
        item.issue_code == "missing_governance_reference" for item in result.issues
    )


def test_control_validation_detects_version_mismatch() -> None:
    base = _complete_governance_context()
    execution = base.execution_results[0]
    mismatched = execution.__class__(
        status=execution.status,
        provider_id=execution.provider_id,
        provider_version=execution.provider_version,
        applied_change_reference=execution.applied_change_reference,
        audit_metadata=execution.audit_metadata,
        adaptation_id=execution.adaptation_id,
        version="mismatch-version",
        source_reference=execution.source_reference,
    )
    context = EvolutionGovernanceFrameworkContext(
        scope_id=base.scope_id,
        version=base.version,
        process_reference=base.process_reference,
        intelligence_result=base.intelligence_result,
        strategy_result=base.strategy_result,
        governance_decision=base.governance_decision,
        governance_reference=base.governance_reference,
        execution_results=(mismatched,),
        operation_records=base.operation_records,
    )
    result = default_enterprise_evolution_governance_framework_engine().evaluate(
        context,
        evaluated_at=_stamp(),
    )
    assert result.status is EvolutionGovernanceFrameworkStatus.REVIEW_REQUIRED
    assert any(
        item.issue_code == "adaptation_version_mismatch" for item in result.issues
    )


def test_audit_records_plugins_sources_and_timestamp() -> None:
    stamp = _stamp()
    result = default_enterprise_evolution_governance_provider().evaluate(
        _complete_governance_context(),
        evaluated_at=stamp,
    )
    audit = result.audit
    assert audit.lifecycle_provider_id == "default_evolution_lifecycle_governance"
    assert audit.policy_provider_ids
    assert audit.control_provider_ids
    assert audit.data_source_refs
    assert audit.evaluated_at == stamp
    assert audit.process_reference is not None
