# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance import (
    EnterpriseEvolutionAssuranceEngine,
    EvolutionAssuranceContext,
    EvolutionAssuranceFinding,
    EvolutionAssuranceFindingSeverity,
    EvolutionAssuranceStatus,
    default_enterprise_evolution_assurance_engine,
    default_enterprise_evolution_assurance_provider,
    default_evolution_assurance_audit_provider,
    default_evolution_compliance_validator_providers,
    default_evolution_evidence_validator_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework import (
    default_enterprise_evolution_governance_framework_engine,
)
from tests.unit.testing_support.decision_e2e.model_matrix.test_enterprise_evolution_governance_framework import (
    _complete_governance_context,
)
from tests.unit.testing_support.decision_e2e.model_matrix.test_enterprise_evolution_intelligence import (
    _stamp,
)


def _complete_assurance_context() -> EvolutionAssuranceContext:
    governance_ctx = _complete_governance_context()
    framework = default_enterprise_evolution_governance_framework_engine().evaluate(
        governance_ctx,
        evaluated_at=_stamp(),
    )
    return EvolutionAssuranceContext(
        scope_id=governance_ctx.scope_id,
        version=governance_ctx.version,
        process_reference=governance_ctx.process_reference,
        governance_framework_result=framework,
        intelligence_result=governance_ctx.intelligence_result,
        strategy_result=governance_ctx.strategy_result,
        governance_decision=governance_ctx.governance_decision,
        governance_reference=governance_ctx.governance_reference,
        execution_results=governance_ctx.execution_results,
        operation_records=governance_ctx.operation_records,
    )


def test_default_assurance_provider_complete_lifecycle_passed() -> None:
    provider = default_enterprise_evolution_assurance_provider()
    result = provider.assess(_complete_assurance_context(), assessed_at=_stamp())
    assert result.status is EvolutionAssuranceStatus.PASSED
    assert not result.findings


@dataclass(frozen=True, slots=True)
class CustomEvolutionQualityValidator:
    @property
    def provider_id(self) -> str:
        return "custom_evolution_quality_validator"

    @property
    def provider_version(self) -> str:
        return "99"

    def validate(
        self,
        context: EvolutionAssuranceContext,
    ) -> tuple[EvolutionAssuranceFinding, ...]:
        return (
            EvolutionAssuranceFinding(
                finding_id="custom-quality-signal",
                severity=EvolutionAssuranceFindingSeverity.WARNING,
                finding_code="custom_quality_signal",
                summary="Custom quality validator plugin attached.",
                lifecycle_stage=None,
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            ),
        )


def test_quality_validator_plugin_swap_engine_unchanged() -> None:
    engine = EnterpriseEvolutionAssuranceEngine(
        quality_validator_providers=(CustomEvolutionQualityValidator(),),
        compliance_validator_providers=default_evolution_compliance_validator_providers(),
        evidence_validator_providers=default_evolution_evidence_validator_providers(),
        assurance_providers=(),
        audit_provider=default_evolution_assurance_audit_provider(),
    )
    result = engine.assess(_complete_assurance_context(), assessed_at=_stamp())
    assert result.status is EvolutionAssuranceStatus.WARNING
    assert any(
        item.provider_id == "custom_evolution_quality_validator"
        for item in result.findings
    )
    assert "custom_evolution_quality_validator" in result.audit.quality_validator_ids


def test_missing_evidence_references_review_required() -> None:
    base = _complete_assurance_context()
    context = EvolutionAssuranceContext(
        scope_id=base.scope_id,
        version=base.version,
        process_reference=None,
        governance_framework_result=base.governance_framework_result,
        intelligence_result=base.intelligence_result,
        strategy_result=base.strategy_result,
        governance_decision=None,
        governance_reference=None,
        execution_results=base.execution_results,
        operation_records=base.operation_records,
    )
    result = default_enterprise_evolution_assurance_engine().assess(
        context,
        assessed_at=_stamp(),
    )
    assert result.status in (
        EvolutionAssuranceStatus.WARNING,
        EvolutionAssuranceStatus.REVIEW_REQUIRED,
    )
    assert any(
        item.finding_code
        in (
            "missing_process_reference",
            "missing_governance_evidence",
            "missing_governance_evidence",
            "required_stage_governance",
        )
        for item in result.findings
    )


def test_compliance_validation_detects_scope_inconsistency() -> None:
    base = _complete_assurance_context()
    framework = base.governance_framework_result
    assert framework is not None
    mismatched_audit = framework.audit.__class__(
        framework_task_id=framework.audit.framework_task_id,
        framework_layer_version=framework.audit.framework_layer_version,
        scope_id="other-scope",
        scope_version=framework.audit.scope_version,
        lifecycle_provider_id=framework.audit.lifecycle_provider_id,
        lifecycle_provider_version=framework.audit.lifecycle_provider_version,
        policy_provider_ids=framework.audit.policy_provider_ids,
        policy_provider_versions=framework.audit.policy_provider_versions,
        control_provider_ids=framework.audit.control_provider_ids,
        control_provider_versions=framework.audit.control_provider_versions,
        governance_provider_ids=framework.audit.governance_provider_ids,
        governance_provider_versions=framework.audit.governance_provider_versions,
        process_reference=framework.audit.process_reference,
        data_source_refs=framework.audit.data_source_refs,
        evaluation_scope_summary=framework.audit.evaluation_scope_summary,
        evaluated_at=framework.audit.evaluated_at,
    )
    mismatched_framework = framework.__class__(
        framework_task_id=framework.framework_task_id,
        status=framework.status,
        issues=framework.issues,
        lifecycle_stages_present=framework.lifecycle_stages_present,
        audit=mismatched_audit,
    )
    context = EvolutionAssuranceContext(
        scope_id=base.scope_id,
        version=base.version,
        process_reference=base.process_reference,
        governance_framework_result=mismatched_framework,
        intelligence_result=base.intelligence_result,
        strategy_result=base.strategy_result,
        governance_decision=base.governance_decision,
        governance_reference=base.governance_reference,
        execution_results=base.execution_results,
        operation_records=base.operation_records,
    )
    result = default_enterprise_evolution_assurance_engine().assess(
        context,
        assessed_at=_stamp(),
    )
    assert result.status is EvolutionAssuranceStatus.REVIEW_REQUIRED
    assert any(
        item.finding_code == "governance_framework_scope_mismatch"
        for item in result.findings
    )


def test_audit_records_plugins_sources_findings_and_timestamp() -> None:
    stamp = _stamp()
    result = default_enterprise_evolution_assurance_provider().assess(
        _complete_assurance_context(),
        assessed_at=stamp,
    )
    audit = result.audit
    assert audit.quality_validator_ids
    assert audit.compliance_validator_ids
    assert audit.evidence_validator_ids
    assert audit.evidence_refs
    assert audit.finding_ids == tuple(item.finding_id for item in result.findings)
    assert audit.assessed_at == stamp
    assert audit.process_reference is not None
