# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-DECISION-SYSTEM-PRODUCTION-QUALIFICATION acceptance bundle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.decision.integration import (
    DecisionIntegrationStatus,
    DecisionSystemIntegrationEngine,
    DecisionSystemIntegrationFactory,
    InMemoryDecisionAuditSink,
    PluginAdmissionDecision,
    REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
    ReferenceDecisionLifecycleReference,
    ReferenceEnterpriseLifecycleState,
    SingleLifecycleAdapterProvider,
)
from intergrax.contracts.decision.integration.admission import (
    DecisionIntegrationPluginDescriptor,
)
from intergrax.contracts.decision.integration.result import (
    DecisionAdapterMetadata,
    DecisionIntegrationResult,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.runtime.decision_integration_composition import (
    production_decision_integration_composition_provider,
    production_decision_system_integration,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleRecord,
    DecisionLifecycleState,
    DecisionSourceKind,
    DecisionSourceReference,
    DecisionType,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing import (
    DataSensitivityClass,
    GovernanceDisposition,
    GovernanceEvaluationEngine,
    GovernancePolicyRef,
    GovernanceRiskTier,
    GovernanceTaskContext,
    default_governance_policies,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline import (
    CapabilityDimensionId,
    CapabilityProfileBuildRequest,
    ObservationLevel,
    build_model_capability_profiles,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation import (
    CapabilitySelectionConstraints,
    ModelSelectionEngine,
    ModelSelectionRequest,
    TaskCapabilityRequirement,
    TaskRequirements,
    default_selection_strategies,
)
from testing_support.decision_e2e.model_matrix.production_decision_orchestration import (
    DecisionExecutionRequest,
    DecisionExecutionStatus,
    DecisionOrchestrationLifecycleStage,
    DecisionOrchestrationOutcome,
    DecisionOrchestrationRequest,
    DecisionOrchestrator,
    EngineBackedGovernanceProvider,
    EngineBackedSelectionProvider,
    RecordingExecutionProvider,
)
from testing_support.decision_e2e.local_ai_incident_qualification import (
    QualificationCliExit,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionStatus,
)
from testing_support.decision_e2e.model_matrix.registry import (
    qualification_matrix_version,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTEGRATION_COMPOSITION = (
    _REPO_ROOT / "intergrax" / "runtime" / "decision_integration_composition.py"
)


def _reference_source(
    decision_id: str = "pq-decision-001",
) -> ReferenceDecisionLifecycleReference:
    record = DecisionLifecycleRecord(
        decision_id=decision_id,
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        lifecycle_state=DecisionLifecycleState.CREATED,
        created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        source_references=(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.MODEL_SELECTION,
                reference_id="sel-pq",
            ),
        ),
    )
    return ReferenceDecisionLifecycleReference(
        source_type=REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
        decision_id=record.decision_id,
        lifecycle_state=ReferenceEnterpriseLifecycleState(record.lifecycle_state.value),
        decision_type=record.decision_type.value,
        created_at_iso=record.created_at.isoformat(),
        mapping_version="1",
    )


def _matrix_outcome(profile_key: str) -> ModelQualificationOutcome:
    stamp = datetime(2026, 9, 12, 9, 0, 0, tzinfo=UTC)
    return ModelQualificationOutcome(
        profile_key=profile_key,
        provider="ollama",
        model_name=profile_key,
        matrix_version=qualification_matrix_version(),
        qualification_task_id="DS-E2E-15J-L1.R6",
        evaluated_at=stamp,
        status=CohortExecutionStatus.EXECUTED,
        exit_code=QualificationCliExit.SUCCESS,
        session_state=QualificationSessionState.FINALIZED,
    )


def _orchestration_request() -> DecisionOrchestrationRequest:
    profiles = build_model_capability_profiles(
        CapabilityProfileBuildRequest(
            matrix_version=qualification_matrix_version(),
            outcomes=(_matrix_outcome("model-a"), _matrix_outcome("model-b")),
        )
    ).profiles
    return DecisionOrchestrationRequest(
        selection_request=ModelSelectionRequest(
            task_requirements=TaskRequirements(
                scenario_id="pq-scenario",
                capability_requirements=(
                    TaskCapabilityRequirement(
                        dimension_id=CapabilityDimensionId.QUALIFICATION_EXIT,
                        minimum_level=ObservationLevel.MODERATE,
                    ),
                ),
            ),
            capability_constraints=CapabilitySelectionConstraints(
                required_matrix_version=qualification_matrix_version(),
                excluded_profile_keys=(),
                require_behavioral_baseline=False,
            ),
            available_model_profiles=profiles,
        ),
        governance_task_context=GovernanceTaskContext(
            scenario_id="pq-scenario",
            data_sensitivity=DataSensitivityClass.PUBLIC,
            risk_tier=GovernanceRiskTier.LOW,
        ),
        applicable_policies=tuple(
            GovernancePolicyRef(
                policy_id=item.policy_id,
                policy_version=item.policy_version,
            )
            for item in default_governance_policies()
        ),
        capability_evidence=profiles,
    )


@dataclass(frozen=True, slots=True)
class _FailingExecutionProvider:
    @property
    def provider_id(self) -> str:
        return "failing-execution"

    def execute(
        self,
        request: DecisionExecutionRequest,
        *,
        executed_at: datetime | None = None,
    ):
        raise RuntimeError("simulated execution provider fault")


def _orchestrator(
    *,
    execution_provider: RecordingExecutionProvider | _FailingExecutionProvider,
) -> DecisionOrchestrator:
    return DecisionOrchestrator(
        selection_provider=EngineBackedSelectionProvider(
            ModelSelectionEngine(strategies=default_selection_strategies())
        ),
        governance_provider=EngineBackedGovernanceProvider(
            GovernanceEvaluationEngine(evaluators=default_governance_policies())
        ),
        execution_provider=execution_provider,
    )


@dataclass(frozen=True, slots=True)
class _StaticAdmissionProvider:
    decision: PluginAdmissionDecision

    def evaluate(
        self,
        descriptor: DecisionIntegrationPluginDescriptor,
    ) -> PluginAdmissionDecision:
        return self.decision


@dataclass(frozen=True, slots=True)
class _CustomLifecycleAdapter:
    @property
    def adapter_id(self) -> str:
        return "pq.custom.lifecycle"

    @property
    def adapter_version(self) -> str:
        return "1.0.0"

    @property
    def mapping_version(self) -> str:
        return "pq-1"

    @property
    def source_type(self) -> str:
        return REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE

    def integrate_lifecycle(
        self,
        source: ReferenceDecisionLifecycleReference,
    ) -> DecisionIntegrationResult:
        integrated_at = datetime.now(tz=UTC)
        from intergrax.contracts.decision.integration.references import (
            PlatformDecisionLifecycleReference,
        )

        metadata = DecisionAdapterMetadata(
            source_type=source.source_type,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            mapping_version=self.mapping_version,
            integrated_at=integrated_at,
        )
        target = PlatformDecisionLifecycleReference(
            reference_decision_id=source.decision_id,
            stage=DecisionLifecycleStage.VERIFICATION,
            transition_index=0,
            mapping_version=self.mapping_version,
        )
        return DecisionIntegrationResult(
            status=DecisionIntegrationStatus.SUCCESS,
            source=source,
            target=target,
            adapter_metadata=metadata,
            detail="pq custom adapter",
        )


def test_qualification_01_successful_end_to_end_flow() -> None:
    orchestration = _orchestrator(
        execution_provider=RecordingExecutionProvider()
    ).orchestrate(
        _orchestration_request(),
    )
    assert orchestration.outcome is DecisionOrchestrationOutcome.SUCCESS
    assert orchestration.governance_result is not None
    assert orchestration.governance_result.disposition is GovernanceDisposition.ALLOW
    assert orchestration.execution_result_reference is not None
    assert (
        orchestration.execution_result_reference.status
        is DecisionExecutionStatus.EXECUTED
    )

    sink = InMemoryDecisionAuditSink()
    integration = production_decision_system_integration(audit_sink=sink)
    integration_result = integration.integrate_lifecycle(_reference_source())
    assert integration_result.status is DecisionIntegrationStatus.SUCCESS
    assert len(sink.entries) == 1


def test_qualification_02_governance_blocked_execution() -> None:
    @dataclass
    class _SpyExecutionProvider:
        calls: int = 0

        @property
        def provider_id(self) -> str:
            return "spy-execution"

        def execute(
            self,
            request: DecisionExecutionRequest,
            *,
            executed_at: datetime | None = None,
        ):
            self.calls += 1
            raise AssertionError("execution must not run when governance blocks")

    spy = _SpyExecutionProvider()
    orchestrator = DecisionOrchestrator(
        selection_provider=EngineBackedSelectionProvider(
            ModelSelectionEngine(strategies=default_selection_strategies())
        ),
        governance_provider=EngineBackedGovernanceProvider(
            GovernanceEvaluationEngine(evaluators=default_governance_policies())
        ),
        execution_provider=spy,
    )
    request = _orchestration_request()
    blocked_request = DecisionOrchestrationRequest(
        selection_request=request.selection_request,
        governance_task_context=GovernanceTaskContext(
            scenario_id="pq-scenario",
            data_sensitivity=DataSensitivityClass.PUBLIC,
            risk_tier=GovernanceRiskTier.HIGH,
        ),
        applicable_policies=request.applicable_policies,
        capability_evidence=request.capability_evidence,
    )
    result = orchestrator.orchestrate(blocked_request)

    assert result.outcome is DecisionOrchestrationOutcome.APPROVAL_REQUIRED
    assert result.governance_result is not None
    assert (
        result.governance_result.disposition is GovernanceDisposition.REQUIRE_APPROVAL
    )
    assert result.execution_result_reference is None
    assert spy.calls == 0
    assert (
        DecisionOrchestrationLifecycleStage.STOPPED
        in result.lifecycle_metadata.lifecycle_stages
    )


def test_qualification_03_execution_failure_propagation() -> None:
    with pytest.raises(RuntimeError, match="simulated execution provider fault"):
        _orchestrator(execution_provider=_FailingExecutionProvider()).orchestrate(
            _orchestration_request(),
        )


def test_qualification_04_evidence_correlation() -> None:
    stamp = datetime(2026, 9, 12, 15, 0, 0, tzinfo=UTC)
    orchestration = _orchestrator(
        execution_provider=RecordingExecutionProvider()
    ).orchestrate(
        _orchestration_request(),
        orchestrated_at=stamp,
    )
    exec_ref = orchestration.execution_result_reference
    assert exec_ref is not None
    assert exec_ref.provider_id == "recording-execution"
    assert exec_ref.governance_decision_id
    assert exec_ref.selection_task_id
    meta = orchestration.lifecycle_metadata
    assert meta.orchestrated_at == stamp
    assert meta.selection_provider_id == "engine-backed-selection"
    assert meta.governance_provider_id == "engine-backed-governance"
    assert meta.execution_provider_id == "recording-execution"

    sink = InMemoryDecisionAuditSink()
    integration = production_decision_system_integration(audit_sink=sink)
    source = _reference_source("pq-correlation-001")
    integration.integrate_lifecycle(source)
    envelope = sink.entries[0]
    assert envelope.record.source.decision_id == source.decision_id
    assert envelope.record.adapter_metadata.mapping_version == "1"
    assert (
        envelope.provider_metadata.provider_id == "decision.integration.audit.recording"
    )


def test_qualification_05_plugin_replacement_without_engine_modification() -> None:
    engine = DecisionSystemIntegrationEngine(
        adapter_providers=(
            SingleLifecycleAdapterProvider(lifecycle_adapter=_CustomLifecycleAdapter()),
        ),
    )
    result = engine.integrate_lifecycle(_reference_source())

    assert type(engine) is DecisionSystemIntegrationEngine
    assert result.adapter_metadata.adapter_id == "pq.custom.lifecycle"
    assert result.target is not None
    assert result.target.stage is DecisionLifecycleStage.VERIFICATION


def test_qualification_06_full_composition_startup() -> None:
    provider = production_decision_integration_composition_provider()
    engine = DecisionSystemIntegrationFactory.create_engine(provider)
    result = engine.integrate_lifecycle(_reference_source("pq-composition-001"))

    assert result.status is DecisionIntegrationStatus.SUCCESS
    assert result.target is not None


def test_qualification_decision_failure_blocks_integration_without_execution() -> None:
    engine = production_decision_system_integration(
        plugin_admission_provider=_StaticAdmissionProvider(
            decision=PluginAdmissionDecision.DENY,
        ),
    )
    result = engine.integrate_lifecycle(_reference_source("pq-failure-001"))
    assert result.status is DecisionIntegrationStatus.FAILED


def test_qualification_integration_boundary_does_not_wire_execution_runtime() -> None:
    text = _INTEGRATION_COMPOSITION.read_text(encoding="utf-8")
    assert "intergrax.runtime.execution" not in text
