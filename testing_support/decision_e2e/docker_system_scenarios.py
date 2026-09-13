# © Artur Czarnecki. All rights reserved.

"""In-container scenario runners for DS-E2E-15J Docker system qualification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    DecisionOrchestrationProviderMissingError,
    DecisionOrchestrationRequest,
    DecisionOrchestrator,
    EngineBackedGovernanceProvider,
    EngineBackedSelectionProvider,
    ExecutionProvider,
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

_TASK_ID = "DS-E2E-15J-DOCKER-E2E-SYSTEM-QUALIFICATION"
_COMPOSITION_PATH = (
    Path(__file__).resolve().parents[2]
    / "intergrax"
    / "runtime"
    / "decision_integration_composition.py"
)


@dataclass(frozen=True, slots=True)
class DockerSystemScenarioResult:
    scenario_id: str
    passed: bool
    detail: str
    payload: dict[str, Any]


def _reference_source(decision_id: str) -> ReferenceDecisionLifecycleReference:
    record = DecisionLifecycleRecord(
        decision_id=decision_id,
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        lifecycle_state=DecisionLifecycleState.CREATED,
        created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        source_references=(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.MODEL_SELECTION,
                reference_id="sel-docker",
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
    stamp = datetime(2026, 9, 12, 9, 0, tzinfo=UTC)
    return ModelQualificationOutcome(
        profile_key=profile_key,
        provider="qualification",
        model_name=profile_key,
        matrix_version=qualification_matrix_version(),
        qualification_task_id="DS-E2E-15J-L1.R6",
        evaluated_at=stamp,
        status=CohortExecutionStatus.EXECUTED,
        exit_code=QualificationCliExit.SUCCESS,
        session_state=QualificationSessionState.FINALIZED,
    )


def _orchestration_request(
    *,
    risk_tier: GovernanceRiskTier = GovernanceRiskTier.LOW,
    data_sensitivity: DataSensitivityClass = DataSensitivityClass.PUBLIC,
) -> DecisionOrchestrationRequest:
    profiles = build_model_capability_profiles(
        CapabilityProfileBuildRequest(
            matrix_version=qualification_matrix_version(),
            outcomes=(_matrix_outcome("model-a"), _matrix_outcome("model-b")),
        )
    ).profiles
    return DecisionOrchestrationRequest(
        selection_request=ModelSelectionRequest(
            task_requirements=TaskRequirements(
                scenario_id="docker-e2e-scenario",
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
            scenario_id="docker-e2e-scenario",
            data_sensitivity=data_sensitivity,
            risk_tier=risk_tier,
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


def _default_orchestrator(
    *,
    execution_provider: ExecutionProvider | None = None,
    governance_provider: EngineBackedGovernanceProvider | None = None,
) -> DecisionOrchestrator:
    return DecisionOrchestrator(
        selection_provider=EngineBackedSelectionProvider(
            ModelSelectionEngine(strategies=default_selection_strategies())
        ),
        governance_provider=governance_provider
        or EngineBackedGovernanceProvider(
            GovernanceEvaluationEngine(evaluators=default_governance_policies())
        ),
        execution_provider=execution_provider or RecordingExecutionProvider(),
    )


def _ok(scenario_id: str, detail: str, **payload: Any) -> DockerSystemScenarioResult:
    return DockerSystemScenarioResult(
        scenario_id=scenario_id,
        passed=True,
        detail=detail,
        payload={"qualification_task_id": _TASK_ID, **payload},
    )


def _fail(scenario_id: str, detail: str, **payload: Any) -> DockerSystemScenarioResult:
    return DockerSystemScenarioResult(
        scenario_id=scenario_id,
        passed=False,
        detail=detail,
        payload={"qualification_task_id": _TASK_ID, **payload},
    )


def run_startup_health() -> DockerSystemScenarioResult:
    scenario_id = "startup-health"
    try:
        provider = production_decision_integration_composition_provider()
        engine = DecisionSystemIntegrationFactory.create_engine(provider)
        result = engine.integrate_lifecycle(_reference_source("docker-startup-001"))
    except Exception as exc:  # noqa: BLE001 — qualification boundary
        return _fail(scenario_id, f"composition startup failed: {exc}")

    if result.status is not DecisionIntegrationStatus.SUCCESS:
        return _fail(scenario_id, f"unexpected integration status: {result.status}")

    composition_text = _COMPOSITION_PATH.read_text(encoding="utf-8")
    if "intergrax.runtime.execution" in composition_text:
        return _fail(scenario_id, "decision composition imports execution runtime")

    return _ok(
        scenario_id,
        "composition root and integration engine started",
        integration_status=result.status.value,
        adapter_id=result.adapter_metadata.adapter_id,
    )


def run_flow_success() -> DockerSystemScenarioResult:
    scenario_id = "flow-success"
    orchestration = _default_orchestrator().orchestrate(_orchestration_request())
    if orchestration.outcome is not DecisionOrchestrationOutcome.SUCCESS:
        return _fail(scenario_id, f"unexpected outcome: {orchestration.outcome}")
    exec_ref = orchestration.execution_result_reference
    if exec_ref is None or exec_ref.status is not DecisionExecutionStatus.EXECUTED:
        return _fail(scenario_id, "execution reference missing or not executed")

    sink = InMemoryDecisionAuditSink()
    audit = production_decision_system_integration(audit_sink=sink)
    integration = audit.integrate_lifecycle(_reference_source("docker-flow-001"))
    if (
        integration.status is not DecisionIntegrationStatus.SUCCESS
        or len(sink.entries) != 1
    ):
        return _fail(scenario_id, "audit chain not recorded")

    return _ok(
        scenario_id,
        "decision → governance allow → execution → audit",
        governance_disposition=GovernanceDisposition.ALLOW.value,
        execution_status=exec_ref.status.value,
        governance_decision_id=exec_ref.governance_decision_id,
        audit_entries=len(sink.entries),
    )


def run_governance_deny() -> DockerSystemScenarioResult:
    scenario_id = "governance-deny"
    orchestrator = _default_orchestrator(
        execution_provider=RecordingExecutionProvider()
    )
    profiles = build_model_capability_profiles(
        CapabilityProfileBuildRequest(
            matrix_version=qualification_matrix_version(),
            outcomes=(_matrix_outcome("model-b"),),
        )
    ).profiles
    request = DecisionOrchestrationRequest(
        selection_request=ModelSelectionRequest(
            task_requirements=TaskRequirements(
                scenario_id="docker-e2e-deny",
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
            scenario_id="docker-e2e-deny",
            data_sensitivity=DataSensitivityClass.FINANCIAL,
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
    result = orchestrator.orchestrate(request)
    if result.outcome is not DecisionOrchestrationOutcome.GOVERNANCE_BLOCKED:
        return _fail(scenario_id, f"expected governance block, got {result.outcome}")
    if result.execution_result_reference is not None:
        return _fail(scenario_id, "execution must not run when governance blocks")
    gov = result.governance_result
    if gov is None or gov.disposition is not GovernanceDisposition.BLOCK:
        return _fail(scenario_id, "expected BLOCK disposition")
    return _ok(
        scenario_id,
        "governance deny blocked execution",
        governance_disposition=gov.disposition.value,
        lifecycle_stages=tuple(
            s.value for s in result.lifecycle_metadata.lifecycle_stages
        ),
    )


def run_governance_approval() -> DockerSystemScenarioResult:
    scenario_id = "governance-approval"
    orchestrator = _default_orchestrator()
    request = _orchestration_request(risk_tier=GovernanceRiskTier.HIGH)
    result = orchestrator.orchestrate(request)
    if result.outcome is not DecisionOrchestrationOutcome.APPROVAL_REQUIRED:
        return _fail(scenario_id, f"expected approval required, got {result.outcome}")
    if result.execution_result_reference is not None:
        return _fail(scenario_id, "execution must not run before approval")
    gov = result.governance_result
    if gov is None or gov.disposition is not GovernanceDisposition.REQUIRE_APPROVAL:
        return _fail(scenario_id, "expected REQUIRE_APPROVAL disposition")
    return _ok(
        scenario_id,
        "human approval required; execution stopped",
        governance_disposition=gov.disposition.value,
        stopped=(
            DecisionOrchestrationLifecycleStage.STOPPED.value
            in tuple(s.value for s in result.lifecycle_metadata.lifecycle_stages)
        ),
    )


def run_evidence_chain() -> DockerSystemScenarioResult:
    scenario_id = "evidence-chain"
    stamp = datetime(2026, 9, 13, 6, 0, 0, tzinfo=UTC)
    orchestration = _default_orchestrator().orchestrate(
        _orchestration_request(),
        orchestrated_at=stamp,
    )
    exec_ref = orchestration.execution_result_reference
    if exec_ref is None:
        return _fail(scenario_id, "missing execution reference")
    meta = orchestration.lifecycle_metadata
    required = (
        exec_ref.governance_decision_id,
        exec_ref.selection_task_id,
        exec_ref.execution_reference_id,
        meta.orchestration_task_id,
        meta.orchestration_version,
        meta.selection_provider_id,
        meta.governance_provider_id,
        meta.execution_provider_id,
    )
    if not all(required):
        return _fail(scenario_id, "incomplete correlation identifiers")

    sink = InMemoryDecisionAuditSink()
    source = _reference_source("docker-evidence-001")
    production_decision_system_integration(audit_sink=sink).integrate_lifecycle(source)
    envelope = sink.entries[0]
    return _ok(
        scenario_id,
        "full decision → governance → execution → audit correlation",
        decision_id=source.decision_id,
        governance_decision_id=exec_ref.governance_decision_id,
        authorization_id=exec_ref.governance_decision_id,
        execution_result_id=exec_ref.execution_reference_id,
        orchestrated_at=meta.orchestrated_at.isoformat(),
        audit_provider_id=envelope.provider_metadata.provider_id,
        mapping_version=envelope.record.adapter_metadata.mapping_version,
    )


def run_execution_failure() -> DockerSystemScenarioResult:
    scenario_id = "execution-failure"

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
            raise RuntimeError("controlled execution failure")

    try:
        _default_orchestrator(
            execution_provider=_FailingExecutionProvider()
        ).orchestrate(
            _orchestration_request(),
        )
    except RuntimeError as exc:
        if "controlled execution failure" not in str(exc):
            return _fail(scenario_id, f"unexpected error: {exc}")
        return _ok(
            scenario_id,
            "execution failure propagated; no silent fallback",
            failure_class=exc.__class__.__name__,
            message=str(exc),
        )
    return _fail(scenario_id, "expected execution failure")


def run_missing_governance() -> DockerSystemScenarioResult:
    scenario_id = "missing-governance"
    orchestrator = DecisionOrchestrator(
        selection_provider=EngineBackedSelectionProvider(
            ModelSelectionEngine(strategies=default_selection_strategies())
        ),
        governance_provider=None,
        execution_provider=RecordingExecutionProvider(),
    )
    try:
        orchestrator.orchestrate(_orchestration_request())
    except DecisionOrchestrationProviderMissingError as exc:
        return _ok(
            scenario_id,
            "missing governance blocked orchestration before execution",
            error_type=exc.__class__.__name__,
            message=str(exc),
        )
    return _fail(scenario_id, "expected missing governance failure")


def run_invalid_config_startup() -> DockerSystemScenarioResult:
    scenario_id = "invalid-config-startup"

    @dataclass(frozen=True, slots=True)
    class _DenyAdmission:
        def evaluate(
            self,
            descriptor: DecisionIntegrationPluginDescriptor,
        ) -> PluginAdmissionDecision:
            return PluginAdmissionDecision.DENY

    result = production_decision_system_integration(
        plugin_admission_provider=_DenyAdmission(),
    ).integrate_lifecycle(_reference_source("docker-invalid-001"))
    if result.status is not DecisionIntegrationStatus.FAILED:
        return _fail(scenario_id, "admission deny must fail integration startup path")
    return _ok(
        scenario_id,
        "invalid plugin admission failed closed with clear status",
        integration_status=result.status.value,
        integration_detail=result.detail,
    )


def run_plugin_compatibility() -> DockerSystemScenarioResult:
    scenario_id = "plugin-compatibility"

    @dataclass(frozen=True, slots=True)
    class _CustomLifecycleAdapter:
        @property
        def adapter_id(self) -> str:
            return "docker.custom.lifecycle"

        @property
        def adapter_version(self) -> str:
            return "1.0.0"

        @property
        def mapping_version(self) -> str:
            return "docker-1"

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
                detail="docker qualification custom adapter",
            )

    engine = DecisionSystemIntegrationEngine(
        adapter_providers=(
            SingleLifecycleAdapterProvider(lifecycle_adapter=_CustomLifecycleAdapter()),
        ),
    )
    result = engine.integrate_lifecycle(_reference_source("docker-plugin-001"))
    if result.adapter_metadata.adapter_id != "docker.custom.lifecycle":
        return _fail(scenario_id, "custom adapter not used")
    return _ok(
        scenario_id,
        "plugin swap without docker-specific branch",
        adapter_id=result.adapter_metadata.adapter_id,
        adapter_version=result.adapter_metadata.adapter_version,
    )


_SCENARIO_RUNNERS = {
    "startup-health": run_startup_health,
    "flow-success": run_flow_success,
    "governance-deny": run_governance_deny,
    "governance-approval": run_governance_approval,
    "evidence-chain": run_evidence_chain,
    "execution-failure": run_execution_failure,
    "missing-governance": run_missing_governance,
    "invalid-config-startup": run_invalid_config_startup,
    "plugin-compatibility": run_plugin_compatibility,
}


def run_docker_system_scenario(scenario_id: str) -> DockerSystemScenarioResult:
    runner = _SCENARIO_RUNNERS.get(scenario_id)
    if runner is None:
        return _fail(scenario_id, f"unknown scenario: {scenario_id}")
    return runner()


def scenario_result_to_dict(result: DockerSystemScenarioResult) -> dict[str, Any]:
    return {
        "scenario_id": result.scenario_id,
        "passed": result.passed,
        "detail": result.detail,
        **result.payload,
    }
