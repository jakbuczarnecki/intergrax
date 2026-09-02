# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.agent_execution_validation import validate_agent_execution
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifact,
    DecisionVersionLineage,
    candidate_decision_ref,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_structural_validation import (
    DecisionStructuralValidator,
    StructuralValidationFailure,
    StructuralValidationOutcome,
    structural_validation_failed,
    structural_validation_passed,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationStageOutcome,
    validate_verification_finding_code,
    validate_verification_requirement_code,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.decision_verification_stages.structural import (
    STRUCTURAL_VERIFICATION_STAGE_KIND,
    AgentExecutionStructuralValidator,
    NonEmptyTextStructuralValidator,
    StructuralVerificationStage,
)
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine

_MODULE_PATHS = (
    Path("intergrax/runtime/decision_verification_stages/structural.py"),
    Path("intergrax/contracts/agent_execution_validation.py"),
    Path("intergrax/contracts/decision_structural_validation.py"),
)
_FORBIDDEN_FRAGMENTS = (
    "runtime.critic",
    "runtime.nexus",
    "L0Gateway",
    "CriticOrchestrator",
    "Any",
    "cast(",
    "type: ignore",
    "pyright: ignore",
    "getattr",
    "setattr",
    "hasattr",
    "inspect",
    "exec(",
    "eval(",
    "object.__setattr__",
    "dict[str, Any]",
)


@dataclass(frozen=True, slots=True)
class IncidentDecisionPayload:
    recommendation: str


@dataclass(frozen=True, slots=True)
class RecommendationExtractor:
    def extract(self, content: IncidentDecisionPayload) -> str:
        return content.recommendation


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _candidate(
    payload: IncidentDecisionPayload,
) -> CandidateDecision[IncidentDecisionPayload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("incident_resolution"),
        content=payload,
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return CandidateDecision(
        identity=identity,
        artifact=artifact,
        lineage=lineage,
    )


def _agent_execution_candidate(
    execution: AgentExecutionResult,
) -> CandidateDecision[AgentExecutionResult]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="agent", subject="run-1"),
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("agent_execution"),
        content=execution,
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return CandidateDecision(
        identity=identity,
        artifact=artifact,
        lineage=lineage,
    )


def _structural_stage(
    *validators: DecisionStructuralValidator[IncidentDecisionPayload],
) -> StructuralVerificationStage[IncidentDecisionPayload]:
    return StructuralVerificationStage(validators=validators)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_valid_candidate_passes() -> None:
    stage = _structural_stage(
        NonEmptyTextStructuralValidator(
            extractor=RecommendationExtractor(),
            field_label="recommendation",
        ),
    )
    record = await stage.verify(_candidate(IncidentDecisionPayload(recommendation="escalate")))
    assert record.outcome is VerificationStageOutcome.PASSED
    assert record.challenge is None


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_invalid_structural_candidate_challenged() -> None:
    stage = _structural_stage(
        NonEmptyTextStructuralValidator(
            extractor=RecommendationExtractor(),
            field_label="recommendation",
        ),
    )
    record = await stage.verify(_candidate(IncidentDecisionPayload(recommendation="  ")))
    assert record.outcome is VerificationStageOutcome.CHALLENGED
    assert record.challenge is not None


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_exact_proposal_identity() -> None:
    candidate = _candidate(IncidentDecisionPayload(recommendation="escalate"))
    stage = _structural_stage(
        NonEmptyTextStructuralValidator(
            extractor=RecommendationExtractor(),
            field_label="recommendation",
        ),
    )
    record = await stage.verify(candidate)
    assert record.proposal_ref == candidate_decision_ref(candidate)


@pytest.mark.unit
@pytest.mark.gate
def test_execution_class_deterministic() -> None:
    stage = StructuralVerificationStage(validators=())
    assert stage.execution_class is VerificationStageExecutionClass.DETERMINISTIC


@pytest.mark.unit
@pytest.mark.gate
def test_stable_stage_kind() -> None:
    stage = StructuralVerificationStage(validators=())
    assert stage.kind == STRUCTURAL_VERIFICATION_STAGE_KIND


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_deterministic_same_input_same_record() -> None:
    candidate = _candidate(IncidentDecisionPayload(recommendation="escalate"))
    stage = _structural_stage(
        NonEmptyTextStructuralValidator(
            extractor=RecommendationExtractor(),
            field_label="recommendation",
        ),
    )
    first = await stage.verify(candidate)
    second = await stage.verify(candidate)
    assert first == second


@pytest.mark.unit
@pytest.mark.gate
def test_no_critic_imports() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        for fragment in ("runtime.critic", "L0Gateway", "CriticOrchestrator"):
            assert fragment not in source


@pytest.mark.unit
@pytest.mark.gate
def test_no_nexus_ownership_dependency() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "runtime.nexus" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_structural_validation_outcome_rejects_incoherent_construction() -> None:
    requirement = validate_verification_requirement_code(
        "verification.structural.non_empty_text",
    )
    finding = validate_verification_finding_code(
        "verification.structural.non_empty_text",
    )
    failure = StructuralValidationFailure(
        requirement_code=requirement,
        finding_code=finding,
        message="field must be non-empty",
    )
    with pytest.raises(ValueError, match="passed=True cannot include failure"):
        StructuralValidationOutcome(passed=True, failure=failure)
    with pytest.raises(ValueError, match="passed=False requires failure"):
        StructuralValidationOutcome(passed=False)


@pytest.mark.unit
@pytest.mark.gate
def test_structural_validation_outcome_helpers() -> None:
    passed = structural_validation_passed()
    assert passed.passed
    assert passed.failure is None

    requirement = validate_verification_requirement_code(
        "verification.structural.non_empty_text",
    )
    finding = validate_verification_finding_code(
        "verification.structural.non_empty_text",
    )
    failed = structural_validation_failed(
        requirement_code=requirement,
        finding_code=finding,
        message="field must be non-empty",
    )
    assert not failed.passed
    assert failed.failure is not None
    assert failed.failure.message == "field must be non-empty"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_pipeline_deterministic_short_circuit_with_structural_stage() -> None:
    stage = _structural_stage(
        NonEmptyTextStructuralValidator(
            extractor=RecommendationExtractor(),
            field_label="recommendation",
        ),
    )
    registration = VerificationStageRegistration(
        kind=STRUCTURAL_VERIFICATION_STAGE_KIND,
        stage=stage,
        required=True,
    )
    pipeline = VerificationPipeline(registry=verification_stage_registry((registration,)))
    result = await pipeline.verify(_candidate(IncidentDecisionPayload(recommendation="  ")))
    assert result.disposition is VerificationDisposition.CHALLENGED


@pytest.mark.unit
@pytest.mark.gate
def test_migration_parity_agent_execution_validation() -> None:
    contract = AgentContract(
        id="agent-1",
        name="Agent",
        description="test",
        validation_rules=["non_empty_summary"],
    )
    execution = AgentExecutionResult(
        agent_id="agent-1",
        run_id="run-1",
        status=AgentExecutionStatus.COMPLETED,
        summary="",
    )
    engine = NexusValidationEngine()
    legacy = engine.validate(execution, contract=contract)
    neutral = validate_agent_execution(execution, contract=contract)
    assert legacy.valid == neutral.valid
    assert legacy.errors == neutral.errors


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_migration_parity_structural_stage_agent_execution() -> None:
    contract = AgentContract(
        id="agent-1",
        name="Agent",
        description="test",
        validation_rules=["non_empty_summary"],
    )
    execution = AgentExecutionResult(
        agent_id="agent-1",
        run_id="run-1",
        status=AgentExecutionStatus.COMPLETED,
        summary="",
    )
    engine = NexusValidationEngine()
    legacy = engine.validate(execution, contract=contract)
    candidate = _agent_execution_candidate(execution)
    stage = StructuralVerificationStage(
        validators=(
            AgentExecutionStructuralValidator(contract=contract),
        ),
    )
    record = await stage.verify(candidate)
    if legacy.valid:
        assert record.outcome is VerificationStageOutcome.PASSED
    else:
        assert record.outcome is VerificationStageOutcome.CHALLENGED


@pytest.mark.unit
@pytest.mark.gate
def test_forbidden_audit_new_modules() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in source
