# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifact,
    DecisionProposalRef,
    DecisionVersionLineage,
    candidate_decision_ref,
    decision_lineage_ref,
    decision_proposal_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationFinding,
    VerificationStageOutcome,
    VerificationStageRecord,
    validate_verification_finding_code,
    validate_verification_requirement_code,
    validate_verification_stage_kind,
    verification_challenge,
    verification_finding,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStage,
    VerificationStageExecutionClass,
    VerificationStageKind,
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

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "intergrax"
    / "runtime"
    / "decision_verification.py"
)

_FORBIDDEN_IMPORT_FRAGMENTS = (
    "runtime.critic",
    "runtime.nexus",
    "runtime.human",
    "runtime.policy",
    "runtime.governance",
    "DecisionLifecycleHost",
    "CanonicalDecisionLifecycleHost",
    "AuthoritativeAcceptedDecision",
    "AuthoritativeResolutionRecord",
    "DecisionResolution",
)


@dataclass(frozen=True, slots=True)
class IncidentDecisionPayload:
    recommendation: str


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _candidate() -> CandidateDecision[IncidentDecisionPayload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("incident_resolution"),
        content=IncidentDecisionPayload(recommendation="escalate"),
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return CandidateDecision(
        identity=identity,
        artifact=artifact,
        lineage=lineage,
    )


def _registration(
    *,
    kind: str,
    stage: VerificationStage[IncidentDecisionPayload],
    required: bool = True,
) -> VerificationStageRegistration[IncidentDecisionPayload]:
    return VerificationStageRegistration(
        kind=validate_verification_stage_kind(kind),
        stage=stage,
        required=required,
    )


@dataclass(frozen=True, slots=True)
class PassedStage:
    kind: VerificationStageKind
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=self.kind,
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(frozen=True, slots=True)
class ChallengedStage:
    kind: VerificationStageKind
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        proposal_ref = candidate_decision_ref(candidate)
        finding = verification_finding(
            code=validate_verification_finding_code("verification.test.challenged"),
            message="stage challenged",
        )
        challenge = verification_challenge(
            proposal_ref=proposal_ref,
            stage=self.kind,
            requirement_code=validate_verification_requirement_code(
                "verification.test.requirement",
            ),
            finding=finding,
        )
        return verification_stage_record(
            proposal_ref=proposal_ref,
            stage=self.kind,
            outcome=VerificationStageOutcome.CHALLENGED,
            challenge=challenge,
        )


@dataclass(frozen=True, slots=True)
class MismatchProposalStage:
    kind: VerificationStageKind
    wrong_ref: DecisionProposalRef
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        return verification_stage_record(
            proposal_ref=self.wrong_ref,
            stage=self.kind,
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(frozen=True, slots=True)
class MismatchKindStage:
    kind: VerificationStageKind
    returned_kind: VerificationStageKind
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=self.returned_kind,
            outcome=VerificationStageOutcome.PASSED,
        )


def _pipeline(
    *stages: VerificationStage[IncidentDecisionPayload],
) -> VerificationPipeline[IncidentDecisionPayload]:
    registrations = tuple(
        _registration(kind=stage.kind, stage=stage) for stage in stages
    )
    return VerificationPipeline(registry=verification_stage_registry(registrations))


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_one_passed_stage_produces_passed_result() -> None:
    candidate = _candidate()
    pipeline = _pipeline(PassedStage(kind=validate_verification_stage_kind("schema")))
    result = await pipeline.verify(candidate)
    assert result.disposition is VerificationDisposition.PASSED
    assert len(result.stage_records) == 1


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_multiple_passed_stages_produce_passed_result() -> None:
    candidate = _candidate()
    pipeline = _pipeline(
        PassedStage(kind=validate_verification_stage_kind("schema")),
        PassedStage(kind=validate_verification_stage_kind("rules")),
    )
    result = await pipeline.verify(candidate)
    assert result.disposition is VerificationDisposition.PASSED
    assert len(result.stage_records) == 2


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_challenged_stage_produces_challenged_result() -> None:
    candidate = _candidate()
    pipeline = _pipeline(
        ChallengedStage(kind=validate_verification_stage_kind("schema")),
    )
    result = await pipeline.verify(candidate)
    assert result.disposition is VerificationDisposition.CHALLENGED
    assert result.stage_records[0].outcome is VerificationStageOutcome.CHALLENGED


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_result_exact_proposal_identity() -> None:
    candidate = _candidate()
    pipeline = _pipeline(PassedStage(kind=validate_verification_stage_kind("schema")))
    result = await pipeline.verify(candidate)
    assert result.proposal_ref == candidate_decision_ref(candidate)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_stage_record_proposal_mismatch_rejected() -> None:
    candidate = _candidate()
    other_identity = replace(
        candidate.identity,
        version=next_decision_version(candidate.identity.version),
    )
    wrong_ref = decision_proposal_ref(
        identity=other_identity,
        lineage_ref=decision_lineage_ref(other_identity.version),
    )
    pipeline = _pipeline(
        MismatchProposalStage(
            kind=validate_verification_stage_kind("schema"),
            wrong_ref=wrong_ref,
        ),
    )
    with pytest.raises(ValueError, match="proposal reference"):
        await pipeline.verify(candidate)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_stage_record_kind_mismatch_rejected() -> None:
    candidate = _candidate()
    pipeline = _pipeline(
        MismatchKindStage(
            kind=validate_verification_stage_kind("schema"),
            returned_kind=validate_verification_stage_kind("rules"),
        ),
    )
    with pytest.raises(ValueError, match="registered stage kind"):
        await pipeline.verify(candidate)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_candidate_remains_unchanged() -> None:
    candidate = _candidate()
    before = replace(candidate)
    pipeline = _pipeline(PassedStage(kind=validate_verification_stage_kind("schema")))
    _ = await pipeline.verify(candidate)
    assert candidate == before


@pytest.mark.unit
@pytest.mark.gate
def test_no_decision_lifecycle_invocation_boundary() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for fragment in _FORBIDDEN_IMPORT_FRAGMENTS:
        assert fragment not in source


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_registry_original_unchanged_after_verify() -> None:
    stage = PassedStage(kind=validate_verification_stage_kind("schema"))
    registration = _registration(kind="schema", stage=stage)
    registry = verification_stage_registry((registration,))
    pipeline = VerificationPipeline(registry=registry)
    candidate = _candidate()
    _ = await pipeline.verify(candidate)
    assert registry.registrations == (registration,)
