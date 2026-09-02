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
    VerificationStageUnavailableError,
    verification_stage_registry,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.decision_verification import (
    VerificationPipeline,
    VerificationPipelineEmptyResultError,
)

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


def _pipeline_from_registrations(
    *registrations: VerificationStageRegistration[IncidentDecisionPayload],
) -> VerificationPipeline[IncidentDecisionPayload]:
    return VerificationPipeline(registry=verification_stage_registry(registrations))


@dataclass(frozen=True, slots=True)
class RecordingStage:
    kind: VerificationStageKind
    execution_class: VerificationStageExecutionClass
    trace: list[VerificationStageKind]

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        self.trace.append(self.kind)
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=self.kind,
            outcome=VerificationStageOutcome.PASSED,
        )


def _recording_stage(
    *,
    kind: str,
    execution_class: VerificationStageExecutionClass,
    trace: list[VerificationStageKind],
) -> RecordingStage:
    return RecordingStage(
        kind=validate_verification_stage_kind(kind),
        execution_class=execution_class,
        trace=trace,
    )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_deterministic_stages_execute_before_probabilistic() -> None:
    trace: list[VerificationStageKind] = []
    registrations = (
        _registration(
            kind="semantic_1",
            stage=_recording_stage(
                kind="semantic_1",
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
                trace=trace,
            ),
        ),
        _registration(
            kind="schema",
            stage=_recording_stage(
                kind="schema",
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
                trace=trace,
            ),
        ),
        _registration(
            kind="domain_llm",
            stage=_recording_stage(
                kind="domain_llm",
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
                trace=trace,
            ),
        ),
        _registration(
            kind="rules",
            stage=_recording_stage(
                kind="rules",
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
                trace=trace,
            ),
        ),
    )
    pipeline = _pipeline_from_registrations(*registrations)
    _ = await pipeline.verify(_candidate())
    assert trace == [
        validate_verification_stage_kind("rules"),
        validate_verification_stage_kind("schema"),
        validate_verification_stage_kind("domain_llm"),
        validate_verification_stage_kind("semantic_1"),
    ]


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_stable_order_within_deterministic_group() -> None:
    trace: list[VerificationStageKind] = []
    registrations = (
        _registration(
            kind="rules",
            stage=_recording_stage(
                kind="rules",
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
                trace=trace,
            ),
        ),
        _registration(
            kind="schema",
            stage=_recording_stage(
                kind="schema",
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
                trace=trace,
            ),
        ),
    )
    pipeline = _pipeline_from_registrations(*registrations)
    _ = await pipeline.verify(_candidate())
    assert trace == [
        validate_verification_stage_kind("rules"),
        validate_verification_stage_kind("schema"),
    ]


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_stable_order_within_probabilistic_group() -> None:
    trace: list[VerificationStageKind] = []
    registrations = (
        _registration(
            kind="domain_llm",
            stage=_recording_stage(
                kind="domain_llm",
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
                trace=trace,
            ),
        ),
        _registration(
            kind="semantic_1",
            stage=_recording_stage(
                kind="semantic_1",
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
                trace=trace,
            ),
        ),
    )
    pipeline = _pipeline_from_registrations(*registrations)
    _ = await pipeline.verify(_candidate())
    assert trace == [
        validate_verification_stage_kind("domain_llm"),
        validate_verification_stage_kind("semantic_1"),
    ]


@dataclass(frozen=True, slots=True)
class RecordingChallengedStage:
    kind: VerificationStageKind
    execution_class: VerificationStageExecutionClass
    trace: list[VerificationStageKind]

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        self.trace.append(self.kind)
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


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_deterministic_challenge_prevents_probabilistic_execution() -> None:
    trace: list[VerificationStageKind] = []
    registrations = (
        _registration(
            kind="schema",
            stage=RecordingChallengedStage(
                kind=validate_verification_stage_kind("schema"),
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
                trace=trace,
            ),
        ),
        _registration(
            kind="rules",
            stage=_recording_stage(
                kind="rules",
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
                trace=trace,
            ),
        ),
        _registration(
            kind="semantic_1",
            stage=_recording_stage(
                kind="semantic_1",
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
                trace=trace,
            ),
        ),
    )
    pipeline = _pipeline_from_registrations(*registrations)
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.CHALLENGED
    assert trace == [
        validate_verification_stage_kind("rules"),
        validate_verification_stage_kind("schema"),
    ]


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_deterministic_stages_all_contribute_before_probabilistic_short_circuit() -> (
    None
):
    trace: list[VerificationStageKind] = []
    registrations = (
        _registration(
            kind="rules",
            stage=RecordingChallengedStage(
                kind=validate_verification_stage_kind("rules"),
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
                trace=trace,
            ),
        ),
        _registration(
            kind="schema",
            stage=_recording_stage(
                kind="schema",
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
                trace=trace,
            ),
        ),
        _registration(
            kind="semantic_1",
            stage=_recording_stage(
                kind="semantic_1",
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
                trace=trace,
            ),
        ),
    )
    pipeline = _pipeline_from_registrations(*registrations)
    result = await pipeline.verify(_candidate())
    assert len(result.stage_records) == 2
    assert trace == [
        validate_verification_stage_kind("rules"),
        validate_verification_stage_kind("schema"),
    ]


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_result_records_preserve_execution_order() -> None:
    trace: list[VerificationStageKind] = []
    registrations = (
        _registration(
            kind="semantic_1",
            stage=_recording_stage(
                kind="semantic_1",
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
                trace=trace,
            ),
        ),
        _registration(
            kind="schema",
            stage=_recording_stage(
                kind="schema",
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
                trace=trace,
            ),
        ),
    )
    pipeline = _pipeline_from_registrations(*registrations)
    result = await pipeline.verify(_candidate())
    assert tuple(record.stage for record in result.stage_records) == tuple(trace)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_registry_unchanged_after_ordered_execution() -> None:
    trace: list[VerificationStageKind] = []
    registration = _registration(
        kind="schema",
        stage=_recording_stage(
            kind="schema",
            execution_class=VerificationStageExecutionClass.DETERMINISTIC,
            trace=trace,
        ),
    )
    registry = verification_stage_registry((registration,))
    pipeline = VerificationPipeline(registry=registry)
    _ = await pipeline.verify(_candidate())
    assert registry.registrations == (registration,)


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


@dataclass(frozen=True, slots=True)
class UnavailableStage:
    kind: VerificationStageKind
    execution_class: VerificationStageExecutionClass

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        raise VerificationStageUnavailableError("stage unavailable")


@dataclass(frozen=True, slots=True)
class ProgrammingErrorStage:
    kind: VerificationStageKind
    execution_class: VerificationStageExecutionClass

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        raise TypeError("programming defect")


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_required_deterministic_unavailable_produces_challenged() -> None:
    pipeline = _pipeline(
        UnavailableStage(
            kind=validate_verification_stage_kind("schema"),
            execution_class=VerificationStageExecutionClass.DETERMINISTIC,
        ),
    )
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.CHALLENGED
    assert result.stage_records[0].outcome is VerificationStageOutcome.CHALLENGED


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_required_probabilistic_unavailable_produces_challenged() -> None:
    pipeline = _pipeline_from_registrations(
        _registration(
            kind="schema",
            stage=PassedStage(kind=validate_verification_stage_kind("schema")),
        ),
        _registration(
            kind="semantic",
            stage=UnavailableStage(
                kind=validate_verification_stage_kind("semantic"),
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
            ),
        ),
    )
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.CHALLENGED
    assert len(result.stage_records) == 2
    assert result.stage_records[1].stage == validate_verification_stage_kind("semantic")


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_unavailable_required_stage_never_omitted() -> None:
    pipeline = _pipeline(
        UnavailableStage(
            kind=validate_verification_stage_kind("schema"),
            execution_class=VerificationStageExecutionClass.DETERMINISTIC,
        ),
    )
    result = await pipeline.verify(_candidate())
    assert len(result.stage_records) == 1


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_unavailable_required_stage_never_passed() -> None:
    pipeline = _pipeline(
        UnavailableStage(
            kind=validate_verification_stage_kind("schema"),
            execution_class=VerificationStageExecutionClass.DETERMINISTIC,
        ),
    )
    result = await pipeline.verify(_candidate())
    assert all(
        record.outcome is not VerificationStageOutcome.PASSED
        for record in result.stage_records
    )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_required_deterministic_unavailable_blocks_probabilistic() -> None:
    trace: list[VerificationStageKind] = []
    registrations = (
        _registration(
            kind="schema",
            stage=UnavailableStage(
                kind=validate_verification_stage_kind("schema"),
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
            ),
        ),
        _registration(
            kind="semantic",
            stage=_recording_stage(
                kind="semantic",
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
                trace=trace,
            ),
        ),
    )
    pipeline = _pipeline_from_registrations(*registrations)
    _ = await pipeline.verify(_candidate())
    assert trace == []


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_optional_unavailable_stage_skipped_without_passed_record() -> None:
    pipeline = _pipeline_from_registrations(
        _registration(
            kind="schema",
            stage=UnavailableStage(
                kind=validate_verification_stage_kind("schema"),
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
            ),
            required=False,
        ),
        _registration(
            kind="rules",
            stage=PassedStage(kind=validate_verification_stage_kind("rules")),
        ),
    )
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.PASSED
    assert len(result.stage_records) == 1
    assert result.stage_records[0].stage == validate_verification_stage_kind("rules")


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_optional_unavailable_with_other_passed_stage_valid_passed() -> None:
    pipeline = _pipeline_from_registrations(
        _registration(
            kind="semantic",
            stage=UnavailableStage(
                kind=validate_verification_stage_kind("semantic"),
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
            ),
            required=False,
        ),
        _registration(
            kind="schema",
            stage=PassedStage(kind=validate_verification_stage_kind("schema")),
        ),
    )
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.PASSED
    assert len(result.stage_records) == 1


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_all_optional_unavailable_fails_explicitly() -> None:
    pipeline = _pipeline_from_registrations(
        _registration(
            kind="schema",
            stage=UnavailableStage(
                kind=validate_verification_stage_kind("schema"),
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
            ),
            required=False,
        ),
    )
    with pytest.raises(VerificationPipelineEmptyResultError):
        await pipeline.verify(_candidate())


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_unexpected_programming_exception_propagates() -> None:
    pipeline = _pipeline(
        ProgrammingErrorStage(
            kind=validate_verification_stage_kind("schema"),
            execution_class=VerificationStageExecutionClass.DETERMINISTIC,
        ),
    )
    with pytest.raises(TypeError, match="programming defect"):
        await pipeline.verify(_candidate())


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_generated_unavailable_challenge_exact_proposal_identity() -> None:
    candidate = _candidate()
    pipeline = _pipeline(
        UnavailableStage(
            kind=validate_verification_stage_kind("schema"),
            execution_class=VerificationStageExecutionClass.DETERMINISTIC,
        ),
    )
    result = await pipeline.verify(candidate)
    record = result.stage_records[0]
    assert record.proposal_ref == candidate_decision_ref(candidate)
    assert record.challenge is not None
    assert record.challenge.proposal_ref == candidate_decision_ref(candidate)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_generated_unavailable_challenge_exact_stage_kind() -> None:
    kind = validate_verification_stage_kind("schema")
    pipeline = _pipeline(
        UnavailableStage(
            kind=kind,
            execution_class=VerificationStageExecutionClass.DETERMINISTIC,
        ),
    )
    result = await pipeline.verify(_candidate())
    record = result.stage_records[0]
    assert record.stage == kind
    assert record.challenge is not None
    assert record.challenge.stage == kind
    assert record.challenge.requirement_code == validate_verification_requirement_code(
        "verification.stage.required_unavailable",
    )
