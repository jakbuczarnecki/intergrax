# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionId,
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
    decision_proposal_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
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
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.decision_verification import (
    VerificationPipeline,
    VerificationPipelineEmptyResultError,
)
from intergrax.runtime.decision_verification_observability import (
    DECISION_VERIFICATION_COMPLETED_EVENT_KIND,
    DECISION_VERIFICATION_PROBABILISTIC_SKIPPED_EVENT_KIND,
    DECISION_VERIFICATION_SIGNAL_PAYLOAD_SCHEMA_ID,
    DECISION_VERIFICATION_STAGE_COMPLETED_EVENT_KIND,
    DECISION_VERIFICATION_STAGE_UNAVAILABLE_EVENT_KIND,
    DECISION_VERIFICATION_STARTED_EVENT_KIND,
    CanonicalRuntimeEventVerificationObserver,
    DecisionVerificationSignalPayloadV1,
    DecisionVerificationSignalPhase,
    DecisionVerificationSkipReason,
    DecisionVerificationUnavailableReason,
    register_decision_verification_domain_signals,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_kind_registry import (
    clear_event_kind_registry,
    get_event_kind_entry,
    list_registered_event_kinds,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent
from testing_support.runtime_events import emit_context_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SECRET_MARKER = "SUPER_SECRET_CANDIDATE_TEXT"
_OBSERVABILITY_MODULE = (
    Path(__file__).resolve().parents[3]
    / "intergrax"
    / "runtime"
    / "decision_verification_observability.py"
)
_FORBIDDEN_PATTERNS = (
    ": Any",
    "dict[str, Any]",
    "cast(",
    "type: ignore",
    "pyright: ignore",
    "getattr(",
    "setattr(",
    "hasattr(",
    "inspect.",
    "exec(",
    "eval(",
    "object.__setattr__",
)


@dataclass(frozen=True, slots=True)
class IncidentDecisionPayload:
    recommendation: str


def _execution_lineage(
    *,
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
) -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id() if task_id is None else task_id,
        run_id=mint_run_id() if run_id is None else run_id,
        attempt_id=mint_attempt_id() if attempt_id is None else attempt_id,
        execution_id=mint_execution_id() if execution_id is None else execution_id,
    )


def _candidate(
    *,
    decision_id: DecisionId | None = None,
    execution: DecisionExecutionLineage | None = None,
    secret_in_artifact: bool = False,
) -> CandidateDecision[IncidentDecisionPayload]:
    lineage_execution = execution or _execution_lineage()
    identity = DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id="tenant-a",
        execution=lineage_execution,
    )
    recommendation = _SECRET_MARKER if secret_in_artifact else "escalate"
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("incident_resolution"),
        content=IncidentDecisionPayload(recommendation=recommendation),
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
    secret_in_message: bool = False

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        proposal_ref = candidate_decision_ref(candidate)
        message = (
            f"stage challenged {_SECRET_MARKER}"
            if self.secret_in_message
            else "stage challenged"
        )
        finding = verification_finding(
            code=validate_verification_finding_code("verification.test.challenged"),
            message=message,
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
class UnavailableStage:
    kind: VerificationStageKind
    execution_class: VerificationStageExecutionClass

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        raise VerificationStageUnavailableError("stage unavailable")


@dataclass(frozen=True, slots=True)
class MismatchProposalStage:
    kind: VerificationStageKind
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        other_identity = replace(
            candidate.identity,
            decision_id=mint_decision_id(),
        )
        wrong_ref = decision_proposal_ref(
            identity=other_identity,
            lineage_ref=decision_lineage_ref(other_identity.version),
        )
        return verification_stage_record(
            proposal_ref=wrong_ref,
            stage=self.kind,
            outcome=VerificationStageOutcome.PASSED,
        )


def _pipeline(
    *stages: VerificationStage[IncidentDecisionPayload],
    observer: CanonicalRuntimeEventVerificationObserver[IncidentDecisionPayload] | None = None,
) -> VerificationPipeline[IncidentDecisionPayload]:
    registrations = tuple(
        _registration(kind=stage.kind, stage=stage) for stage in stages
    )
    return VerificationPipeline(
        registry=verification_stage_registry(registrations),
        observer=observer,
    )


def _pipeline_from_registrations(
    *registrations: VerificationStageRegistration[IncidentDecisionPayload],
    observer: CanonicalRuntimeEventVerificationObserver[IncidentDecisionPayload] | None = None,
) -> VerificationPipeline[IncidentDecisionPayload]:
    return VerificationPipeline(
        registry=verification_stage_registry(registrations),
        observer=observer,
    )


def _observer_for_candidate(
    candidate: CandidateDecision[IncidentDecisionPayload],
) -> tuple[RuntimeEventBus, CanonicalRuntimeEventVerificationObserver[IncidentDecisionPayload]]:
    bus = RuntimeEventBus(record_history=True)
    execution = candidate.identity.execution
    ctx = emit_context_test_identity(
        task_id=execution.task_id,
        run_id=execution.run_id,
        attempt_id=execution.attempt_id,
        execution_id=execution.execution_id,
        tenant_id=candidate.identity.tenant_id,
        bus=bus,
    )
    return bus, CanonicalRuntimeEventVerificationObserver(ctx=ctx)


def _event_kinds(events: list[RuntimeEvent]) -> list[str]:
    return [event.event_kind or "" for event in events]


def _payload_data(event: RuntimeEvent) -> dict[str, object]:
    payload = event.payload
    if type(payload) is not dict:
        raise TypeError("expected dict payload envelope on RuntimeEvent")
    data = payload.get("data")
    if type(data) is not dict:
        raise TypeError("expected typed payload data dict")
    return data


def _serialized_events(events: list[RuntimeEvent]) -> str:
    return json.dumps([event.model_dump(mode="json") for event in events])


@pytest.fixture(autouse=True)
def _register_verification_domain_signals() -> Iterator[None]:
    clear_event_kind_registry()
    register_decision_verification_domain_signals()
    yield
    clear_event_kind_registry()


def test_payload_rejects_invalid_disposition() -> None:
    with pytest.raises(ValueError, match="overall_disposition"):
        DecisionVerificationSignalPayloadV1(
            phase=DecisionVerificationSignalPhase.COMPLETED.value,
            decision_id="decision_" + "a" * 32,
            decision_version=1,
            branch_id="main",
            overall_disposition="invalid",
            executed_stage_count=1,
            challenged_stage_count=0,
        )


def test_payload_rejects_invalid_stage_outcome() -> None:
    with pytest.raises(ValueError, match="stage_outcome"):
        DecisionVerificationSignalPayloadV1(
            phase=DecisionVerificationSignalPhase.STAGE_COMPLETED.value,
            decision_id="decision_" + "a" * 32,
            decision_version=1,
            branch_id="main",
            stage_kind="schema",
            execution_class="deterministic",
            required=True,
            stage_outcome="invalid",
        )


def test_payload_rejects_invalid_decision_version() -> None:
    with pytest.raises(ValueError, match="decision_version"):
        DecisionVerificationSignalPayloadV1(
            phase=DecisionVerificationSignalPhase.STARTED.value,
            decision_id="decision_" + "a" * 32,
            decision_version=0,
            branch_id="main",
            stage_count=1,
        )


def test_payload_rejects_started_with_stage_fields() -> None:
    with pytest.raises(ValueError, match="started phase"):
        DecisionVerificationSignalPayloadV1(
            phase=DecisionVerificationSignalPhase.STARTED.value,
            decision_id="decision_" + "a" * 32,
            decision_version=1,
            branch_id="main",
            stage_count=1,
            stage_kind="schema",
        )


def test_payload_redact_returns_equivalent_safe_payload() -> None:
    payload = DecisionVerificationSignalPayloadV1(
        phase=DecisionVerificationSignalPhase.STARTED.value,
        decision_id="decision_" + "a" * 32,
        decision_version=1,
        branch_id="main",
        stage_count=2,
    )
    assert payload.redact() == payload


def test_all_verification_event_kinds_registered() -> None:
    kinds = list_registered_event_kinds()
    for kind in (
        DECISION_VERIFICATION_STARTED_EVENT_KIND,
        DECISION_VERIFICATION_STAGE_COMPLETED_EVENT_KIND,
        DECISION_VERIFICATION_STAGE_UNAVAILABLE_EVENT_KIND,
        DECISION_VERIFICATION_PROBABILISTIC_SKIPPED_EVENT_KIND,
        DECISION_VERIFICATION_COMPLETED_EVENT_KIND,
    ):
        assert kind in kinds
        entry = get_event_kind_entry(kind)
        assert entry is not None
        assert entry.payload_schema_id == DECISION_VERIFICATION_SIGNAL_PAYLOAD_SCHEMA_ID


@pytest.mark.asyncio
async def test_observer_absent_matches_baseline_semantics() -> None:
    candidate = _candidate()
    baseline = VerificationPipeline(
        registry=verification_stage_registry(
            (
                _registration(
                    kind="schema",
                    stage=PassedStage(kind=validate_verification_stage_kind("schema")),
                ),
            ),
        ),
    )
    observed = _pipeline(
        PassedStage(kind=validate_verification_stage_kind("schema")),
        observer=None,
    )
    baseline_result = await baseline.verify(candidate)
    observed_result = await observed.verify(candidate)
    assert observed_result == baseline_result


@pytest.mark.asyncio
async def test_pass_flow_emits_started_stage_completed_completed() -> None:
    candidate = _candidate()
    bus, observer = _observer_for_candidate(candidate)
    pipeline = _pipeline(
        PassedStage(kind=validate_verification_stage_kind("schema")),
        PassedStage(kind=validate_verification_stage_kind("rules")),
        observer=observer,
    )
    result = await pipeline.verify(candidate)
    assert result.disposition is VerificationDisposition.PASSED
    assert _event_kinds(bus.history) == [
        DECISION_VERIFICATION_STARTED_EVENT_KIND,
        DECISION_VERIFICATION_STAGE_COMPLETED_EVENT_KIND,
        DECISION_VERIFICATION_STAGE_COMPLETED_EVENT_KIND,
        DECISION_VERIFICATION_COMPLETED_EVENT_KIND,
    ]
    started = _payload_data(bus.history[0])
    completed = _payload_data(bus.history[-1])
    proposal_ref = candidate_decision_ref(candidate)
    assert started["decision_id"] == str(proposal_ref.identity.decision_id)
    assert started["decision_version"] == proposal_ref.identity.version.value
    assert started["branch_id"] == str(proposal_ref.lineage_ref.branch_id)
    assert started["stage_count"] == 2
    assert completed["overall_disposition"] == VerificationDisposition.PASSED.value
    assert bus.history[0].task_id == candidate.identity.execution.task_id
    assert bus.history[0].execution_id == candidate.identity.execution.execution_id


@pytest.mark.asyncio
async def test_challenge_flow_emits_probabilistic_skipped() -> None:
    candidate = _candidate()
    bus, observer = _observer_for_candidate(candidate)
    pipeline = _pipeline_from_registrations(
        _registration(
            kind="rules",
            stage=ChallengedStage(kind=validate_verification_stage_kind("rules")),
        ),
        _registration(
            kind="schema",
            stage=PassedStage(kind=validate_verification_stage_kind("schema")),
        ),
        _registration(
            kind="semantic",
            stage=PassedStage(
                kind=validate_verification_stage_kind("semantic"),
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
            ),
        ),
        observer=observer,
    )
    result = await pipeline.verify(candidate)
    assert result.disposition is VerificationDisposition.CHALLENGED
    assert _event_kinds(bus.history) == [
        DECISION_VERIFICATION_STARTED_EVENT_KIND,
        DECISION_VERIFICATION_STAGE_COMPLETED_EVENT_KIND,
        DECISION_VERIFICATION_STAGE_COMPLETED_EVENT_KIND,
        DECISION_VERIFICATION_PROBABILISTIC_SKIPPED_EVENT_KIND,
        DECISION_VERIFICATION_COMPLETED_EVENT_KIND,
    ]
    challenged_payload = _payload_data(bus.history[1])
    assert challenged_payload["stage_outcome"] == VerificationStageOutcome.CHALLENGED.value
    skipped_payload = _payload_data(bus.history[3])
    assert skipped_payload["skipped_stage_count"] == 1
    assert skipped_payload["skip_reason"] == (
        DecisionVerificationSkipReason.DETERMINISTIC_CHALLENGE.value
    )


@pytest.mark.asyncio
async def test_required_unavailable_emits_unavailable_and_challenged_completion() -> None:
    candidate = _candidate()
    bus, observer = _observer_for_candidate(candidate)
    pipeline = _pipeline_from_registrations(
        _registration(
            kind="schema",
            stage=UnavailableStage(
                kind=validate_verification_stage_kind("schema"),
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
            ),
        ),
        _registration(
            kind="semantic",
            stage=PassedStage(
                kind=validate_verification_stage_kind("semantic"),
                execution_class=VerificationStageExecutionClass.PROBABILISTIC,
            ),
        ),
        observer=observer,
    )
    result = await pipeline.verify(candidate)
    assert result.disposition is VerificationDisposition.CHALLENGED
    assert _event_kinds(bus.history) == [
        DECISION_VERIFICATION_STARTED_EVENT_KIND,
        DECISION_VERIFICATION_STAGE_UNAVAILABLE_EVENT_KIND,
        DECISION_VERIFICATION_STAGE_COMPLETED_EVENT_KIND,
        DECISION_VERIFICATION_PROBABILISTIC_SKIPPED_EVENT_KIND,
        DECISION_VERIFICATION_COMPLETED_EVENT_KIND,
    ]
    unavailable_payload = _payload_data(bus.history[1])
    assert unavailable_payload["required"] is True
    assert unavailable_payload["unavailable_reason_category"] == (
        DecisionVerificationUnavailableReason.REQUIRED_UNAVAILABLE.value
    )
    completed_payload = _payload_data(bus.history[2])
    assert completed_payload["stage_outcome"] == VerificationStageOutcome.CHALLENGED.value


@pytest.mark.asyncio
async def test_optional_unavailable_visible_without_passed_record() -> None:
    candidate = _candidate()
    bus, observer = _observer_for_candidate(candidate)
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
        observer=observer,
    )
    result = await pipeline.verify(candidate)
    assert result.disposition is VerificationDisposition.PASSED
    assert len(result.stage_records) == 1
    assert _event_kinds(bus.history) == [
        DECISION_VERIFICATION_STARTED_EVENT_KIND,
        DECISION_VERIFICATION_STAGE_COMPLETED_EVENT_KIND,
        DECISION_VERIFICATION_STAGE_UNAVAILABLE_EVENT_KIND,
        DECISION_VERIFICATION_COMPLETED_EVENT_KIND,
    ]
    unavailable_payload = _payload_data(bus.history[2])
    assert unavailable_payload["required"] is False
    assert unavailable_payload["unavailable_reason_category"] is None


@pytest.mark.asyncio
async def test_malformed_stage_result_does_not_emit_stage_completed() -> None:
    candidate = _candidate()
    bus, observer = _observer_for_candidate(candidate)
    pipeline = _pipeline(
        MismatchProposalStage(kind=validate_verification_stage_kind("schema")),
        observer=observer,
    )
    with pytest.raises(ValueError, match="proposal reference"):
        await pipeline.verify(candidate)
    assert _event_kinds(bus.history) == [DECISION_VERIFICATION_STARTED_EVENT_KIND]


@pytest.mark.asyncio
async def test_all_optional_unavailable_emits_started_and_unavailable_only() -> None:
    candidate = _candidate()
    bus, observer = _observer_for_candidate(candidate)
    pipeline = _pipeline_from_registrations(
        _registration(
            kind="schema",
            stage=UnavailableStage(
                kind=validate_verification_stage_kind("schema"),
                execution_class=VerificationStageExecutionClass.DETERMINISTIC,
            ),
            required=False,
        ),
        observer=observer,
    )
    with pytest.raises(VerificationPipelineEmptyResultError):
        await pipeline.verify(candidate)
    assert _event_kinds(bus.history) == [
        DECISION_VERIFICATION_STARTED_EVENT_KIND,
        DECISION_VERIFICATION_STAGE_UNAVAILABLE_EVENT_KIND,
    ]


@pytest.mark.asyncio
async def test_no_private_content_in_serialized_events() -> None:
    candidate = _candidate(secret_in_artifact=True)
    bus, observer = _observer_for_candidate(candidate)
    pipeline = _pipeline(
        ChallengedStage(
            kind=validate_verification_stage_kind("schema"),
            secret_in_message=True,
        ),
        observer=observer,
    )
    await pipeline.verify(candidate)
    serialized = _serialized_events(bus.history)
    assert _SECRET_MARKER not in serialized


@pytest.mark.asyncio
async def test_concurrent_verification_identities_do_not_cross() -> None:
    execution_a = _execution_lineage()
    execution_b = _execution_lineage()
    candidate_a = _candidate(decision_id=mint_decision_id(), execution=execution_a)
    candidate_b = _candidate(decision_id=mint_decision_id(), execution=execution_b)
    bus_a, observer_a = _observer_for_candidate(candidate_a)
    bus_b, observer_b = _observer_for_candidate(candidate_b)
    pipeline_a = _pipeline(
        PassedStage(kind=validate_verification_stage_kind("schema")),
        observer=observer_a,
    )
    pipeline_b = _pipeline(
        PassedStage(kind=validate_verification_stage_kind("schema")),
        observer=observer_b,
    )
    result_a, result_b = await asyncio.gather(
        pipeline_a.verify(candidate_a),
        pipeline_b.verify(candidate_b),
    )
    assert result_a.disposition is VerificationDisposition.PASSED
    assert result_b.disposition is VerificationDisposition.PASSED
    for event in bus_a.history:
        assert _payload_data(event)["decision_id"] == str(candidate_a.identity.decision_id)
    for event in bus_b.history:
        assert _payload_data(event)["decision_id"] == str(candidate_b.identity.decision_id)


def test_observability_module_forbidden_patterns_absent() -> None:
    source = _OBSERVABILITY_MODULE.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_PATTERNS:
        assert pattern not in source
