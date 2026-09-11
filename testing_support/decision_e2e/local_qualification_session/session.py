# © Artur Czarnecki. All rights reserved.

"""Local qualification session controller."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from testing_support.decision_e2e.local_qualification_session.attempt_evidence import (
    assess_session_attempt_evidence,
)
from testing_support.decision_e2e.local_qualification_session.atomic_io import atomic_write_json
from testing_support.decision_e2e.local_qualification_session.checkpoint import (
    IllegalSessionTransitionError,
    SessionCheckpoint,
    load_checkpoint,
    persist_checkpoint,
    transition_state,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QUALIFICATION_SESSION_SCHEMA_VERSION,
    AttemptEvidenceStatus,
    CanonicalRunRecord,
    EvidenceCompletenessStatus,
    QualificationIdentityStatus,
    QualificationPreconditionResult,
    QualificationRuntimeIdentity,
    QualificationSessionState,
    QualificationSpec,
    SessionIntegrityReport,
    SourceDriftReport,
    SourceFingerprintSnapshot,
    TraceReadbackStatus,
)
from testing_support.decision_e2e.local_qualification_session.finalization import (
    FinalizationContext,
    aggregate_trace_readback_status,
    finalize_with_failure_capture,
)
from testing_support.decision_e2e.local_qualification_session.identity import (
    evaluate_model_identity_match,
    evaluate_preconditions,
    evaluate_runtime_identity_match,
)
from testing_support.decision_e2e.local_qualification_session.run_registry import RunRegistry
from testing_support.decision_e2e.local_qualification_session.source_fingerprint import (
    compare_source_snapshots,
)
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)


@dataclass(frozen=True, slots=True)
class SessionStartResult:
    state: QualificationSessionState
    precondition: QualificationPreconditionResult
    checkpoint: SessionCheckpoint | None


@dataclass(frozen=True, slots=True)
class RunPersistenceResult:
    state: QualificationSessionState
    checkpoint: SessionCheckpoint


class LocalQualificationSession:
    def __init__(
        self,
        *,
        session_dir: Path,
        spec: QualificationSpec,
        frozen_source: SourceFingerprintSnapshot,
        task_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._session_dir = session_dir
        self._spec = spec
        self._frozen_source = frozen_source
        self._task_id = task_id or f"DS-E2E-15J-L1-{uuid.uuid4().hex[:8]}"
        self._session_id = session_id or f"qual-{uuid.uuid4().hex[:12]}"
        self._registry = RunRegistry(spec.run_count)
        self._state = QualificationSessionState.CREATED
        self._integrity_failures: list[str] = []
        existing = load_checkpoint(session_dir)
        if existing is not None:
            self._hydrate_from_checkpoint(existing)

    @property
    def state(self) -> QualificationSessionState:
        return self._state

    @property
    def spec(self) -> QualificationSpec:
        return self._spec

    @property
    def frozen_source(self) -> SourceFingerprintSnapshot:
        return self._frozen_source

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def task_id(self) -> str:
        return self._task_id

    def completed_run_indices(self) -> tuple[int, ...]:
        return self._registry.completed_indices()

    def pending_run_indices(self) -> tuple[int, ...]:
        completed = set(self._registry.completed_indices())
        return tuple(index for index in range(self._spec.run_count) if index not in completed)

    def _hydrate_from_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        self._state = checkpoint.state
        self._session_id = checkpoint.session_id
        self._task_id = checkpoint.task_id
        self._integrity_failures = list(checkpoint.integrity_failures)
        runs_path = self._session_dir / "runs.json"
        if runs_path.is_file():
            payload = json.loads(runs_path.read_text(encoding="utf-8"))
            runs = payload.get("runs")
            if isinstance(runs, list):
                for item in runs:
                    if not isinstance(item, dict):
                        continue
                    run_index = item.get("run_index")
                    run_id = item.get("run_id")
                    trace_events = item.get("trace_events")
                    if not isinstance(run_index, int) or not isinstance(run_id, str):
                        continue
                    events: tuple[dict[str, object], ...] = ()
                    if isinstance(trace_events, list):
                        events = tuple(
                            dict(event) for event in trace_events if isinstance(event, dict)
                        )
                    self._registry.register(
                        CanonicalRunRecord(
                            run_index=run_index,
                            run_id=run_id,
                            trace_events=events,
                        )
                    )

    def start(
        self,
        observed: QualificationRuntimeIdentity | None,
        *,
        config_fingerprint: str,
        source_fingerprint: str,
        strict_tool_capable: bool = True,
        structured_output_capable: bool = True,
    ) -> SessionStartResult:
        precondition = evaluate_preconditions(
            self._spec.experiment_identity,
            observed,
            config_fingerprint=config_fingerprint,
            source_fingerprint=source_fingerprint,
            strict_tool_capable=strict_tool_capable,
            structured_output_capable=structured_output_capable,
        )
        if not precondition.eligible:
            transition_state(self._state, QualificationSessionState.BLOCKED)
            self._state = QualificationSessionState.BLOCKED
            checkpoint = self._persist_checkpoint()
            return SessionStartResult(
                state=self._state,
                precondition=precondition,
                checkpoint=checkpoint,
            )

        transition_state(self._state, QualificationSessionState.PRECONDITIONS_PASSED)
        self._state = QualificationSessionState.PRECONDITIONS_PASSED
        transition_state(self._state, QualificationSessionState.RUNNING)
        self._state = QualificationSessionState.RUNNING
        checkpoint = self._persist_checkpoint()
        return SessionStartResult(
            state=self._state,
            precondition=precondition,
            checkpoint=checkpoint,
        )

    def assert_resume_identity(
        self,
        observed: QualificationRuntimeIdentity | None,
        *,
        config_fingerprint: str,
        source_fingerprint: str,
    ) -> QualificationPreconditionResult:
        return evaluate_preconditions(
            self._spec.experiment_identity,
            observed,
            config_fingerprint=config_fingerprint,
            source_fingerprint=source_fingerprint,
            strict_tool_capable=True,
            structured_output_capable=True,
        )

    def checkpoint_source_drift(
        self,
        current_source: SourceFingerprintSnapshot,
    ) -> SourceDriftReport:
        report = compare_source_snapshots(self._frozen_source, current_source)
        if report.qualification_semantic_source_drift:
            if self._state is QualificationSessionState.RUNNING:
                transition_state(self._state, QualificationSessionState.INVALID)
                self._state = QualificationSessionState.INVALID
            self._integrity_failures.append("PARTIAL_INVALID_SOURCE_DRIFT")
            self._persist_checkpoint()
        return report

    def persist_canonical_run(self, record: CanonicalRunRecord) -> RunPersistenceResult:
        if self._state not in {
            QualificationSessionState.RUNNING,
            QualificationSessionState.PARTIAL,
        }:
            raise IllegalSessionTransitionError(
                f"cannot persist run while session state is {self._state.value}"
            )
        self._registry.register(record)
        self._write_runs_json()
        checkpoint = self._persist_checkpoint()
        return RunPersistenceResult(state=self._state, checkpoint=checkpoint)

    def register_run(self, record: CanonicalRunRecord) -> RunPersistenceResult:
        return self.persist_canonical_run(record)

    def integrity_report(
        self,
        observed: QualificationRuntimeIdentity | None,
        *,
        config_matches: bool,
        source_matches: bool,
    ) -> SessionIntegrityReport:
        alignment_statuses: list[TraceReadbackStatus] = []
        trace_runs: list[tuple[dict[str, object], ...]] = []
        for item in self._registry.records():
            readback = read_typed_alignment_events(item.trace_events)
            alignment_statuses.append(readback.status)
            trace_runs.append(item.trace_events)
        attempt_status = assess_session_attempt_evidence(tuple(trace_runs))
        attempt_completeness = (
            EvidenceCompletenessStatus.INCOMPLETE
            if attempt_status is AttemptEvidenceStatus.NOT_AVAILABLE
            else EvidenceCompletenessStatus.COMPLETE
        )
        return SessionIntegrityReport(
            runtime_identity_status=evaluate_runtime_identity_match(
                self._spec.experiment_identity, observed
            ),
            model_identity_status=evaluate_model_identity_match(
                self._spec.experiment_identity, observed
            ),
            source_identity_status=(
                QualificationIdentityStatus.MATCH
                if source_matches
                else QualificationIdentityStatus.MISMATCH
            ),
            config_identity_status=(
                QualificationIdentityStatus.MATCH
                if config_matches
                else QualificationIdentityStatus.MISMATCH
            ),
            trace_readback_status=aggregate_trace_readback_status(tuple(alignment_statuses)),
            attempt_evidence_status=attempt_completeness,
            artifact_completeness_status=EvidenceCompletenessStatus.INCOMPLETE,
            finalization_status=self._state,
        )

    def prepare_finalize_only(self) -> None:
        if self._state is QualificationSessionState.FINALIZED:
            transition_state(self._state, QualificationSessionState.COLLECTED)
            self._state = QualificationSessionState.COLLECTED
            return
        if self._state in {
            QualificationSessionState.FAILED_ARTIFACT_GENERATION,
            QualificationSessionState.FAILED_ARTIFACT_VALIDATION,
            QualificationSessionState.FAILED_FINALIZATION,
        }:
            transition_state(self._state, QualificationSessionState.COLLECTED)
            self._state = QualificationSessionState.COLLECTED
            return
        if self._state in {
            QualificationSessionState.RUNNING,
            QualificationSessionState.PARTIAL,
        }:
            if self.pending_run_indices():
                raise IllegalSessionTransitionError("finalize-only requires collected runs")

    def finalize(
        self,
        *,
        integrity: SessionIntegrityReport,
        inject_derived_failure: Callable[[], None] | None = None,
        observed: QualificationRuntimeIdentity | None = None,
        temperature: float = 0.0,
        regenerate_derived: bool = True,
    ) -> SessionIntegrityReport:
        if self._state is QualificationSessionState.INVALID:
            raise IllegalSessionTransitionError("invalid session cannot finalize as valid")
        if integrity.attempt_evidence_status is EvidenceCompletenessStatus.INCOMPLETE:
            raise IllegalSessionTransitionError("attempt evidence incomplete")

        def _on_state(target: QualificationSessionState) -> None:
            if self._state is target:
                return
            transition_state(self._state, target)
            self._state = target

        context = FinalizationContext(
            session_dir=self._session_dir,
            spec=self._spec,
            session_id=self._session_id,
            task_id=self._task_id,
            observed=observed,
            temperature=temperature,
            on_session_state=_on_state,
        )
        outcome = finalize_with_failure_capture(
            context,
            integrity=integrity,
            inject_derived_failure=inject_derived_failure,
            regenerate_derived=regenerate_derived,
        )
        self._state = outcome.state
        self._persist_checkpoint(finalization_phase=outcome.phase)
        if outcome.integrity is None:
            raise RuntimeError("finalization outcome missing integrity report")
        return outcome.integrity

    def _persist_checkpoint(
        self,
        *,
        finalization_phase: object | None = None,
    ) -> SessionCheckpoint:
        from testing_support.decision_e2e.local_qualification_session.contracts import (
            FinalizationPhase,
        )

        phase: FinalizationPhase | None = None
        if isinstance(finalization_phase, FinalizationPhase):
            phase = finalization_phase
        checkpoint = SessionCheckpoint(
            schema_version=QUALIFICATION_SESSION_SCHEMA_VERSION,
            session_id=self._session_id,
            task_id=self._task_id,
            state=self._state,
            spec=self._spec,
            frozen_source=self._frozen_source,
            completed_run_indices=self._registry.completed_indices(),
            canonical_run_ids=tuple(record.run_id for record in self._registry.records()),
            integrity_failures=tuple(self._integrity_failures),
            finalization_phase=phase,
            artifact_generation_state={},
        )
        persist_checkpoint(self._session_dir, checkpoint)
        return checkpoint

    def _write_runs_json(self) -> None:
        payload = {
            "session_id": self._session_id,
            "task_id": self._task_id,
            "runs": [
                {
                    "run_index": record.run_index,
                    "run_id": record.run_id,
                    "trace_events": list(record.trace_events),
                }
                for record in self._registry.records()
            ],
        }
        atomic_write_json(self._session_dir / "runs.json", payload)
