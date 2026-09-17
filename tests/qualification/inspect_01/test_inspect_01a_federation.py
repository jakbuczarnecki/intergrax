# © Artur Czarnecki. All rights reserved.

"""INSPECT-01-A federation qualification gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_reconstruction_models import (
    ExecutionReconstruction,
    RuntimeHistoryCompleteness,
)
from intergrax.contracts.functional_evidence.models import (
    PipelineEvidenceKind,
    PlatformFunctionalEvidence,
)
from intergrax.contracts.functional_evidence.persistence import FunctionalEvidenceQueryPage
from intergrax.contracts.platform_causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.execution_event_position import ExecutionEventPosition
from intergrax.contracts.runtime_inspection import (
    RuntimeInspectionCompleteness,
    RuntimeInspectionDiagnosticSection,
    RuntimeInspectionEvidenceSection,
    RuntimeInspectionNotFoundError,
    RuntimeInspectionQuery,
    RuntimeInspectionScopeLookupOutcome,
    RuntimeInspectionScopeLookupResult,
    RuntimeInspectionTenantBoundaryError,
)
from intergrax.contracts.runtime_inspection.sections import RuntimeInspectionEvidenceReference
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionExecutionScope,
    RuntimeInspectionScopeLookupOutcome as Outcome,
)
from intergrax.runtime.runtime_inspection.federation import FederatedRuntimeInspectionReadService

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-inspect-a"
_EXEC = mint_execution_id()
_TASK = mint_task_id()
_RUN = mint_run_id()
_ATTEMPT = mint_attempt_id()
_SCOPE = RuntimeInspectionExecutionScope(
    tenant_id=_TENANT,
    task_id=_TASK,
    run_id=_RUN,
    attempt_id=_ATTEMPT,
    execution_id=_EXEC,
)


class _ScopeReader:
    source_id = "scope"

    def __init__(self, outcome: RuntimeInspectionScopeLookupOutcome = Outcome.FOUND) -> None:
        self._outcome = outcome

    def resolve_scope(self, *, tenant_id: str, execution_id: ExecutionId):
        if self._outcome is Outcome.NOT_FOUND:
            return RuntimeInspectionScopeLookupResult(outcome=Outcome.NOT_FOUND)
        if self._outcome is Outcome.TENANT_DENIED:
            return RuntimeInspectionScopeLookupResult(outcome=Outcome.TENANT_DENIED)
        return RuntimeInspectionScopeLookupResult(outcome=Outcome.FOUND, scope=_SCOPE)


def _reconstruction(*, secret_in_kind: str | None = None) -> ExecutionReconstruction:
    event = RuntimeEvent(
        tenant_id=_TENANT,
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXEC,
        event_type=RuntimeEventType.STEP_STARTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        event_kind=secret_in_kind or "started",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    positioned = PositionedRuntimeEvent(event=event, position=ExecutionEventPosition(1))
    causal = PlatformCausalEvidence(
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=_TENANT,
        source=MessageBusTaskRef(provider="test", task_id="t1", tenant_id=_TENANT),
        target=RuntimeExecutionRef(
            task_id=_TASK,
            run_id=_RUN,
            attempt_id=_ATTEMPT,
            execution_id=_EXEC,
            tenant_id=_TENANT,
        ),
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return ExecutionReconstruction(
        tenant_id=_TENANT,
        task_id=_TASK,
        run_id=_RUN,
        causal_evidence=(causal,),
        positioned_events=(positioned,),
        attempts=(),
        runtime_history_completeness=RuntimeHistoryCompleteness.COMPLETE,
    )


class _FactsReader:
    source_id = "execution_facts"

    def __init__(self, reconstruction: ExecutionReconstruction | None = None) -> None:
        self._reconstruction = reconstruction or _reconstruction()
        self.calls = 0

    def read_execution_facts(self, scope: RuntimeInspectionExecutionScope):
        self.calls += 1
        return self._reconstruction


class _DiagnosticReader:
    source_id = "diagnostics"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def read_diagnostics(self, scope: RuntimeInspectionExecutionScope):
        self.calls += 1
        if self.fail:
            raise RuntimeError("diagnostic unavailable")
        return RuntimeInspectionDiagnosticSection(
            completeness=RuntimeInspectionCompleteness.COMPLETE,
            source_id=self.source_id,
            findings=(),
        )


class _EvidenceReader:
    source_id = "evidence"

    def __init__(self) -> None:
        self.calls = 0

    def read_evidence_references(self, scope: RuntimeInspectionExecutionScope):
        self.calls += 1
        evidence_id = mint_event_id()
        return RuntimeInspectionEvidenceSection(
            references=(
                RuntimeInspectionEvidenceReference(
                    evidence_id=evidence_id,
                    kind=PipelineEvidenceKind.VALIDATION.value,
                    source_id=self.source_id,
                ),
            ),
            completeness=RuntimeInspectionCompleteness.COMPLETE,
            source_id=self.source_id,
        )


def _service(
    *,
    scope_outcome: RuntimeInspectionScopeLookupOutcome = Outcome.FOUND,
    diagnostic_fail: bool = False,
    include_evidence: bool = True,
) -> FederatedRuntimeInspectionReadService:
    return FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(scope_outcome),
        execution_facts_reader=_FactsReader(),
        diagnostic_reader=_DiagnosticReader(fail=diagnostic_fail),
        evidence_reader=_EvidenceReader() if include_evidence else None,
    )


def test_a_q1_canonical_execution_snapshot() -> None:
    snapshot = _service().inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.identity.execution_id == _EXEC
    assert snapshot.identity.task_id == _TASK
    assert snapshot.schema_version == "runtime_inspection_snapshot.v1"


def test_a_q2_timeline_bounded_deterministic() -> None:
    service = _service()
    query = RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC, timeline_limit=1)
    first = service.inspect(query)
    second = service.inspect(query)
    assert first.timeline.is_truncated is True
    assert first.timeline.entries == second.timeline.entries
    assert first.timeline.entries[0].sequence_key == 1


def test_a_q3_diagnostics_delegate_to_read_service() -> None:
    from intergrax.runtime.runtime_inspection.adapters.diagnostic_read import (
        DiagnosticReadServiceInspectionAdapter,
    )

    class _RecordingService:
        def assess_execution_scope_for_inspection(self, **kwargs):
            self.kwargs = kwargs
            return None

    recording = _RecordingService()
    adapter = DiagnosticReadServiceInspectionAdapter(recording)  # type: ignore[arg-type]
    adapter.read_diagnostics(_SCOPE)
    assert recording.kwargs["tenant_id"] == _TENANT


def test_a_q4_evidence_refs_only() -> None:
    snapshot = _service().inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.evidence is not None
    assert snapshot.evidence.references
    assert all(ref.evidence_id for ref in snapshot.evidence.references)


def test_a_q5_completeness_complete_vs_partial() -> None:
    complete = _service(include_evidence=False).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    assert complete.completeness in {
        RuntimeInspectionCompleteness.COMPLETE,
        RuntimeInspectionCompleteness.PARTIAL,
    }
    partial = _service(diagnostic_fail=True).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    assert partial.completeness is RuntimeInspectionCompleteness.PARTIAL
    assert partial.source_failures


def test_a_q6_partial_source_failure() -> None:
    snapshot = _service(diagnostic_fail=True).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    assert snapshot.execution.has_runtime_events is True
    assert snapshot.source_failures[0].domain == "diagnostics"


def test_a_q7_typed_not_found() -> None:
    with pytest.raises(RuntimeInspectionNotFoundError):
        _service(scope_outcome=Outcome.NOT_FOUND).inspect(
            RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
        )


def test_a_q8_tenant_isolation() -> None:
    with pytest.raises(RuntimeInspectionTenantBoundaryError):
        _service(scope_outcome=Outcome.TENANT_DENIED).inspect(
            RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
        )


def test_a_q11_replaceable_adapters() -> None:
    class _AltDiagnostic:
        source_id = "alt_diagnostics"

        def read_diagnostics(self, scope: RuntimeInspectionExecutionScope):
            return RuntimeInspectionDiagnosticSection(
                completeness=RuntimeInspectionCompleteness.PARTIAL,
                source_id=self.source_id,
                findings=(),
            )

    service = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        diagnostic_reader=_AltDiagnostic(),
    )
    snapshot = service.inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.diagnostics is not None
    assert snapshot.diagnostics.source_id == "alt_diagnostics"
