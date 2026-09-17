# © Artur Czarnecki. All rights reserved.

"""INSPECT-01-B core domain read adoption qualification gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from intergrax.contracts.agent_runtime_governance import (
    GovernanceAuditEvent,
    ToolAuthorizationDecisionState,
    mint_governance_audit_event_id,
)
from intergrax.contracts.execution_continuation import (
    ContinuationReason,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
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
from intergrax.contracts.execution_reconstruction_models import (
    ExecutionReconstruction,
    RuntimeHistoryCompleteness,
)
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.execution_event_position import ExecutionEventPosition
from intergrax.contracts.runtime_inspection import (
    RuntimeInspectionCompleteness,
    RuntimeInspectionQuery,
    RuntimeInspectionSourceFailureCode,
)
from intergrax.contracts.runtime_inspection.sections import RuntimeInspectionToolSection
from intergrax.contracts.runtime_inspection.sources import RuntimeInspectionExecutionScope
from intergrax.contracts.tool_runtime_read import (
    ToolRuntimeExecutionScope,
    ToolRuntimeInvocationOutcome,
    ToolRuntimeInvocationReadResult,
    ToolRuntimeInvocationRecord,
)
from intergrax.runtime.agent_governance.audit import InMemoryGovernanceAuditSink
from intergrax.runtime.agent_governance.audit_read import InMemoryGovernanceAuditReadAdapter
from intergrax.runtime.execution.continuation.continuation_snapshot_read import (
    ExecutionContinuationStateStoreReadAdapter,
)
from intergrax.runtime.execution.continuation.persistence import InMemoryExecutionContinuationStateStore
from intergrax.runtime.runtime_inspection.adapters.continuation_read import (
    ExecutionContinuationInspectionAdapter,
)
from intergrax.runtime.runtime_inspection.adapters.governance_read import (
    GovernanceAuditInspectionAdapter,
)
from intergrax.runtime.runtime_inspection.adapters.tool_runtime_read import (
    ReconstructionToolRuntimeInvocationReader,
    ToolRuntimeInvocationInspectionAdapter,
)
from intergrax.runtime.runtime_inspection.federation import FederatedRuntimeInspectionReadService
from intergrax.runtime.runtime_inspection.redaction import payload_contains_raw_secret
from tests.qualification.inspect_01.test_inspect_01a_federation import (
    _EXEC,
    _FactsReader,
    _SCOPE,
    _ScopeReader,
    _TENANT,
    _reconstruction,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_RAW_SECRET = "INSPECT_B_SECRET_xyz"
_TOOL_ID = "rag.retrieve"


def _tool_events(
    *,
    outcome: RuntimeEventType,
    error_code: str | None = None,
) -> tuple[PositionedRuntimeEvent, ...]:
    base = {
        "tenant_id": _TENANT,
        "task_id": _SCOPE.task_id,
        "run_id": _SCOPE.run_id,
        "attempt_id": _SCOPE.attempt_id,
        "execution_id": _EXEC,
        "phase": ExecutionPhase.STEP_EXECUTION,
    }
    requested = RuntimeEvent(
        event_type=RuntimeEventType.TOOL_REQUESTED,
        payload={
            "tool_id": _TOOL_ID,
            "status": "requested",
            "args_digest": "abc123digest",
            "agent_id": "agent-1",
        },
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        **base,
    )
    outcome_payload = {
        "tool_id": _TOOL_ID,
        "status": "completed" if outcome is RuntimeEventType.TOOL_COMPLETED else "failed",
        "args_digest": "abc123digest",
    }
    if error_code:
        outcome_payload["error_code"] = error_code
    if _RAW_SECRET in json.dumps(outcome_payload):
        raise AssertionError("secret in payload setup")
    outcome_event = RuntimeEvent(
        event_type=outcome,
        payload=outcome_payload,
        timestamp=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        **base,
    )
    return (
        PositionedRuntimeEvent(event=requested, position=ExecutionEventPosition(1)),
        PositionedRuntimeEvent(event=outcome_event, position=ExecutionEventPosition(2)),
    )


def _reconstruction_with_tools(
    outcome: RuntimeEventType,
    *,
    error_code: str | None = None,
) -> ExecutionReconstruction:
    base = _reconstruction()
    tool_rows = _tool_events(outcome=outcome, error_code=error_code)
    return ExecutionReconstruction(
        tenant_id=base.tenant_id,
        task_id=base.task_id,
        run_id=base.run_id,
        causal_evidence=base.causal_evidence,
        positioned_events=(*base.positioned_events, *tool_rows),
        attempts=base.attempts,
        runtime_history_completeness=RuntimeHistoryCompleteness.COMPLETE,
    )


def _federated_with_tools(reconstruction: ExecutionReconstruction) -> FederatedRuntimeInspectionReadService:
    facts = _FactsReader(reconstruction)
    tool_port = ReconstructionToolRuntimeInvocationReader(facts)
    return FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=facts,
        tool_reader=ToolRuntimeInvocationInspectionAdapter(tool_port),
    )


def _governance_service(sink: InMemoryGovernanceAuditSink) -> FederatedRuntimeInspectionReadService:
    audit = InMemoryGovernanceAuditReadAdapter(sink)
    return FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        governance_reader=GovernanceAuditInspectionAdapter(audit),
    )


def _record_governance(
    sink: InMemoryGovernanceAuditSink,
    *,
    decision: ToolAuthorizationDecisionState,
) -> None:
    sink.record(
        GovernanceAuditEvent(
            event_id=mint_governance_audit_event_id(),
            tenant_id=_TENANT,
            execution_id=_EXEC,
            run_id=_SCOPE.run_id,
            attempt_id=_SCOPE.attempt_id,
            task_id=_SCOPE.task_id,
            agent_id="agent-1",
            capability="tool.invoke",
            tool_id=_TOOL_ID,
            decision=decision,
            policy_results=(),
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )


def test_b_q1_tool_success_read() -> None:
    snapshot = _federated_with_tools(
        _reconstruction_with_tools(RuntimeEventType.TOOL_COMPLETED),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.tools is not None
    assert len(snapshot.tools.invocations) == 1
    assert snapshot.tools.invocations[0].outcome is ToolRuntimeInvocationOutcome.COMPLETED


def test_b_q2_tool_failure_read() -> None:
    snapshot = _federated_with_tools(
        _reconstruction_with_tools(RuntimeEventType.TOOL_FAILED, error_code="provider_error"),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.tools is not None
    inv = snapshot.tools.invocations[0]
    assert inv.outcome is ToolRuntimeInvocationOutcome.FAILED
    assert inv.failure_classification == "provider_error"


def test_b_q3_governance_allow() -> None:
    sink = InMemoryGovernanceAuditSink()
    _record_governance(sink, decision=ToolAuthorizationDecisionState.ALLOW)
    snapshot = _governance_service(sink).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    assert snapshot.governance is not None
    assert snapshot.governance.decisions[0].outcome is ToolAuthorizationDecisionState.ALLOW


def test_b_q4_governance_deny() -> None:
    sink = InMemoryGovernanceAuditSink()
    _record_governance(sink, decision=ToolAuthorizationDecisionState.DENY)
    snapshot = _governance_service(sink).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    assert snapshot.governance is not None
    assert snapshot.governance.decisions[0].outcome is ToolAuthorizationDecisionState.DENY
    assert snapshot.tools is None


def test_b_q5_governance_hitl() -> None:
    sink = InMemoryGovernanceAuditSink()
    _record_governance(sink, decision=ToolAuthorizationDecisionState.REQUIRE_APPROVAL)
    store = InMemoryExecutionContinuationStateStore()
    pending = PendingExecutionContinuation(
        continuation_id="cont-1",
        identity=ExecutionContinuationIdentity(
            task_id=_SCOPE.task_id,
            run_id=_SCOPE.run_id,
            attempt_id=_SCOPE.attempt_id,
            execution_id=_EXEC,
        ),
        lifecycle_state=ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=1,
        reason=ContinuationReason.SECURITY,
    )
    store.begin_current_episode_if_predecessor_allows(pending)
    cont_reader = ExecutionContinuationStateStoreReadAdapter(store)
    service = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        governance_reader=GovernanceAuditInspectionAdapter(
            InMemoryGovernanceAuditReadAdapter(sink),
        ),
        continuation_reader=ExecutionContinuationInspectionAdapter(cont_reader),
    )
    snapshot = service.inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.governance is not None
    assert snapshot.governance.decisions[0].approval_required is True
    assert snapshot.continuation is not None
    assert snapshot.continuation.waiting_for_human is True


def test_b_q6_continuation_waiting() -> None:
    store = InMemoryExecutionContinuationStateStore()
    pending = PendingExecutionContinuation(
        continuation_id="cont-wait",
        identity=ExecutionContinuationIdentity(
            task_id=_SCOPE.task_id,
            run_id=_SCOPE.run_id,
            attempt_id=_SCOPE.attempt_id,
            execution_id=_EXEC,
        ),
        lifecycle_state=ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=2,
        reason=ContinuationReason.SECURITY,
    )
    store.begin_current_episode_if_predecessor_allows(pending)
    service = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        continuation_reader=ExecutionContinuationInspectionAdapter(
            ExecutionContinuationStateStoreReadAdapter(store),
        ),
    )
    snapshot = service.inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.continuation is not None
    assert snapshot.continuation.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN


def test_b_q7_continuation_resumed() -> None:
    store = InMemoryExecutionContinuationStateStore()
    pending = PendingExecutionContinuation(
        continuation_id="cont-resumed",
        identity=ExecutionContinuationIdentity(
            task_id=_SCOPE.task_id,
            run_id=_SCOPE.run_id,
            attempt_id=_SCOPE.attempt_id,
            execution_id=_EXEC,
        ),
        lifecycle_state=ExecutionContinuationLifecycleState.RESUMED,
        revision=4,
        reason=ContinuationReason.SECURITY,
    )
    store.begin_current_episode_if_predecessor_allows(pending)
    snapshot = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        continuation_reader=ExecutionContinuationInspectionAdapter(
            ExecutionContinuationStateStoreReadAdapter(store),
        ),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.continuation is not None
    assert snapshot.continuation.resume_status == "resumed"


def test_b_q8_continuation_recovery_read() -> None:
    store = InMemoryExecutionContinuationStateStore()
    pending = PendingExecutionContinuation(
        continuation_id="cont-durable",
        identity=ExecutionContinuationIdentity(
            task_id=_SCOPE.task_id,
            run_id=_SCOPE.run_id,
            attempt_id=_SCOPE.attempt_id,
            execution_id=_EXEC,
        ),
        lifecycle_state=ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=1,
        reason=ContinuationReason.SECURITY,
    )
    store.begin_current_episode_if_predecessor_allows(pending)
    reader = ExecutionContinuationStateStoreReadAdapter(store)
    adapter = ExecutionContinuationInspectionAdapter(reader)
    section = adapter.read_continuation_state(_SCOPE)
    assert section.continuation_id == "cont-durable"


def test_b_q9_partial_tool_source() -> None:
    class _FailReader:
        source_id = "tool_fail"

        def read_tool_invocations(self, scope: RuntimeInspectionExecutionScope):
            raise RuntimeError("unavailable")

    service = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        tool_reader=_FailReader(),
    )
    snapshot = service.inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.tools is None
    assert snapshot.completeness is RuntimeInspectionCompleteness.PARTIAL
    assert any(f.domain == "tool_runtime" for f in snapshot.source_failures)


def test_b_q10_partial_governance_source() -> None:
    class _FailGov:
        source_id = "gov_fail"

        def read_governance_decisions(self, scope: RuntimeInspectionExecutionScope):
            raise RuntimeError("unavailable")

    snapshot = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        governance_reader=_FailGov(),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.governance is None
    assert any(f.domain == "governance" for f in snapshot.source_failures)


def test_b_q11_partial_continuation_source() -> None:
    class _FailCont:
        source_id = "cont_fail"

        def read_continuation_state(self, scope: RuntimeInspectionExecutionScope):
            raise RuntimeError("unavailable")

    snapshot = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        continuation_reader=_FailCont(),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.continuation is None
    assert any(f.domain == "continuation" for f in snapshot.source_failures)


def test_b_q12_empty_vs_unavailable() -> None:
    snapshot = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        tool_reader=ToolRuntimeInvocationInspectionAdapter(
            ReconstructionToolRuntimeInvocationReader(_FactsReader(_reconstruction())),
        ),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.tools is not None
    assert snapshot.tools.invocations == ()
    assert snapshot.tools.source_available is True
    assert not any(f.domain == "tool_runtime" for f in snapshot.source_failures)


def test_b_q13_cross_tenant_integrity_fail_closed() -> None:
    wrong_exec = mint_execution_id()

    class _BadToolReader:
        source_id = "bad_tool"

        def list_invocations(self, scope: ToolRuntimeExecutionScope, *, limit: int):
            return ToolRuntimeInvocationReadResult(
                records=(
                    ToolRuntimeInvocationRecord(
                        invocation_id="inv-1",
                        tool_id=_TOOL_ID,
                        execution_id=wrong_exec,
                        attempt_id=_SCOPE.attempt_id,
                        sequence_key=1,
                        outcome=ToolRuntimeInvocationOutcome.COMPLETED,
                        status_label="completed",
                        failure_classification=None,
                        args_digest_ref=None,
                        provider_correlation_ref=None,
                        governance_evidence_refs=(),
                        evidence_refs=(),
                        safe_summary="ok",
                    ),
                ),
                is_truncated=False,
            )

    snapshot = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        tool_reader=ToolRuntimeInvocationInspectionAdapter(_BadToolReader()),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.tools is None
    failure = next(f for f in snapshot.source_failures if f.domain == "tool_runtime")
    assert failure.code is RuntimeInspectionSourceFailureCode.INTEGRITY


def test_b_q14_secret_redaction() -> None:
    reconstruction = _reconstruction_with_tools(RuntimeEventType.TOOL_COMPLETED)
    snapshot = _federated_with_tools(reconstruction).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    serialized = json.dumps(snapshot.model_dump(mode="json"), sort_keys=True)
    assert not payload_contains_raw_secret(serialized, raw_secret=_RAW_SECRET)


def test_b_q15_zero_side_effects() -> None:
    sink = InMemoryGovernanceAuditSink()
    _record_governance(sink, decision=ToolAuthorizationDecisionState.ALLOW)
    before_events = len(sink.events)
    _governance_service(sink).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert len(sink.events) == before_events


def test_b_q16_replaceable_read_sources() -> None:
    class _CustomTool:
        source_id = "custom_tool"

        def read_tool_invocations(self, scope: RuntimeInspectionExecutionScope):
            return RuntimeInspectionToolSection(
                invocations=(),
                completeness=RuntimeInspectionCompleteness.COMPLETE,
                source_id=self.source_id,
            )

    snapshot = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        tool_reader=_CustomTool(),
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.tools is not None
    assert snapshot.tools.source_id == "custom_tool"


def test_b_q17_no_opaque_abi_on_new_sections() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    sections = (root / "intergrax" / "contracts" / "runtime_inspection" / "sections.py").read_text(
        encoding="utf-8",
    )
    assert "dict[str, Any]" not in sections
    assert ": Any" not in sections


def test_b_q18_federation_still_no_registry() -> None:
    import ast
    from pathlib import Path

    from intergrax.runtime.runtime_inspection import federation as federation_module

    source = Path(federation_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "getattr" not in names
    assert "register_source" not in source
