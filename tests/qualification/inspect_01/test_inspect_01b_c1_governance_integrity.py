# © Artur Czarnecki. All rights reserved.

"""INSPECT-01-B-C1 governance tenant and identity integrity gates."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.agent_runtime_governance import (
    GovernanceAuditEvent,
    ToolAuthorizationDecisionState,
    mint_governance_audit_event_id,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.governance_audit_read import (
    GovernanceAuditReadPort,
    GovernanceAuditReadTenantBoundaryError,
)
from intergrax.contracts.runtime_inspection import (
    RuntimeInspectionCompleteness,
    RuntimeInspectionQuery,
    RuntimeInspectionSourceFailureCode,
)
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
    RuntimeInspectionTenantBoundaryError,
)
from intergrax.contracts.runtime_inspection.sections import RuntimeInspectionGovernanceSection
from intergrax.contracts.runtime_inspection.sources import RuntimeInspectionExecutionScope
from intergrax.runtime.agent_governance.audit import InMemoryGovernanceAuditSink
from intergrax.runtime.agent_governance.audit_read import InMemoryGovernanceAuditReadAdapter
from intergrax.runtime.runtime_inspection.adapters.governance_read import (
    GovernanceAuditInspectionAdapter,
)
from intergrax.runtime.runtime_inspection.federation import FederatedRuntimeInspectionReadService
from tests.qualification.inspect_01.catalog import (
    INSPECT_01_A_Q_CATALOG,
    INSPECT_01_B_Q_CATALOG,
    INSPECT_01_C1_Q_CATALOG,
)
from tests.qualification.inspect_01.test_inspect_01a_federation import (
    _EXEC,
    _FactsReader,
    _SCOPE,
    _ScopeReader,
    _TENANT,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_OTHER_TENANT = "tenant-inspect-b"


def _audit_event(**overrides: object) -> GovernanceAuditEvent:
    payload = {
        "event_id": mint_governance_audit_event_id(),
        "tenant_id": _TENANT,
        "execution_id": _EXEC,
        "run_id": _SCOPE.run_id,
        "attempt_id": _SCOPE.attempt_id,
        "task_id": _SCOPE.task_id,
        "agent_id": "agent-1",
        "capability": "tool.invoke",
        "tool_id": "rag.retrieve",
        "decision": ToolAuthorizationDecisionState.ALLOW,
        "policy_results": (),
        "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    payload.update(overrides)
    return GovernanceAuditEvent(**payload)


def _governance_service(sink: InMemoryGovernanceAuditSink) -> FederatedRuntimeInspectionReadService:
    audit = InMemoryGovernanceAuditReadAdapter(sink)
    return FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(),
        governance_reader=GovernanceAuditInspectionAdapter(audit),
    )


def test_c1_q1_tenant_filtered_by_governance_read_port() -> None:
    sink = InMemoryGovernanceAuditSink()
    sink.record(_audit_event())
    sink.record(
        _audit_event(
            tenant_id=_OTHER_TENANT,
            execution_id=mint_execution_id(),
        ),
    )
    reader = InMemoryGovernanceAuditReadAdapter(sink)
    events = reader.list_audit_events_for_execution(
        tenant_id=_TENANT,
        execution_id=_EXEC,
        limit=10,
    )
    assert len(events) == 1
    assert events[0].tenant_id == _TENANT


def test_c1_q2_cross_tenant_same_execution_id_fail_closed() -> None:
    sink = InMemoryGovernanceAuditSink()
    sink.record(_audit_event(tenant_id=_OTHER_TENANT))
    reader = InMemoryGovernanceAuditReadAdapter(sink)
    with pytest.raises(GovernanceAuditReadTenantBoundaryError):
        reader.list_audit_events_for_execution(
            tenant_id=_TENANT,
            execution_id=_EXEC,
            limit=10,
        )
    with pytest.raises(RuntimeInspectionTenantBoundaryError):
        _governance_service(sink).inspect(
            RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
        )


def test_c1_q3_task_id_mismatch_source_integrity() -> None:
    sink = InMemoryGovernanceAuditSink()
    sink.record(_audit_event(task_id=mint_task_id()))
    with pytest.raises(RuntimeInspectionError) as exc_info:
        _governance_service(sink).inspect(
            RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
        )
    assert exc_info.value.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY


def test_c1_q4_run_id_mismatch_source_integrity() -> None:
    sink = InMemoryGovernanceAuditSink()
    sink.record(_audit_event(run_id=mint_run_id()))
    with pytest.raises(RuntimeInspectionError) as exc_info:
        _governance_service(sink).inspect(
            RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
        )
    assert exc_info.value.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY


def test_c1_q5_attempt_id_mismatch_source_integrity() -> None:
    sink = InMemoryGovernanceAuditSink()
    sink.record(_audit_event(attempt_id=mint_attempt_id()))
    with pytest.raises(RuntimeInspectionError) as exc_info:
        _governance_service(sink).inspect(
            RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
        )
    assert exc_info.value.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY


def test_c1_q6_execution_id_mismatch_source_integrity() -> None:
    class _BadRead(GovernanceAuditReadPort):
        source_id = "bad_governance_read"

        def list_audit_events_for_execution(self, *, tenant_id, execution_id, limit):
            return (_audit_event(execution_id=mint_execution_id()),)

    with pytest.raises(RuntimeInspectionError) as exc_info:
        GovernanceAuditInspectionAdapter(_BadRead()).read_governance_decisions(_SCOPE)
    assert exc_info.value.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY


def test_c1_q7_valid_empty_governance_source_not_unavailable() -> None:
    sink = InMemoryGovernanceAuditSink()
    snapshot = _governance_service(sink).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    assert snapshot.governance is not None
    assert snapshot.governance.decisions == ()
    assert snapshot.governance.source_available is True
    assert snapshot.governance.completeness is RuntimeInspectionCompleteness.COMPLETE
    assert not any(f.domain == "governance" for f in snapshot.source_failures)


def test_c1_q8_governance_availability_failure_partial() -> None:
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
    assert snapshot.completeness is RuntimeInspectionCompleteness.PARTIAL
    failure = next(f for f in snapshot.source_failures if f.domain == "governance")
    assert failure.code is RuntimeInspectionSourceFailureCode.UNAVAILABLE


def test_c1_q9_governance_integrity_failure_not_partial() -> None:
    sink = InMemoryGovernanceAuditSink()
    sink.record(_audit_event(task_id=mint_task_id()))
    with pytest.raises(RuntimeInspectionError):
        _governance_service(sink).inspect(
            RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
        )


def test_c1_q10_no_policy_re_evaluation() -> None:
    source = Path(
        "intergrax/runtime/runtime_inspection/adapters/governance_read.py",
    ).read_text(encoding="utf-8")
    forbidden = (
        "evaluate(",
        "authorize(",
        "enforce(",
        "resolve_policy",
        "AgentRuntimeGovernancePipeline",
        "AgentRuntimePolicyEngine",
    )
    for token in forbidden:
        assert token not in source


def test_c1_q11_no_read_side_writes() -> None:
    sink = InMemoryGovernanceAuditSink()
    sink.record(_audit_event())
    before = len(sink.events)
    _governance_service(sink).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    assert len(sink.events) == before


def test_c1_q12_custom_governance_audit_read_port() -> None:
    class _CustomAudit(GovernanceAuditReadPort):
        source_id = "custom_governance_audit"

        def list_audit_events_for_execution(self, *, tenant_id, execution_id, limit):
            assert tenant_id == _TENANT
            return (_audit_event(),)

    section = GovernanceAuditInspectionAdapter(_CustomAudit()).read_governance_decisions(_SCOPE)
    assert section.source_id == "custom_governance_audit"
    assert len(section.decisions) == 1


def test_c1_q13_provider_neutral_contracts() -> None:
    text = Path("intergrax/contracts/governance_audit_read.py").read_text(encoding="utf-8")
    assert "sqlalchemy" not in text.lower()
    assert "redis" not in text.lower()
    assert "dict[str," not in text


def test_c1_q14_no_opaque_abi_vendor_leakage() -> None:
    text = Path("intergrax/contracts/agent_runtime_governance.py").read_text(encoding="utf-8")
    assert "GovernanceAuditEvent" in text
    assert "tenant_id" in text
    assert "Any" not in text.split("GovernanceAuditEvent")[1].split("class ")[0]


def test_c1_q15_full_inspect_a_b_regression() -> None:
    for entry in INSPECT_01_A_Q_CATALOG:
        assert entry.pytest_node_ids
    for entry in INSPECT_01_B_Q_CATALOG:
        assert entry.pytest_node_ids
    for entry in INSPECT_01_C1_Q_CATALOG:
        assert entry.pytest_node_ids


def test_inspect_01_c1_catalog_covers_c1_q1_through_c1_q15() -> None:
    ids = {entry.q_id for entry in INSPECT_01_C1_Q_CATALOG}
    expected = {f"C1-Q{i}" for i in range(1, 16)}
    assert ids == expected
