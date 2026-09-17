# © Artur Czarnecki. All rights reserved.

"""INSPECT-01-RQ enterprise requalification — system-level gates RQ-Q1..RQ-Q10."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from intergrax.contracts.agent_runtime_governance import ToolAuthorizationDecisionState
from intergrax.contracts.execution_continuation import (
    ContinuationReason,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_identity import mint_execution_id, mint_task_id
from intergrax.contracts.execution_reconstruction_models import ExecutionReconstruction
from intergrax.contracts.memory_runtime_read import (
    MemoryRuntimeExecutionScope,
    MemoryRuntimeOperationReadPort,
    MemoryRuntimeOperationReadResult,
)
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.contracts.runtime_inspection import (
    RuntimeInspectionCompleteness,
    RuntimeInspectionErrorCode,
    RuntimeInspectionQuery,
    RuntimeInspectionTenantBoundaryError,
)
from intergrax.contracts.runtime_inspection.errors import RuntimeInspectionError
from intergrax.contracts.runtime_inspection.sources import RuntimeInspectionExecutionScope
from intergrax.runtime.agent_governance.audit import InMemoryGovernanceAuditSink
from intergrax.runtime.agent_governance.audit_read import InMemoryGovernanceAuditReadAdapter
from intergrax.runtime.execution.continuation.continuation_snapshot_read import (
    ExecutionContinuationStateStoreReadAdapter,
)
from intergrax.runtime.execution.continuation.persistence import InMemoryExecutionContinuationStateStore
from intergrax.runtime.runtime_inspection.adapters.artifact_read import (
    ArtifactMetadataInspectionAdapter,
    ReconstructionArtifactMetadataReader,
)
from intergrax.runtime.runtime_inspection.adapters.continuation_read import (
    ExecutionContinuationInspectionAdapter,
)
from intergrax.runtime.runtime_inspection.adapters.external_work_read import (
    ExternalWorkInspectionAdapter,
    ReconstructionExternalWorkFactReader,
)
from intergrax.runtime.runtime_inspection.adapters.governance_read import (
    GovernanceAuditInspectionAdapter,
)
from intergrax.runtime.runtime_inspection.adapters.memory_read import (
    MemoryOperationInspectionAdapter,
    ReconstructionMemoryOperationReader,
)
from intergrax.runtime.runtime_inspection.adapters.model_read import (
    ModelInvocationInspectionAdapter,
    ReconstructionModelInvocationReader,
)
from intergrax.runtime.runtime_inspection.adapters.tool_runtime_read import (
    ReconstructionToolRuntimeInvocationReader,
    ToolRuntimeInvocationInspectionAdapter,
)
from intergrax.runtime.runtime_inspection.adapters.truncation_completeness import (
    completeness_for_read_result,
)
from intergrax.runtime.runtime_inspection.federation import FederatedRuntimeInspectionReadService
from intergrax.runtime.runtime_inspection.redaction import payload_contains_raw_secret
from tests.qualification.inspect_01.test_inspect_01a_federation import (
    _DiagnosticReader,
    _EvidenceReader,
    _EXEC,
    _FactsReader,
    _SCOPE,
    _ScopeReader,
    _TENANT,
)
from tests.qualification.inspect_01.test_inspect_01b_core_domains import (
    _RAW_SECRET as _B_SECRET,
    _record_governance,
    _reconstruction_with_tools,
    _tool_events,
)
from tests.qualification.inspect_01.test_inspect_01c_extended_domains import (
    _extended_reconstruction,
    _memory_record,
    _positioned,
    _RAW_SECRET as _C_SECRET,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_INSPECTION_DIR = _REPO_ROOT / "intergrax" / "runtime" / "runtime_inspection"
_CONTRACTS_ROOT = _REPO_ROOT / "intergrax" / "contracts"
_RUNTIME_INSPECTION_CONTRACT_DIR = _CONTRACTS_ROOT / "runtime_inspection"
_RQ_Q10_CANONICAL_CONTRACT_FILES = (
    _CONTRACTS_ROOT / "memory_runtime_read.py",
    _CONTRACTS_ROOT / "model_runtime_read.py",
    _CONTRACTS_ROOT / "external_work_runtime_read.py",
    _CONTRACTS_ROOT / "execution_artifact_read.py",
    _CONTRACTS_ROOT / "execution_scope_identity.py",
    _CONTRACTS_ROOT / "tool_runtime_read.py",
    _CONTRACTS_ROOT / "governance_audit_read.py",
    _CONTRACTS_ROOT / "execution_continuation_read.py",
)
_FORBIDDEN_REFLECTION_NAMES = frozenset({"getattr", "setattr", "hasattr"})
_FORBIDDEN_ABI_TOKENS = ("dict[str, Any]",)
_VENDOR_MODULE_TOKENS = frozenset(
    {
        "openai",
        "anthropic",
        "boto3",
        "aws",
        "azure",
        "gcp",
        "redis",
        "celery",
        "kafka",
        "postgres",
        "psycopg",
        "s3",
    },
)
_FORBIDDEN_CONTRACT_IMPORT_PREFIXES = (
    "agents.",
    "applications.",
    "intergrax.runtime.",
)
_OTHER_TENANT = "tenant-inspect-rq-other"


def _rq_q10_scan_paths() -> tuple[Path, ...]:
    paths: set[Path] = set(_RUNTIME_INSPECTION_DIR.rglob("*.py"))
    paths.update(_RUNTIME_INSPECTION_CONTRACT_DIR.rglob("*.py"))
    paths.update(_RQ_Q10_CANONICAL_CONTRACT_FILES)
    return tuple(sorted(paths))


def _rq_q10_contract_paths() -> tuple[Path, ...]:
    paths: set[Path] = set(_RUNTIME_INSPECTION_CONTRACT_DIR.rglob("*.py"))
    paths.update(_RQ_Q10_CANONICAL_CONTRACT_FILES)
    return tuple(sorted(paths))


def _rq_q10_reflection_hits(tree: ast.AST) -> frozenset[str]:
    hits: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_REFLECTION_NAMES:
            hits.add(node.id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_REFLECTION_NAMES
        ):
            hits.add(node.func.id)
    return frozenset(hits)


def _rq_q10_imported_modules(tree: ast.AST) -> tuple[str, ...]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)


def _rq_q10_vendor_module_hits(modules: tuple[str, ...]) -> frozenset[str]:
    hits: set[str] = set()
    for module in modules:
        parts = module.lower().split(".")
        for token in _VENDOR_MODULE_TOKENS:
            if token in parts or module.lower() == token or module.lower().startswith(f"{token}."):
                hits.add(token)
    return frozenset(hits)


def _rq_q10_forbidden_contract_imports(modules: tuple[str, ...]) -> tuple[str, ...]:
    blocked: list[str] = []
    for module in modules:
        for prefix in _FORBIDDEN_CONTRACT_IMPORT_PREFIXES:
            if module == prefix.removesuffix(".") or module.startswith(prefix):
                blocked.append(module)
                break
    return tuple(blocked)


def _representative_reconstruction() -> ExecutionReconstruction:
    extended = _extended_reconstruction()
    tool_rows = _tool_events(outcome=RuntimeEventType.TOOL_COMPLETED)
    offset = len(extended.positioned_events)
    merged_tools = tuple(
        _positioned(pe.event, offset + pe.position.value) for pe in tool_rows
    )
    return replace(
        extended,
        positioned_events=(*extended.positioned_events, *merged_tools),
    )


def _continuation_store_waiting() -> InMemoryExecutionContinuationStateStore:
    store = InMemoryExecutionContinuationStateStore()
    pending = PendingExecutionContinuation(
        continuation_id="cont-rq",
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
    return store


def _enterprise_service(
    reconstruction: ExecutionReconstruction | None = None,
    *,
    governance_sink: InMemoryGovernanceAuditSink | None = None,
    continuation_store: InMemoryExecutionContinuationStateStore | None = None,
    memory_reader: object | None = None,
    diagnostic_fail: bool = False,
    tool_reader: object | None = None,
) -> FederatedRuntimeInspectionReadService:
    facts = _FactsReader(reconstruction or _representative_reconstruction())
    sink = governance_sink or InMemoryGovernanceAuditSink()
    if governance_sink is None:
        _record_governance(sink, decision=ToolAuthorizationDecisionState.ALLOW)
    store = continuation_store or _continuation_store_waiting()
    cont_adapter = ExecutionContinuationInspectionAdapter(
        ExecutionContinuationStateStoreReadAdapter(store),
    )
    default_memory = MemoryOperationInspectionAdapter(ReconstructionMemoryOperationReader(facts))
    default_tool = ToolRuntimeInvocationInspectionAdapter(
        ReconstructionToolRuntimeInvocationReader(facts),
    )
    return FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=facts,
        diagnostic_reader=_DiagnosticReader(fail=diagnostic_fail),
        evidence_reader=_EvidenceReader(),
        tool_reader=tool_reader or default_tool,
        governance_reader=GovernanceAuditInspectionAdapter(InMemoryGovernanceAuditReadAdapter(sink)),
        continuation_reader=cont_adapter,
        memory_reader=memory_reader or default_memory,
        model_reader=ModelInvocationInspectionAdapter(ReconstructionModelInvocationReader(facts)),
        external_work_reader=ExternalWorkInspectionAdapter(ReconstructionExternalWorkFactReader(facts)),
        artifact_reader=ArtifactMetadataInspectionAdapter(ReconstructionArtifactMetadataReader(facts)),
    )


def _canonical_payload(snapshot) -> dict[str, object]:
    payload = snapshot.model_dump(mode="json")
    payload.pop("observed_at", None)
    evidence = payload.get("evidence")
    if isinstance(evidence, dict):
        for ref in evidence.get("references", ()):
            if isinstance(ref, dict):
                ref.pop("evidence_id", None)
    return payload


def test_rq_q1_complete_federation_snapshot() -> None:
    snapshot = _enterprise_service().inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    assert snapshot.identity.execution_id == _EXEC
    assert snapshot.tools is not None and snapshot.tools.invocations
    assert snapshot.governance is not None and snapshot.governance.decisions
    assert snapshot.continuation is not None
    assert snapshot.memory is not None and snapshot.memory.operations
    assert snapshot.model is not None and snapshot.model.invocations
    assert snapshot.external_work is not None and snapshot.external_work.work_entries
    assert snapshot.artifacts is not None and snapshot.artifacts.artifacts
    assert snapshot.evidence is not None
    assert snapshot.diagnostics is not None


def test_rq_q2_tenant_fail_closed_across_federation() -> None:
    class _ForeignMemory(MemoryRuntimeOperationReadPort):
        source_id = "foreign_mem"

        def list_operations(self, scope, *, limit: int):
            return MemoryRuntimeOperationReadResult(
                records=(_memory_record(tenant_id=_OTHER_TENANT),),
                is_truncated=False,
            )

    with pytest.raises(RuntimeInspectionTenantBoundaryError):
        _enterprise_service(
            memory_reader=MemoryOperationInspectionAdapter(_ForeignMemory()),
        ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))


def test_rq_q3_integrity_failure_not_downgraded() -> None:
    class _WrongExecution(MemoryRuntimeOperationReadPort):
        source_id = "bad_exec"

        def list_operations(self, scope, *, limit: int):
            return MemoryRuntimeOperationReadResult(
                records=(_memory_record(execution_id=mint_execution_id()),),
                is_truncated=False,
            )

    with pytest.raises(RuntimeInspectionError) as exc_info:
        _enterprise_service(
            memory_reader=MemoryOperationInspectionAdapter(_WrongExecution()),
        ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert exc_info.value.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY


def test_rq_q4_optional_unavailable_source_partial() -> None:
    snapshot = _enterprise_service(diagnostic_fail=True).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    assert snapshot.completeness is RuntimeInspectionCompleteness.PARTIAL
    assert snapshot.tools is not None
    assert any(f.domain == "diagnostics" for f in snapshot.source_failures)


def test_rq_q5_plugin_custom_source_substitution() -> None:
    class _CustomMemoryPort(MemoryRuntimeOperationReadPort):
        source_id = "rq_custom_memory"

        def list_operations(
            self,
            scope: MemoryRuntimeExecutionScope,
            *,
            limit: int,
        ) -> MemoryRuntimeOperationReadResult:
            assert scope.tenant_id == _TENANT
            assert scope.task_id == _SCOPE.task_id
            assert scope.run_id == _SCOPE.run_id
            assert scope.attempt_id == _SCOPE.attempt_id
            assert scope.execution_id == _EXEC
            record = _memory_record(source_category="rq_custom_canonical")
            return MemoryRuntimeOperationReadResult(
                records=(record,),
                is_truncated=False,
            )

    memory_reader = MemoryOperationInspectionAdapter(_CustomMemoryPort())
    snapshot = _enterprise_service(memory_reader=memory_reader).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    assert snapshot.memory is not None
    assert snapshot.memory.source_id == "rq_custom_memory"
    assert len(snapshot.memory.operations) == 1
    operation = snapshot.memory.operations[0]
    assert operation.execution_id == _EXEC
    assert operation.attempt_id == _SCOPE.attempt_id
    assert snapshot.identity.task_id == _SCOPE.task_id
    assert snapshot.identity.run_id == _SCOPE.run_id
    assert snapshot.identity.attempt_id == _SCOPE.attempt_id
    assert snapshot.identity.execution_id == _EXEC
    assert snapshot.tools is not None


def test_rq_q6_read_only_e2e() -> None:
    reconstruction = _representative_reconstruction()
    event_count = len(reconstruction.positioned_events)
    sink = InMemoryGovernanceAuditSink()
    _record_governance(sink, decision=ToolAuthorizationDecisionState.ALLOW)
    audit_before = len(sink.events)
    store = _continuation_store_waiting()
    identity = ExecutionContinuationIdentity(
        task_id=_SCOPE.task_id,
        run_id=_SCOPE.run_id,
        attempt_id=_SCOPE.attempt_id,
        execution_id=_EXEC,
    )
    before_pending = store.resolve_current_episode_for_identity(identity)
    assert before_pending is not None
    before_revision = before_pending.revision
    service = _enterprise_service(
        reconstruction,
        governance_sink=sink,
        continuation_store=store,
    )
    service.inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert len(reconstruction.positioned_events) == event_count
    assert len(sink.events) == audit_before
    after_pending = store.resolve_current_episode_for_identity(identity)
    assert after_pending is not None
    assert after_pending.revision == before_revision


def test_rq_q7_redaction_boundary() -> None:
    reconstruction = _representative_reconstruction()
    events = list(reconstruction.positioned_events)
    llm_idx = next(i for i, pe in enumerate(events) if pe.event.event_type.name == "LLM_CALL")
    llm = events[llm_idx].event.model_copy(
        update={
            "payload": {
                "model": "m",
                "label": _C_SECRET,
                "prompt_tokens": 1,
                "completion_tokens": 0,
                "total_tokens": 1,
            },
        },
    )
    events[llm_idx] = _positioned(llm, events[llm_idx].position.value)
    tool_idx = len(events) - 1
    tool_ev = events[tool_idx].event.model_copy(
        update={
            "payload": {
                **events[tool_idx].event.payload,
                "args_digest": f"digest-{_B_SECRET}",
            },
        },
    )
    events[tool_idx] = _positioned(tool_ev, events[tool_idx].position.value)
    snapshot = _enterprise_service(replace(reconstruction, positioned_events=tuple(events))).inspect(
        RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
    )
    serialized = json.dumps(snapshot.model_dump(mode="json"), sort_keys=True)
    assert _C_SECRET not in serialized
    assert _B_SECRET not in serialized
    assert not payload_contains_raw_secret(serialized, raw_secret=_C_SECRET)
    assert not payload_contains_raw_secret(serialized, raw_secret=_B_SECRET)


def test_rq_q8_deterministic_repeated_inspection() -> None:
    service = _enterprise_service()
    query = RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC)
    first = _canonical_payload(service.inspect(query))
    second = _canonical_payload(service.inspect(query))
    assert first == second


def test_rq_q9_truncation_completeness_consistency() -> None:
    class _Truncated(MemoryRuntimeOperationReadPort):
        source_id = "rq_trunc"

        def list_operations(self, scope, *, limit: int):
            return MemoryRuntimeOperationReadResult(
                records=tuple(_memory_record(sequence_key=i) for i in range(1, limit + 1)),
                is_truncated=True,
            )

    adapter = MemoryOperationInspectionAdapter(_Truncated(), operation_limit=3)
    section = adapter.read_memory_operations(_SCOPE)
    assert section.is_truncated is True
    assert section.completeness is RuntimeInspectionCompleteness.PARTIAL
    assert completeness_for_read_result(is_truncated=True) is RuntimeInspectionCompleteness.PARTIAL

    snapshot = _enterprise_service(
        memory_reader=adapter,
    ).inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert snapshot.memory is not None
    assert snapshot.memory.is_truncated is True
    assert snapshot.memory.completeness is RuntimeInspectionCompleteness.PARTIAL


def test_rq_q10_architecture_prohibited_pattern_gate() -> None:
    runtime_paths = sorted(_RUNTIME_INSPECTION_DIR.rglob("*.py"))
    contract_paths = _rq_q10_contract_paths()
    assert runtime_paths, "runtime inspection tree must be non-empty"
    assert contract_paths, "canonical contract surface must be non-empty"

    for path in _rq_q10_scan_paths():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        reflection = _rq_q10_reflection_hits(tree)
        assert not reflection, (
            f"{path.relative_to(_REPO_ROOT)} uses forbidden reflection names: {reflection}"
        )
        vendor_hits = _rq_q10_vendor_module_hits(_rq_q10_imported_modules(tree))
        assert not vendor_hits, (
            f"{path.relative_to(_REPO_ROOT)} imports vendor modules: {vendor_hits}"
        )

    for path in contract_paths:
        text = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_ABI_TOKENS:
            assert token not in text, (
                f"{path.relative_to(_REPO_ROOT)} contains forbidden ABI token {token!r}"
            )
        tree = ast.parse(text, filename=str(path))
        blocked = _rq_q10_forbidden_contract_imports(_rq_q10_imported_modules(tree))
        assert not blocked, (
            f"{path.relative_to(_REPO_ROOT)} imports concrete/runtime modules: {blocked}"
        )

    bad = _representative_reconstruction()
    events = list(bad.positioned_events)
    first = events[0].event.model_copy(update={"task_id": mint_task_id()})
    events[0] = _positioned(first, 1)
    with pytest.raises(RuntimeInspectionError) as exc_info:
        _enterprise_service(replace(bad, positioned_events=tuple(events))).inspect(
            RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC),
        )
    assert exc_info.value.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY
