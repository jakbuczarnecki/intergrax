# © Artur Czarnecki. All rights reserved.

"""MP-5F-B5-C1 — principal transport and authoritative source scope integrity."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.collaborative_work.context_view_source_adapters import (
    DefaultKnowledgeContextSource,
    DefaultMemoryContextSource,
)
from intergrax.collaborative_work.context_view_source_mapping import (
    candidate_scope_from_memory_evaluated,
)
from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.contracts.context_view import (
    ContextViewOperationScope,
    ContextViewScope,
    ContextViewVisibilityClass,
)
from intergrax.contracts.context_view_source_ports import (
    ContextViewKnowledgeSourceRequest,
    ContextViewMemorySourceRequest,
    ContextViewSourceOutcome,
)
from intergrax.contracts.context_view_visibility_policy import (
    suggested_context_view_source_visibility,
)
from intergrax.knowledge.contracts.knowledge_reference_read import (
    KnowledgeChunkCanonicalRef,
    KnowledgeReferenceReadOutcome,
    KnowledgeReferenceReadRequest,
    KnowledgeReferenceReadResult,
    KnowledgeReferenceReadScope,
    KnowledgeScopedResourceRef,
)
from intergrax.memory.contracts.memory_reference_read import (
    MemoryRecordCanonicalRef,
    MemoryReferenceReadOutcome,
    MemoryReferenceReadRequest,
    MemoryReferenceReadResult,
    MemoryReferenceReadScope,
    MemoryScopedResourceRef,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAPPING = _REPO_ROOT / "intergrax" / "collaborative_work" / "context_view_source_mapping.py"
_ADAPTER = _REPO_ROOT / "intergrax" / "collaborative_work" / "context_view_source_adapters.py"


class _ImmediateAsyncRunner:
    def run(self, coro):  # type: ignore[no-untyped-def]
        return asyncio.run(coro)


def _scope(**overrides: object) -> ContextViewScope:
    payload = {"tenant_id": "tenant-a", "workspace_id": "ws-1"}
    payload.update(overrides)
    return ContextViewScope(**payload)


def _identity(
    *,
    principal_id: str = "principal-1",
    principal_type: PrincipalType = PrincipalType.USER,
    auth_subject: str | None = None,
    user_id: str | None = None,
) -> RequestIdentity:
    subject = auth_subject if auth_subject is not None else principal_id
    uid = user_id if user_id is not None else principal_id
    return RequestIdentity(
        tenant_id="tenant-a",
        user_id=uid,
        principal_type=principal_type,
        auth_subject=subject,
    )


def _memory_request(**overrides: object) -> ContextViewMemorySourceRequest:
    payload = {
        "scope": _scope(),
        "acting_principal_id": "principal-1",
        "principal_identity": _identity(),
        "eligible_visibility_classes": (ContextViewVisibilityClass.WORKSPACE_SHARED,),
    }
    payload.update(overrides)
    return ContextViewMemorySourceRequest(**payload)


class _RecordingMemoryReader:
    def __init__(self, result: MemoryReferenceReadResult) -> None:
        self._result = result
        self.last_identity: RequestIdentity | None = None

    async def read_references(
        self,
        identity: RequestIdentity,
        request: MemoryReferenceReadRequest,
    ) -> MemoryReferenceReadResult:
        self.last_identity = identity
        return self._result


def test_principal_identity_mismatch_rejected_at_request_construction() -> None:
    with pytest.raises(ValidationError, match="acting_principal_id"):
        ContextViewMemorySourceRequest(
            scope=_scope(),
            acting_principal_id="principal-1",
            principal_identity=_identity(principal_id="other"),
            eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
        )


def test_user_principal_preserved_to_memory_reader() -> None:
    identity = _identity(principal_id="user-a", principal_type=PrincipalType.USER)
    evaluated = MemoryReferenceReadScope(
        tenant_id="tenant-a",
        workspace_id="ws-1",
        user_id="user-a",
    )
    ref = MemoryRecordCanonicalRef(tenant_id="tenant-a", memory_id="m1", revision=1)
    reader = _RecordingMemoryReader(
        MemoryReferenceReadResult(
            outcome=MemoryReferenceReadOutcome.OK,
            references=(ref,),
            evaluated_scope=evaluated,
        ),
    )
    adapter = DefaultMemoryContextSource(reader=reader, async_runner=_ImmediateAsyncRunner())
    result = adapter.list_candidates(
        _memory_request(
            acting_principal_id="user-a",
            principal_identity=identity,
        ),
    )
    assert result.outcome is ContextViewSourceOutcome.OK
    assert reader.last_identity is not None
    assert reader.last_identity.principal_type is PrincipalType.USER
    assert reader.last_identity.auth_subject == "user-a"


def test_service_principal_preserved_as_agent_semantic() -> None:
    identity = RequestIdentity(
        tenant_id="tenant-a",
        user_id="agent-runtime-1",
        principal_type=PrincipalType.SERVICE,
        auth_subject="agent-runtime-1",
    )
    evaluated = MemoryReferenceReadScope(tenant_id="tenant-a", workspace_id="ws-1")
    ref = MemoryRecordCanonicalRef(tenant_id="tenant-a", memory_id="m1", revision=1)
    reader = _RecordingMemoryReader(
        MemoryReferenceReadResult(
            outcome=MemoryReferenceReadOutcome.OK,
            references=(ref,),
            evaluated_scope=evaluated,
        ),
    )
    adapter = DefaultMemoryContextSource(reader=reader, async_runner=_ImmediateAsyncRunner())
    adapter.list_candidates(
        _memory_request(
            acting_principal_id="agent-runtime-1",
            principal_identity=identity,
        ),
    )
    assert reader.last_identity is not None
    assert reader.last_identity.principal_type is PrincipalType.SERVICE
    assert reader.last_identity.user_id == "agent-runtime-1"


def test_memory_wrong_workspace_in_evaluated_scope_rejected() -> None:
    ref = MemoryRecordCanonicalRef(tenant_id="tenant-a", memory_id="m1", revision=1)
    reader = _RecordingMemoryReader(
        MemoryReferenceReadResult(
            outcome=MemoryReferenceReadOutcome.OK,
            references=(ref,),
            evaluated_scope=MemoryReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-other",
            ),
        ),
    )
    adapter = DefaultMemoryContextSource(reader=reader, async_runner=_ImmediateAsyncRunner())
    outcome = adapter.list_candidates(_memory_request()).outcome
    assert outcome is ContextViewSourceOutcome.SCOPE_REJECTED


def test_memory_work_item_mismatch_rejected() -> None:
    ref = MemoryRecordCanonicalRef(tenant_id="tenant-a", memory_id="m1", revision=1)
    reader = _RecordingMemoryReader(
        MemoryReferenceReadResult(
            outcome=MemoryReferenceReadOutcome.OK,
            references=(ref,),
            evaluated_scope=MemoryReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-1",
                resource=MemoryScopedResourceRef(resource_kind="work_item", resource_id="wi-b"),
            ),
        ),
    )
    adapter = DefaultMemoryContextSource(reader=reader, async_runner=_ImmediateAsyncRunner())
    outcome = adapter.list_candidates(
        _memory_request(scope=_scope(work_item_id="wi-a")),
    ).outcome
    assert outcome is ContextViewSourceOutcome.SCOPE_REJECTED


def test_memory_candidate_scope_from_evaluated_authority() -> None:
    ref = MemoryRecordCanonicalRef(tenant_id="tenant-a", memory_id="m1", revision=1)
    reader = _RecordingMemoryReader(
        MemoryReferenceReadResult(
            outcome=MemoryReferenceReadOutcome.OK,
            references=(ref,),
            evaluated_scope=MemoryReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-1",
                resource=MemoryScopedResourceRef(resource_kind="work_item", resource_id="wi-1"),
            ),
        ),
    )
    adapter = DefaultMemoryContextSource(reader=reader, async_runner=_ImmediateAsyncRunner())
    result = adapter.list_candidates(
        _memory_request(scope=_scope(work_item_id="wi-1")),
    )
    assert result.outcome is ContextViewSourceOutcome.OK
    assert result.candidates[0].candidate_scope.work_item_id == "wi-1"
    assert result.candidates[0].candidate_scope.workspace_id == "ws-1"


class _RecordingKnowledgeReader:
    def __init__(self, result: KnowledgeReferenceReadResult) -> None:
        self._result = result

    def read_references(
        self,
        identity: RequestIdentity,
        request: KnowledgeReferenceReadRequest,
    ) -> KnowledgeReferenceReadResult:
        return self._result


def test_knowledge_wrong_document_in_evaluated_scope_rejected() -> None:
    ref = KnowledgeChunkCanonicalRef(
        tenant_id="tenant-a",
        knowledge_ref="vec-1",
        document_id="doc-a",
    )
    reader = _RecordingKnowledgeReader(
        KnowledgeReferenceReadResult(
            outcome=KnowledgeReferenceReadOutcome.OK,
            references=(ref,),
            evaluated_scope=KnowledgeReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-1",
                resource=KnowledgeScopedResourceRef(document_id="doc-b"),
            ),
        ),
    )
    adapter = DefaultKnowledgeContextSource(reader=reader)
    req = ContextViewKnowledgeSourceRequest(
        scope=_scope(
            operation_scope=ContextViewOperationScope(
                operation_id="op.read",
                resource_scope="doc-a",
            ),
        ),
        acting_principal_id="principal-1",
        principal_identity=_identity(),
        eligible_visibility_classes=(ContextViewVisibilityClass.WORKSPACE_SHARED,),
        reference_read_query_text="enumerate",
    )
    assert adapter.list_candidates(req).outcome is ContextViewSourceOutcome.SCOPE_REJECTED


def test_visibility_selection_uses_mp5c_canonical_priority() -> None:
    classes = (
        ContextViewVisibilityClass.PRIVATE_TO_PRINCIPAL,
        ContextViewVisibilityClass.WORKSPACE_SHARED,
        ContextViewVisibilityClass.WORK_ITEM,
    )
    assert suggested_context_view_source_visibility(classes) is (
        ContextViewVisibilityClass.PRIVATE_TO_PRINCIPAL
    )


def _forbidden_reflection_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"getattr", "hasattr", "setattr"}:
                violations.append(f"{path.name}:{node.lineno} uses {node.func.id}")
    return violations


def test_no_reflection_in_adapter_boundary() -> None:
    assert _forbidden_reflection_calls(_MAPPING) == []
    assert _forbidden_reflection_calls(_ADAPTER) == []


def test_no_private_contract_import_in_mapping() -> None:
    text = _MAPPING.read_text(encoding="utf-8")
    assert "_ContextViewSourceRequestBase" not in text


def test_memory_candidate_mapping_never_fabricates_from_request_scope() -> None:
    tree = ast.parse(_MAPPING.read_text(encoding="utf-8"), filename=str(_MAPPING))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "candidate_scope_from_memory_evaluated":
            continue
        segment = ast.get_source_segment(_MAPPING.read_text(encoding="utf-8"), node) or ""
        assert "request_scope" not in segment
        return
    raise AssertionError("candidate_scope_from_memory_evaluated not found")


def test_memory_candidate_scope_does_not_claim_unproven_work_item_or_operation() -> None:
    evaluated = MemoryReferenceReadScope(tenant_id="tenant-a", workspace_id="ws-1")
    request = _scope(
        work_item_id="work-item-a1",
        operation_scope=ContextViewOperationScope(
            operation_id="op-compose",
            resource_scope="context-a",
        ),
    )
    candidate = candidate_scope_from_memory_evaluated(evaluated)
    assert candidate.work_item_id is None
    assert candidate.operation_scope is None
    ref = MemoryRecordCanonicalRef(tenant_id="tenant-a", memory_id="m1", revision=1)
    reader = _RecordingMemoryReader(
        MemoryReferenceReadResult(
            outcome=MemoryReferenceReadOutcome.OK,
            references=(ref,),
            evaluated_scope=evaluated,
        ),
    )
    adapter = DefaultMemoryContextSource(reader=reader, async_runner=_ImmediateAsyncRunner())
    result = adapter.list_candidates(_memory_request(scope=request))
    assert result.outcome is ContextViewSourceOutcome.OK
    assert result.candidates[0].candidate_scope.work_item_id is None
    assert result.candidates[0].candidate_scope.operation_scope is None
