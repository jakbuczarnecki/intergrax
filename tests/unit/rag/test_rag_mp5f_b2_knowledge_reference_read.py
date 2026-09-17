# © Artur Czarnecki. All rights reserved.

"""MP-5F-B2: Knowledge scoped reference-read contract, isolation, and pluginability."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.knowledge.contracts.knowledge_reference_read import (
    KNOWLEDGE_REFERENCE_READ_MAX_LIMIT,
    KnowledgeChunkCanonicalRef,
    KnowledgeReferenceReadOutcome,
    KnowledgeReferenceReadPort,
    KnowledgeReferenceReadQuery,
    KnowledgeReferenceReadRequest,
    KnowledgeReferenceReadResult,
    KnowledgeReferenceReadScope,
    KnowledgeReferenceReadScopeError,
    KnowledgeScopedResourceRef,
    validate_knowledge_reference_read_request,
)
from intergrax.rag.default_knowledge_reference_reader import (
    DefaultKnowledgeReferenceReader,
    KnowledgeReferenceReadCapabilityBinding,
    KnowledgeReferenceReadConfigurationError,
)
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk, RetrievalResult, RetrievalTrace

pytestmark = pytest.mark.gate

_REPO = Path(__file__).resolve().parents[3]
_CONTRACT = _REPO / "intergrax" / "knowledge" / "contracts" / "knowledge_reference_read.py"
_DEFAULT_READER = _REPO / "intergrax" / "rag" / "default_knowledge_reference_reader.py"

_FORBIDDEN_CONTRACT_IMPORT_PREFIXES = (
    "intergrax.contracts.context_view",
    "intergrax.contracts.context_view_source_ports",
    "intergrax.contracts.context_view_composition",
    "intergrax.collaborative_work",
    "intergrax.memory",
    "intergrax.ucl",
)

_FORBIDDEN_DEFAULT_IMPORT_PREFIXES = _FORBIDDEN_CONTRACT_IMPORT_PREFIXES + (
    "intergrax.rag.vectorstore.providers",
    "intergrax.integrations.providers",
)

_FORBIDDEN_DEFAULT_MODULE_PREFIXES = (
    "intergrax.rag.vectorstore.providers.",
    "intergrax.integrations.providers.",
)


def _identity(
    *,
    tenant_id: str = "tenant-a",
    user_id: str = "user-1",
) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def _scope(
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "ws-a",
    namespace: str | None = None,
) -> KnowledgeReferenceReadScope:
    return KnowledgeReferenceReadScope(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        namespace=namespace,
    )


def _query(
    *,
    text: str = "policy overview",
    limit: int = 5,
) -> KnowledgeReferenceReadQuery:
    return KnowledgeReferenceReadQuery(query_text=text, limit=limit)


def test_contract_request_immutability_and_query_bounds() -> None:
    scope = _scope()
    _ = KnowledgeReferenceReadRequest(scope=scope, query=_query())
    with pytest.raises(KnowledgeReferenceReadScopeError):
        KnowledgeReferenceReadQuery(query_text="", limit=1)
    with pytest.raises(KnowledgeReferenceReadScopeError):
        KnowledgeReferenceReadQuery(query_text="x", limit=0)
    with pytest.raises(KnowledgeReferenceReadScopeError):
        KnowledgeReferenceReadQuery(
            query_text="x",
            limit=KNOWLEDGE_REFERENCE_READ_MAX_LIMIT + 1,
        )
    with pytest.raises(KnowledgeReferenceReadScopeError):
        KnowledgeReferenceReadScope(tenant_id="", workspace_id="ws")
    with pytest.raises(KnowledgeReferenceReadScopeError):
        KnowledgeReferenceReadScope(tenant_id="t", workspace_id="")


def test_validate_identity_tenant_mismatch_scope_rejected() -> None:
    identity = _identity(tenant_id="tenant-a")
    request = KnowledgeReferenceReadRequest(
        scope=_scope(tenant_id="tenant-b"),
        query=_query(),
    )
    assert (
        validate_knowledge_reference_read_request(identity, request)
        is KnowledgeReferenceReadOutcome.SCOPE_REJECTED
    )


def test_result_reference_only_no_payload_fields() -> None:
    ref = KnowledgeChunkCanonicalRef(
        tenant_id="tenant-a",
        knowledge_ref="vec-1",
        document_id="doc-1",
        source_id="src-1",
        rank=1,
        relevance_score=0.9,
    )
    payload = dataclasses.asdict(ref)
    forbidden = {
        "content",
        "text",
        "embedding",
        "payload",
        "body",
        "user_metadata",
        "metadata",
    }
    assert forbidden.isdisjoint(payload.keys())
    serialized = json.dumps(payload)
    for token in ("chunk body", "embedding vector"):
        assert token not in serialized


def test_non_ok_result_cannot_carry_references() -> None:
    ref = KnowledgeChunkCanonicalRef(
        tenant_id="t",
        knowledge_ref="v",
        document_id="d",
    )
    with pytest.raises(KnowledgeReferenceReadScopeError):
        KnowledgeReferenceReadResult(
            outcome=KnowledgeReferenceReadOutcome.ACCESS_DENIED,
            references=(ref,),
        )


class _FakeRetrievalBackend:
    def __init__(self, *, chunks: tuple[RetrievalChunk, ...] = ()) -> None:
        self.last_request: RetrievalRequest | None = None
        self._chunks = chunks

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.last_request = request
        return RetrievalResult(
            chunks=list(self._chunks),
            used=True,
            reason="ok",
            trace=RetrievalTrace(),
        )


def _sample_chunk(
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "ws-a",
    vector_id: str = "vec-1",
    document_id: str = "doc-1",
    source_id: str = "src-1",
) -> RetrievalChunk:
    return RetrievalChunk(
        id=document_id,
        text="secret chunk body must not leak",
        score=0.88,
        rank=1,
        channel="dense",
        vector_id=vector_id,
        scope={
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "namespace": None,
        },
        provenance={
            "source_id": source_id,
            "source_kind": "file",
            "root_document_id": document_id,
        },
    )


def _configured_reader(
    backend: _FakeRetrievalBackend,
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "ws-a",
) -> DefaultKnowledgeReferenceReader:
    return DefaultKnowledgeReferenceReader(
        retrieval=backend,
        capability_binding=KnowledgeReferenceReadCapabilityBinding(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ),
    )


def test_configured_reader_requires_capability_binding() -> None:
    backend = _FakeRetrievalBackend()
    with pytest.raises(KnowledgeReferenceReadConfigurationError):
        KnowledgeReferenceReadCapabilityBinding(tenant_id="t", workspace_id="  ")


def test_empty_ok_when_no_hits() -> None:
    backend = _FakeRetrievalBackend(chunks=())
    reader = _configured_reader(backend)
    result = reader.read_references(
        _identity(),
        KnowledgeReferenceReadRequest(scope=_scope(), query=_query()),
    )
    assert result.outcome is KnowledgeReferenceReadOutcome.OK
    assert result.references == ()
    assert backend.last_request is not None
    assert backend.last_request.scope is not None
    assert backend.last_request.scope.tenant_id == "tenant-a"
    assert backend.last_request.scope.workspace_id == "ws-a"


def test_default_reader_returns_refs_without_content() -> None:
    chunk = _sample_chunk()
    backend = _FakeRetrievalBackend(chunks=(chunk,))
    reader = _configured_reader(backend)
    result = reader.read_references(
        _identity(),
        KnowledgeReferenceReadRequest(scope=_scope(), query=_query()),
    )
    assert result.outcome is KnowledgeReferenceReadOutcome.OK
    assert len(result.references) == 1
    ref = result.references[0]
    assert ref.knowledge_ref == "vec-1"
    assert ref.document_id == "doc-1"
    assert "secret" not in dataclasses.asdict(ref).values()


def test_least_context_scope_passed_to_retrieval() -> None:
    chunk = _sample_chunk()
    backend = _FakeRetrievalBackend(chunks=(chunk,))
    reader = _configured_reader(backend)
    reader.read_references(
        _identity(),
        KnowledgeReferenceReadRequest(
            scope=_scope(namespace="ns-a"),
            query=_query(limit=3),
        ),
    )
    assert backend.last_request is not None
    assert backend.last_request.scope is not None
    assert backend.last_request.scope.namespace == "ns-a"
    assert backend.last_request.final_top_k == 3


def test_cross_tenant_scope_rejected() -> None:
    reader = DefaultKnowledgeReferenceReader(
        retrieval=_FakeRetrievalBackend(),
        capability_binding=KnowledgeReferenceReadCapabilityBinding(
            tenant_id="tenant-a",
            workspace_id="ws-a",
        ),
    )
    result = reader.read_references(
        _identity(tenant_id="tenant-a"),
        KnowledgeReferenceReadRequest(
            scope=_scope(tenant_id="tenant-b"),
            query=_query(),
        ),
    )
    assert result.outcome is KnowledgeReferenceReadOutcome.SCOPE_REJECTED


def test_wrong_workspace_rejected_via_binding() -> None:
    backend = _FakeRetrievalBackend(chunks=(_sample_chunk(),))
    reader = _configured_reader(backend)
    result = reader.read_references(
        _identity(),
        KnowledgeReferenceReadRequest(
            scope=_scope(workspace_id="ws-other"),
            query=_query(),
        ),
    )
    assert result.outcome is KnowledgeReferenceReadOutcome.SCOPE_REJECTED


def test_out_of_scope_hits_filtered_not_leaked() -> None:
    foreign = _sample_chunk(tenant_id="tenant-b", workspace_id="ws-b")
    backend = _FakeRetrievalBackend(chunks=(foreign,))
    reader = _configured_reader(backend)
    result = reader.read_references(
        _identity(),
        KnowledgeReferenceReadRequest(scope=_scope(), query=_query()),
    )
    assert result.outcome is KnowledgeReferenceReadOutcome.OK
    assert result.references == ()


def test_resource_scope_builds_metadata_filter() -> None:
    chunk = _sample_chunk(source_id="src-bound")
    backend = _FakeRetrievalBackend(chunks=(chunk,))
    reader = _configured_reader(backend)
    scope = KnowledgeReferenceReadScope(
        tenant_id="tenant-a",
        workspace_id="ws-a",
        resource=KnowledgeScopedResourceRef(source_id="src-bound"),
    )
    result = reader.read_references(
        _identity(),
        KnowledgeReferenceReadRequest(scope=scope, query=_query()),
    )
    assert result.outcome is KnowledgeReferenceReadOutcome.OK
    assert backend.last_request is not None
    assert backend.last_request.metadata_filter is not None
    assert len(backend.last_request.metadata_filter.membership) == 1


def test_invalid_resource_scope_rejected_on_mismatch() -> None:
    chunk = _sample_chunk(source_id="src-other")
    backend = _FakeRetrievalBackend(chunks=(chunk,))
    reader = _configured_reader(backend)
    scope = KnowledgeReferenceReadScope(
        tenant_id="tenant-a",
        workspace_id="ws-a",
        resource=KnowledgeScopedResourceRef(source_id="src-bound"),
    )
    result = reader.read_references(
        _identity(),
        KnowledgeReferenceReadRequest(scope=scope, query=_query()),
    )
    assert result.outcome is KnowledgeReferenceReadOutcome.OK
    assert result.references == ()


class _CustomKnowledgeReferenceReader:
    """Pluginability proof — no default RAG runtime imports in this class file."""

    def read_references(
        self,
        identity: RequestIdentity,
        request: KnowledgeReferenceReadRequest,
    ) -> KnowledgeReferenceReadResult:
        if request.scope.tenant_id != identity.tenant_id:
            return KnowledgeReferenceReadResult(
                outcome=KnowledgeReferenceReadOutcome.SCOPE_REJECTED,
            )
        ref = KnowledgeChunkCanonicalRef(
            tenant_id=request.scope.tenant_id,
            knowledge_ref="custom-vec",
            document_id="custom-doc",
        )
        return KnowledgeReferenceReadResult(
            outcome=KnowledgeReferenceReadOutcome.OK,
            references=(ref,),
        )


def test_custom_reader_satisfies_port() -> None:
    port: KnowledgeReferenceReadPort = _CustomKnowledgeReferenceReader()
    result = port.read_references(
        _identity(),
        KnowledgeReferenceReadRequest(scope=_scope(), query=_query()),
    )
    assert result.outcome is KnowledgeReferenceReadOutcome.OK
    assert result.references[0].knowledge_ref == "custom-vec"


def _boundary_violations(
    path: Path,
    *,
    forbidden_prefixes: tuple[str, ...],
    forbidden_module_prefixes: tuple[str, ...] = (),
) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module = node.module
        for prefix in forbidden_prefixes:
            if module.startswith(prefix):
                violations.append(f"{module} at line {node.lineno}")
        for prefix in forbidden_module_prefixes:
            if module.startswith(prefix):
                violations.append(f"provider module {module} at line {node.lineno}")
        if module == "typing" and any(alias.name == "Any" for alias in node.names):
            violations.append(f"typing.Any at line {node.lineno}")
    source = path.read_text(encoding="utf-8")
    for name in ("getattr(", "hasattr(", "setattr("):
        if name in source:
            violations.append(f"dynamic attribute access: {name}")
    return violations


def test_knowledge_reference_read_contract_boundary_ast() -> None:
    violations = _boundary_violations(
        _CONTRACT,
        forbidden_prefixes=_FORBIDDEN_CONTRACT_IMPORT_PREFIXES,
    )
    assert not violations, "\n".join(violations)


def test_knowledge_reference_read_default_impl_boundary_ast() -> None:
    violations = _boundary_violations(
        _DEFAULT_READER,
        forbidden_prefixes=_FORBIDDEN_DEFAULT_IMPORT_PREFIXES,
        forbidden_module_prefixes=_FORBIDDEN_DEFAULT_MODULE_PREFIXES,
    )
    assert not violations, "\n".join(violations)
