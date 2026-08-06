# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Hybrid Ask execution tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import pytest
from local_workspace_application.workspaces.hybrid_ask_execution import (
    KnowledgeQueryExecutionResultV1,
    KnowledgeQueryOrchestratorV1,
    LiveCapabilityExecutionResultV1,
    LiveCapabilityExecutorV1,
    LiveCapabilityHandlerRegistryV1,
    LiveCapabilityResultItemV1,
    LiveExecutionOutcomeV1,
    WorkspaceIndexedEvidenceRetrieverV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    AskAudienceV1,
    HybridAskIndexedRetrievalStatusV1,
    HybridAskLiveExecutionStatusV1,
    HybridAskTruncationStateV1,
    IndexedWorkspaceEvidenceV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    EffectiveLiveCallBudgetV1,
    EvidencePlanV1,
    IndexedRetrievalDirectiveV1,
    KnowledgeQueryAudienceV1,
    ResolvedLiveResourceScopeV1,
    ValidatedEvidencePlanV1,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    KnowledgeAudienceEligibilityV1,
    LiveResultRetentionV1,
    QueryPolicyModeV2,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeConfigurationHead,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceSource,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_TENANT = "tenant-1"
_WORKSPACE = "workspace-1"
_KIND = IntegrationCategory.WIKI_KNOWLEDGE
_PROVIDER = "neutral_provider"
_CAPABILITY = "vendor.neutral_provider.issues.read"


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    query: str


def _budget(
    *,
    max_live_calls: int = 2,
    max_total_duration_ms: int = 30_000,
    max_result_items: int = 10,
    max_result_bytes: int = 1_000,
) -> EffectiveLiveCallBudgetV1:
    return EffectiveLiveCallBudgetV1(
        max_live_calls=max_live_calls,
        max_total_duration_ms=max_total_duration_ms,
        max_result_items=max_result_items,
        max_result_bytes=max_result_bytes,
    )


def _call(
    call_id: str = "call-1",
    *,
    budget: EffectiveLiveCallBudgetV1 | None = None,
    capability_id: str = _CAPABILITY,
) -> Any:
    from local_workspace_application.workspaces.hybrid_ask_policy import (
        ExecutableLiveCallV1,
    )

    return ExecutableLiveCallV1(
        call_id=call_id,
        live_access_binding_id="binding-1",
        connection_ref="connection-1",
        provider_id=_PROVIDER,
        integration_kind=_KIND,
        capability_id=capability_id,
            contract_version="1",
            source_kind="issues",
        validated_request=_Request(query="validated"),
        remote_resource_id="resource-1",
        resolved_resource_scope=ResolvedLiveResourceScopeV1(
            remote_resource_id="resource-1",
            scope_token="validated-scope",
        ),
        effective_budget=budget or _budget(),
    )


def _plan(
    mode: QueryPolicyModeV2,
    *,
    calls: tuple[Any, ...] = (),
    directive: IndexedRetrievalDirectiveV1 | None = None,
    budget: EffectiveLiveCallBudgetV1 | None = None,
    audience: KnowledgeQueryAudienceV1 = KnowledgeQueryAudienceV1.PERSONAL,
) -> ValidatedEvidencePlanV1:
    effective_budget = budget or _budget(max_live_calls=max(len(calls), 1))
    return ValidatedEvidencePlanV1(
        plan=EvidencePlanV1(
            plan_id=f"plan-{mode.value}",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=1,
            mode=mode,
            indexed_retrieval_directive=directive,
            ordered_live_call_proposals=(),
            budget_snapshot=effective_budget,
            audience_context=AudienceContextV1(
                audience=audience
            ),
        ),
        executable_live_calls=calls,
        effective_budget=effective_budget,
    )


def _item(
    remote_item_id: str,
    content: str = "live content",
) -> LiveCapabilityResultItemV1:
    from hashlib import sha256

    return LiveCapabilityResultItemV1(
        remote_item_id=remote_item_id,
        safe_display_name=f"Item {remote_item_id}",
        content=content,
        content_hash=sha256(content.encode()).hexdigest(),
        retrieved_at=_NOW,
        remote_updated_at=_NOW,
    )


def _live_result(
    call_id: str,
    items: tuple[LiveCapabilityResultItemV1, ...],
    *,
    item_count: int | None = None,
    byte_count: int | None = None,
) -> LiveCapabilityExecutionResultV1:
    return LiveCapabilityExecutionResultV1(
        call_id=call_id,
        normalized_outcome=LiveExecutionOutcomeV1.COMPLETED,
        items=items,
        item_count=item_count if item_count is not None else len(items),
        byte_count=byte_count
        if byte_count is not None
        else sum(len(item.content.encode()) for item in items),
        started_at=_NOW,
        completed_at=_NOW,
    )


class _NeutralIntegration:
    provider_id = _PROVIDER
    integration_kind = _KIND.value


class _RecordingResolver:
    def __init__(self, integration: object | None = None) -> None:
        self.integration = integration or _NeutralIntegration()
        self.calls: list[dict[str, object]] = []

    def resolve(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.integration


class _NeutralHandler:
    provider_id = _PROVIDER
    integration_kind = _KIND
    source_kind = "issues"
    capability_id = _CAPABILITY
    contract_version = "1"
    request_schema_ref = "schema://vendor-knowledge/live/neutral_provider/issues/read/request/v1"
    result_schema_ref = "schema://vendor-knowledge/live/neutral_provider/issues/read/result/v1"
    expected_request_model = _Request

    def __init__(
        self,
        items: tuple[LiveCapabilityResultItemV1, ...] = (),
        *,
        failure: Exception | None = None,
        delay: float = 0,
    ) -> None:
        self.items = items
        self.failure = failure
        self.delay = delay
        self.calls: list[tuple[object, object, object]] = []

    async def execute(self, *, integration: object, call: object, context: object):
        self.calls.append((integration, call, context))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failure is not None:
            raise self.failure
        return _live_result(call.call_id, self.items)


class _NeutralRetriever:
    def __init__(self, evidence: tuple[IndexedWorkspaceEvidenceV1, ...] = ()) -> None:
        self.evidence = evidence
        self.calls: list[dict[str, object]] = []

    async def retrieve(self, **kwargs: object):
        self.calls.append(kwargs)
        return self.evidence


def _indexed_binding(
    *,
    audience_eligibility: KnowledgeAudienceEligibilityV1 = (
        KnowledgeAudienceEligibilityV1.PERSONAL_ONLY
    ),
    status: WorkspaceIndexedSourceBindingStatusV1 = (
        WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    ),
    binding_id: str = "binding-1",
    source_id: str = "source-1",
) -> WorkspaceIndexedSourceBinding:
    return WorkspaceIndexedSourceBinding(
        indexed_source_binding_id=binding_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_source_binding_ref="knowledge-source-1",
        source_id=source_id,
        status=status,
        audience_eligibility=audience_eligibility,
        mutation_id="mutation-1",
        effective_revision=1,
        semantic_identity_hash=sha256(binding_id.encode()).hexdigest(),
        created_at=_NOW,
        updated_at=_NOW,
    )


class _AuthoritativeRepository:
    def __init__(
        self,
        binding: WorkspaceIndexedSourceBinding,
        *,
        document_tenant_id: str = _TENANT,
        document_workspace_id: str = _WORKSPACE,
        source_tenant_id: str = _TENANT,
        source_workspace_id: str = _WORKSPACE,
    ) -> None:
        self.binding = binding
        self.workspace = Workspace(
            workspace_id=_WORKSPACE,
            tenant_id=_TENANT,
            name="Workspace",
            created_at=_NOW,
            updated_at=_NOW,
        )
        self.document = WorkspaceDocumentReference(
            document_id="document-1",
            tenant_id=document_tenant_id,
            workspace_id=document_workspace_id,
            source_id=binding.source_id,
            source_path="docs/document.txt",
            file_name="document.txt",
            content_hash="sha256:" + "a" * 64,
            indexed_at=_NOW,
        )
        self.source = WorkspaceSource(
            source_id=binding.source_id,
            tenant_id=source_tenant_id,
            workspace_id=source_workspace_id,
            path="docs",
            created_at=_NOW,
        )

    def get_workspace(self, **kwargs: object) -> Workspace | None:
        return self.workspace

    def get_knowledge_configuration_head(
        self, **kwargs: object
    ) -> WorkspaceKnowledgeConfigurationHead:
        return WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=1,
            updated_at=_NOW,
        )

    def list_knowledge_connection_attachment_versions(
        self, **kwargs: object
    ) -> list[object]:
        return []

    def list_knowledge_indexed_source_versions(
        self, **kwargs: object
    ) -> list[WorkspaceIndexedSourceBinding]:
        return [self.binding]

    def list_knowledge_live_access_versions(self, **kwargs: object) -> list[object]:
        return []

    def list_knowledge_query_policy_versions(self, **kwargs: object) -> list[object]:
        return []

    def get_document_ref(self, **kwargs: object) -> WorkspaceDocumentReference | None:
        return self.document

    def get_source(self, **kwargs: object) -> WorkspaceSource | None:
        return self.source


class _SearchExecution:
    def __init__(self, evidence: list[dict[str, object]]) -> None:
        self.structured_data = {"search_summary": {"evidence": evidence}}


class _SearchTaskResult:
    def __init__(self, evidence: list[dict[str, object]]) -> None:
        self.agent_id = "agent-1"
        self.run_id = "run-1"
        self.task_id = "task-1"
        self.metadata: dict[str, object] = {}
        self.execution_result = _SearchExecution(evidence)

    def model_copy(self, *, update: dict[str, object]) -> _SearchTaskResult:
        self.metadata = dict(update["metadata"])  # type: ignore[arg-type]
        return self


class _SearchTaskExecutor:
    def __init__(self, result: _SearchTaskResult) -> None:
        self.result = result

    async def execute(self, task: object) -> _SearchTaskResult:
        return self.result


def _authoritative_retriever(
    binding: WorkspaceIndexedSourceBinding,
    *,
    metadata: dict[str, object] | None = None,
    repository: _AuthoritativeRepository | None = None,
) -> WorkspaceIndexedEvidenceRetrieverV1:
    repo = repository or _AuthoritativeRepository(binding)
    search_metadata = {"indexed_source_binding_id": binding.indexed_source_binding_id}
    if metadata is not None:
        search_metadata.update(metadata)
    result = _SearchTaskResult(
        [
            {
                "document_id": "document-1",
                "source_id": binding.source_id,
                "workspace_id": _WORKSPACE,
                "source_path": "docs/document.txt",
                "file_name": "document.txt",
                "score": 0.9,
                "snippet": "indexed content",
                "metadata": search_metadata,
            }
        ]
    )
    return WorkspaceIndexedEvidenceRetrieverV1(
        task_executor=_SearchTaskExecutor(result),  # type: ignore[arg-type]
        workspace_repository=repo,  # type: ignore[arg-type]
        clock=lambda: _NOW,
    )


def _indexed_evidence() -> IndexedWorkspaceEvidenceV1:
    content = "indexed content"
    return IndexedWorkspaceEvidenceV1(
        evidence_id="idx:workspace-1:document-1:chunk-1",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        safe_display_name="document.txt",
        retrieved_at=_NOW,
        content=content,
        content_hash=sha256(content.encode()).hexdigest(),
        audience=AskAudienceV1.PERSONAL,
        source_id="source-1",
        document_id="document-1",
        chunk_id="chunk-1",
        score=0.9,
    )


def _retrieve_authoritative(
    retriever: WorkspaceIndexedEvidenceRetrieverV1,
    *,
    audience: KnowledgeQueryAudienceV1 = KnowledgeQueryAudienceV1.PERSONAL,
) -> tuple[IndexedWorkspaceEvidenceV1, ...]:
    return asyncio.run(
        retriever.retrieve(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=1,
            question="question",
            directive=IndexedRetrievalDirectiveV1(max_results=5),
            audience_context=AudienceContextV1(audience=audience),
        )
    )


def test_indexed_retriever_accepts_personal_eligible_source() -> None:
    binding = _indexed_binding()

    evidence = _retrieve_authoritative(_authoritative_retriever(binding))

    assert len(evidence) == 1
    assert evidence[0].audience is AskAudienceV1.PERSONAL
    assert evidence[0].indexed_source_binding_id == binding.indexed_source_binding_id


def test_indexed_retriever_accepts_shared_eligible_source_for_shared_request() -> None:
    binding = _indexed_binding(
        audience_eligibility=KnowledgeAudienceEligibilityV1.SHARED_ALLOWED
    )

    evidence = _retrieve_authoritative(
        _authoritative_retriever(binding),
        audience=KnowledgeQueryAudienceV1.SHARED,
    )

    assert evidence[0].audience is AskAudienceV1.SHARED


def test_indexed_retriever_rejects_personal_only_source_for_shared_request() -> None:
    binding = _indexed_binding()

    with pytest.raises(ValueError, match="indexed_source_shared_access_forbidden"):
        _retrieve_authoritative(
            _authoritative_retriever(binding),
            audience=KnowledgeQueryAudienceV1.SHARED,
        )


def test_indexed_retriever_rejects_disabled_historical_binding() -> None:
    binding = _indexed_binding(status=WorkspaceIndexedSourceBindingStatusV1.DISABLED)

    with pytest.raises(ValueError, match="indexed_source_binding_ambiguous"):
        _retrieve_authoritative(_authoritative_retriever(binding))


def test_search_metadata_cannot_override_authoritative_binding_eligibility() -> None:
    binding = _indexed_binding()

    with pytest.raises(ValueError, match="indexed_source_shared_access_forbidden"):
        _retrieve_authoritative(
            _authoritative_retriever(
                binding,
                metadata={
                    "indexed_source_binding_id": binding.indexed_source_binding_id,
                    "shared_allowed": True,
                    "audience": "shared",
                },
            ),
            audience=KnowledgeQueryAudienceV1.SHARED,
        )


def test_indexed_retriever_rejects_binding_id_mismatch() -> None:
    binding = _indexed_binding()

    with pytest.raises(ValueError, match="indexed_source_binding_metadata_mismatch"):
        _retrieve_authoritative(
            _authoritative_retriever(
                binding,
                metadata={"indexed_source_binding_id": "binding-attacker"},
            )
        )


@pytest.mark.parametrize(
    ("repository_kwargs", "error_match"),
    [
        ({"document_tenant_id": "tenant-other"}, "search_evidence_unverified"),
        ({"source_workspace_id": "workspace-other"}, "indexed_source_ownership_unverified"),
    ],
)
def test_indexed_retriever_fails_closed_on_tenant_or_workspace_mismatch(
    repository_kwargs: dict[str, str],
    error_match: str,
) -> None:
    binding = _indexed_binding()
    repository = _AuthoritativeRepository(binding, **repository_kwargs)

    with pytest.raises(Exception, match=error_match):
        _retrieve_authoritative(
            _authoritative_retriever(binding, repository=repository)
        )


def _executor(
    handler: Any,
    resolver: _RecordingResolver | None = None,
    *,
    clock=lambda: _NOW,
    monotonic=lambda: 100.0,
    ids=lambda: "receipt-1",
) -> tuple[LiveCapabilityExecutorV1, _RecordingResolver]:
    actual_resolver = resolver or _RecordingResolver()
    return (
        LiveCapabilityExecutorV1(
            handler_registry=LiveCapabilityHandlerRegistryV1((handler,)),
            integration_resolver=actual_resolver,
            clock=clock,
            monotonic=monotonic,
            id_factory=ids,
        ),
        actual_resolver,
    )


def test_registry_requires_exact_identity_and_isolates_capability_collisions() -> None:
    first = _NeutralHandler()

    class _OtherProvider(_NeutralHandler):
        provider_id = "other_provider"
        capability_id = "vendor.other_provider.issues.read"
        request_schema_ref = (
            "schema://vendor-knowledge/live/other_provider/issues/read/request/v1"
        )
        result_schema_ref = (
            "schema://vendor-knowledge/live/other_provider/issues/read/result/v1"
        )

    registry = LiveCapabilityHandlerRegistryV1((first, _OtherProvider()))
    assert registry.resolve(
        provider_id=_PROVIDER,
        integration_kind=_KIND,
        capability_id=_CAPABILITY,
    ) is first
    with pytest.raises(LookupError, match="live_capability_unavailable"):
        registry.resolve(
            provider_id="missing-provider",
            integration_kind=_KIND,
            capability_id=_CAPABILITY,
        )

    with pytest.raises(ValueError, match="duplicate_live_handler_identity"):
        LiveCapabilityHandlerRegistryV1((first, _NeutralHandler()))


def test_executor_resolves_exact_tenant_connection_and_forwards_validated_call() -> None:
    handler = _NeutralHandler((_item("item-1"),))
    resolver = _RecordingResolver()
    executor, _ = _executor(handler, resolver)

    result = asyncio.run(
        executor.execute(
            run_id="run-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            call=_call(),
            audience=KnowledgeQueryAudienceV1.PERSONAL,
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )

    assert result.error_code is None
    assert resolver.calls == [
        {
            "tenant_id": _TENANT,
            "connection_ref": "connection-1",
            "provider_id": _PROVIDER,
            "integration_kind": _KIND,
        }
    ]
    assert handler.calls[0][0] is resolver.integration
    assert handler.calls[0][1].validated_request == _Request(query="validated")
    assert (
        handler.calls[0][1].resolved_resource_scope.scope_token
        == "validated-scope"
    )


def test_unavailable_connection_is_normalized_without_handler_invocation() -> None:
    handler = _NeutralHandler((_item("item-1"),))

    class _UnavailableResolver:
        def resolve(self, **kwargs: object) -> object:
            raise RuntimeError("credential details must not escape")

    executor, _ = _executor(handler, _UnavailableResolver())  # type: ignore[arg-type]
    result = asyncio.run(
        executor.execute(
            run_id="run-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            call=_call(),
            audience=KnowledgeQueryAudienceV1.PERSONAL,
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )

    assert result.error_code == "live_binding_unavailable"
    assert handler.calls == []


def test_live_execution_normalizes_items_ids_receipts_and_ephemeral_retention() -> None:
    handler = _NeutralHandler((_item("item-1"), _item("item-2")))
    executor, _ = _executor(handler)
    result = asyncio.run(
        executor.execute(
            run_id="run-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            call=_call(),
            audience=KnowledgeQueryAudienceV1.PERSONAL,
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )

    assert result.receipt is None
    assert result.item_count == 2
    assert result.byte_count == len(b"live content") * 2
    assert [item.remote_item_id for item in result.items] == ["item-1", "item-2"]


def test_receipt_only_returns_one_safe_receipt_without_body() -> None:
    handler = _NeutralHandler((_item("item-1"),))
    executor, _ = _executor(handler)
    result = asyncio.run(
        executor.execute(
            run_id="run-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            call=_call(),
            audience=KnowledgeQueryAudienceV1.PERSONAL,
            retention=LiveResultRetentionV1.RECEIPT_ONLY,
        )
    )

    assert result.receipt is not None
    dumped = result.receipt.model_dump()
    assert dumped["item_count"] == 1
    assert "content" not in dumped
    assert "excerpt" not in dumped
    assert "provider_client" not in dumped


def test_budget_truncation_recomputes_counts_and_is_explicit() -> None:
    handler = _NeutralHandler((_item("item-1", "12345"), _item("item-2", "67890")))
    executor, _ = _executor(handler)
    result = asyncio.run(
        executor.execute(
            run_id="run-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            call=_call(
                budget=_budget(max_result_items=1, max_result_bytes=5)
            ),
            audience=KnowledgeQueryAudienceV1.PERSONAL,
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )

    assert result.truncated is True
    assert result.item_count == 1
    assert result.byte_count == 5
    assert result.items[0].content == "12345"
    assert len(result.items[0].content_hash) == 64


def test_handler_exception_and_malformed_result_are_safe() -> None:
    failed, _ = _executor(_NeutralHandler(failure=RuntimeError("secret provider text")))
    failure = asyncio.run(
        failed.execute(
            run_id="run-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            call=_call(),
            audience=KnowledgeQueryAudienceV1.PERSONAL,
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )
    assert failure.error_code == "live_execution_failed"
    assert "secret" not in str(failure)

    class _Malformed:
        provider_id = _PROVIDER
        integration_kind = _KIND
        source_kind = "issues"
        capability_id = _CAPABILITY
        contract_version = "1"
        request_schema_ref = _NeutralHandler.request_schema_ref
        result_schema_ref = _NeutralHandler.result_schema_ref
        expected_request_model = _Request

        def __init__(self) -> None:
            self.calls: list[tuple[object, object, object]] = []

        async def execute(self, **kwargs: object) -> object:
            self.calls.append((kwargs["integration"], kwargs["call"], kwargs["context"]))
            return {"call_id": "call-1", "items": [{"provider_client": object()}]}

    malformed, _ = _executor(_Malformed())
    invalid = asyncio.run(
        malformed.execute(
            run_id="run-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            call=_call(),
            audience=KnowledgeQueryAudienceV1.PERSONAL,
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )
    assert invalid.error_code == "live_result_invalid"


def test_timeout_is_normalized_without_raw_exception() -> None:
    handler = _NeutralHandler((_item("item-1"),), delay=0.05)
    executor, _ = _executor(handler)
    result = asyncio.run(
        executor.execute(
            run_id="run-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            call=_call(budget=_budget(max_total_duration_ms=1)),
            audience=KnowledgeQueryAudienceV1.PERSONAL,
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )
    assert result.error_code == "live_execution_timeout"


def test_orchestrator_mode_matrix_and_hybrid_ordering() -> None:
    indexed = _NeutralRetriever((_indexed_evidence(),))
    handler = _NeutralHandler((_item("item-1"),))
    live_executor, _ = _executor(handler)
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed,
        live_executor=live_executor,
    )

    indexed_result = asyncio.run(
        orchestrator.execute(
            run_id="run-indexed",
            question="question",
            validated_plan=_plan(
                QueryPolicyModeV2.INDEXED_ONLY,
                directive=IndexedRetrievalDirectiveV1(max_results=5),
            ),
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )
    assert indexed_result.indexed_retrieval_status is HybridAskIndexedRetrievalStatusV1.COMPLETED
    assert indexed_result.live_execution_status is HybridAskLiveExecutionStatusV1.SKIPPED
    assert len(indexed_result.indexed_evidence) == 1
    assert not handler.calls

    live_result = asyncio.run(
        orchestrator.execute(
            run_id="run-live",
            question="question",
            validated_plan=_plan(
                QueryPolicyModeV2.LIVE_ONLY,
                calls=(_call(),),
            ),
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )
    assert live_result.indexed_retrieval_status is HybridAskIndexedRetrievalStatusV1.SKIPPED
    assert live_result.live_execution_status is HybridAskLiveExecutionStatusV1.COMPLETED
    assert len(indexed.calls) == 1


def test_hybrid_forwards_audience_and_fails_closed_on_indexed_failure() -> None:
    indexed = _NeutralRetriever((_indexed_evidence(),))
    handler = _NeutralHandler((_item("item-1"),))
    live_executor, _ = _executor(handler)
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed,
        live_executor=live_executor,
    )
    result = asyncio.run(
        orchestrator.execute(
            run_id="run-hybrid",
            question="question",
            validated_plan=_plan(
                QueryPolicyModeV2.HYBRID,
                calls=(_call(),),
                directive=IndexedRetrievalDirectiveV1(max_results=5),
            ),
            retention=LiveResultRetentionV1.RECEIPT_ONLY,
        )
    )
    assert len(result.indexed_evidence) == 1
    assert len(result.live_evidence) == 1
    assert result.error_code is None
    assert result.receipts
    assert indexed.calls[0]["tenant_id"] == _TENANT
    assert indexed.calls[0]["workspace_id"] == _WORKSPACE
    assert indexed.calls[0]["question"] == "question"

    class _FailingRetriever:
        async def retrieve(self, **kwargs: object):
            raise RuntimeError("retrieval failure")

    failed_orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=_FailingRetriever(),
        live_executor=live_executor,
    )
    failed = asyncio.run(
        failed_orchestrator.execute(
            run_id="run-hybrid-failed",
            question="question",
            validated_plan=_plan(
                QueryPolicyModeV2.HYBRID,
                calls=(_call(),),
                directive=IndexedRetrievalDirectiveV1(max_results=5),
            ),
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )
    assert failed.error_code == "indexed_retrieval_failed"
    assert not handler.calls[1:]


@pytest.mark.parametrize(
    "evidence_factory",
    [
        lambda: (_indexed_evidence().model_copy(update={"tenant_id": "tenant-other"}),),
        lambda: (_indexed_evidence().model_copy(update={"workspace_id": "workspace-other"}),),
        lambda: (
            _indexed_evidence().model_copy(update={"audience": AskAudienceV1.SHARED}),
        ),
        lambda: (_indexed_evidence(), _indexed_evidence()),
        lambda: (_indexed_evidence().model_copy(update={"content_hash": "f" * 64}),),
    ],
    ids=("wrong-tenant", "wrong-workspace", "wrong-audience", "duplicate-id", "bad-hash"),
)
def test_orchestrator_indexed_result_fence_blocks_invalid_evidence_and_live(
    evidence_factory: Any,
) -> None:
    indexed = _NeutralRetriever(tuple(evidence_factory()))
    handler = _NeutralHandler((_item("item-1"),))
    live_executor, _ = _executor(handler)
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed,
        live_executor=live_executor,
    )

    result = asyncio.run(
        orchestrator.execute(
            run_id="run-hybrid-invalid-indexed",
            question="question",
            validated_plan=_plan(
                QueryPolicyModeV2.HYBRID,
                calls=(_call(),),
                directive=IndexedRetrievalDirectiveV1(max_results=5),
            ),
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )

    assert result.error_code == "indexed_retrieval_failed"
    assert result.indexed_evidence == ()
    assert handler.calls == []


def test_hybrid_does_not_downgrade_after_required_live_failure() -> None:
    indexed = _NeutralRetriever((_indexed_evidence(),))
    handler = _NeutralHandler(failure=RuntimeError("provider failure"))
    live_executor, _ = _executor(handler)
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed,
        live_executor=live_executor,
    )
    result = asyncio.run(
        orchestrator.execute(
            run_id="run-hybrid-live-failed",
            question="question",
            validated_plan=_plan(
                QueryPolicyModeV2.HYBRID,
                calls=(_call(),),
                directive=IndexedRetrievalDirectiveV1(max_results=5),
            ),
            retention=LiveResultRetentionV1.EPHEMERAL,
        )
    )

    assert result.error_code == "live_execution_failed"
    assert result.partial_failure is True
    assert result.indexed_evidence
    assert result.live_evidence == ()
    assert result.live_execution_status is HybridAskLiveExecutionStatusV1.PARTIAL


def test_new_execution_result_is_strict_and_timestamps_are_aware() -> None:
    with pytest.raises(ValidationError):
        LiveCapabilityResultItemV1(
            remote_item_id="item",
            safe_display_name="item",
            content="content",
            content_hash="hash",
            retrieved_at=datetime(2026, 8, 4, 10, 0),
        )

    assert isinstance(
        KnowledgeQueryExecutionResultV1(
            run_id="run",
            plan_id="plan",
            mode=QueryPolicyModeV2.INDEXED_ONLY,
            indexed_retrieval_status=HybridAskIndexedRetrievalStatusV1.SKIPPED,
            live_execution_status=HybridAskLiveExecutionStatusV1.SKIPPED,
            truncation_state=HybridAskTruncationStateV1.NONE,
            partial_failure=False,
            started_at=_NOW,
            completed_at=_NOW,
        ),
        KnowledgeQueryExecutionResultV1,
    )


def test_orchestrator_runs_optional_expansion_stage_through_shared_executor() -> None:
    indexed = _NeutralRetriever()
    handler = _NeutralHandler((_item("item-1"),))
    live_executor, _ = _executor(handler)
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed,
        live_executor=live_executor,
    )

    class _Expansion:
        def expand(self, **kwargs: object) -> tuple[Any, ...]:
            if kwargs["stage"] == 1:
                return (_call("call-2"),)
            return ()

    result = asyncio.run(
        orchestrator.execute(
            run_id="run-staged",
            question="question",
            validated_plan=_plan(
                QueryPolicyModeV2.LIVE_ONLY,
                calls=(_call("call-1", budget=_budget(max_live_calls=2)),),
                budget=_budget(max_live_calls=2),
            ),
            retention=LiveResultRetentionV1.EPHEMERAL,
            live_expansion=_Expansion(),  # type: ignore[arg-type]
        )
    )

    assert result.error_code is None
    assert len(handler.calls) == 2
    assert {item.call_id for item in result.live_evidence} == {
        "call-1",
        "call-2",
    }
