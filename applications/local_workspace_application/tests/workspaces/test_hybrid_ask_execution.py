# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Hybrid Ask execution tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from local_workspace_application.workspaces.hybrid_ask_execution import (
    KnowledgeQueryExecutionResultV1,
    KnowledgeQueryOrchestratorV1,
    LiveCapabilityExecutionResultV1,
    LiveCapabilityExecutorV1,
    LiveCapabilityHandlerRegistryV1,
    LiveCapabilityResultItemV1,
    LiveExecutionOutcomeV1,
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
    LiveResultRetentionV1,
    QueryPolicyModeV2,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_TENANT = "tenant-1"
_WORKSPACE = "workspace-1"
_KIND = IntegrationCategory.WIKI_KNOWLEDGE


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
    capability_id: str = "cap.read",
) -> Any:
    from local_workspace_application.workspaces.hybrid_ask_policy import (
        ExecutableLiveCallV1,
    )

    return ExecutableLiveCallV1(
        call_id=call_id,
        live_access_binding_id="binding-1",
        connection_ref="connection-1",
        provider_id="neutral-provider",
        integration_kind=_KIND,
        capability_id=capability_id,
        validated_request={"query": "validated"},
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
                audience=KnowledgeQueryAudienceV1.PERSONAL
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
    provider_id = "neutral-provider"
    integration_kind = _KIND.value


class _RecordingResolver:
    def __init__(self, integration: object | None = None) -> None:
        self.integration = integration or _NeutralIntegration()
        self.calls: list[dict[str, object]] = []

    def resolve(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.integration


class _NeutralHandler:
    provider_id = "neutral-provider"
    integration_kind = _KIND
    capability_id = "cap.read"
    contract_version = "1"

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
        return _live_result(getattr(call, "call_id"), self.items)


class _NeutralRetriever:
    def __init__(self, evidence: tuple[IndexedWorkspaceEvidenceV1, ...] = ()) -> None:
        self.evidence = evidence
        self.calls: list[dict[str, object]] = []

    async def retrieve(self, **kwargs: object):
        self.calls.append(kwargs)
        return self.evidence


def _indexed_evidence() -> IndexedWorkspaceEvidenceV1:
    return IndexedWorkspaceEvidenceV1(
        evidence_id="idx:workspace-1:document-1:chunk-1",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        safe_display_name="document.txt",
        retrieved_at=_NOW,
        content="indexed content",
        content_hash="a" * 64,
        audience=AskAudienceV1.PERSONAL,
        source_id="source-1",
        document_id="document-1",
        chunk_id="chunk-1",
        score=0.9,
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
        provider_id = "other-provider"

    registry = LiveCapabilityHandlerRegistryV1((first, _OtherProvider()))
    assert registry.resolve(
        provider_id="neutral-provider",
        integration_kind=_KIND,
        capability_id="cap.read",
    ) is first
    with pytest.raises(LookupError, match="live_capability_unavailable"):
        registry.resolve(
            provider_id="missing-provider",
            integration_kind=_KIND,
            capability_id="cap.read",
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
            "provider_id": "neutral-provider",
            "integration_kind": _KIND,
        }
    ]
    assert handler.calls[0][0] is resolver.integration
    assert getattr(handler.calls[0][1], "validated_request") == {"query": "validated"}
    assert (
        getattr(handler.calls[0][1], "resolved_resource_scope").scope_token
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
    assert result.byte_count == len("live content".encode()) * 2
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
        provider_id = "neutral-provider"
        integration_kind = _KIND
        capability_id = "cap.read"
        contract_version = "1"

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
