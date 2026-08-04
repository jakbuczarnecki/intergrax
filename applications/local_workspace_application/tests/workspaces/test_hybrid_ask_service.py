# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Workspace Ask V2 application acceptance proof."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Sequence

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.hybrid_ask_execution import (
    KnowledgeQueryOrchestratorV1,
    LiveCapabilityExecutionResultV1,
    LiveExecutionOutcomeV1,
    LiveCapabilityResultItemV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    AskAudienceV1,
    IndexedWorkspaceEvidenceV1,
    LiveExecutionReceiptV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    KnowledgeQueryAudienceV1,
    ResolvedLiveResourceScopeV1,
)
from local_workspace_application.workspaces.hybrid_ask_service import (
    WorkspaceAskCommandV2,
    WorkspaceAskServiceV2,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    LiveResultRetentionV1,
    QueryPolicyModeV2,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicyV2,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceSource,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_TENANT = "tenant-neutral"
_WORKSPACE = "workspace-neutral"
_KIND = IntegrationCategory.WIKI_KNOWLEDGE
_CONTENT = "Live result body exists only during synthesis."
_LIVE_ID = f"live:call-1:{sha256(b'item-1').hexdigest()}"


class _RecordingLLM(LLMAdapter):
    provider = "fake"
    model = "fake"

    def __init__(self, used_ids: list[str]) -> None:
        super().__init__()
        self.used_ids = used_ids
        self.calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        del messages, temperature, max_tokens, run_id
        self.calls += 1
        return build_adapter_response(
            content=json.dumps(
                {
                    "status": "completed",
                    "answer": "The answer is grounded in the selected evidence.",
                    "used_evidence_ids": self.used_ids,
                }
            )
        )


class _WorkspaceAuthority:
    def get_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        if tenant_id == _TENANT and workspace_id == _WORKSPACE:
            return Workspace(
                workspace_id=_WORKSPACE,
                tenant_id=_TENANT,
                name="Neutral Workspace",
                created_at=_NOW,
                updated_at=_NOW,
            )
        return None


class _Repository:
    def __init__(self) -> None:
        self.document = WorkspaceDocumentReference(
            document_id="document-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id="source-1",
            source_path="docs/document.txt",
            file_name="document.txt",
            content_hash="sha256:" + "a" * 64,
            indexed_at=_NOW,
        )
        self.source = WorkspaceSource(
            source_id="source-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            path="docs",
            created_at=_NOW,
        )

    def get_document_ref(self, **_: object) -> WorkspaceDocumentReference:
        return self.document

    def get_source(self, **_: object) -> WorkspaceSource:
        return self.source


class _Configuration:
    def __init__(self, mode: QueryPolicyModeV2, retention: LiveResultRetentionV1):
        attachment = WorkspaceConnectionAttachment(
            attachment_id="attachment-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref="connection-1",
            safe_display_label="Neutral Connection",
            status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
            mutation_id="mutation-1",
            effective_revision=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
        binding = WorkspaceLiveAccessBinding(
            live_access_binding_id="binding-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref="connection-1",
            allowed_capability_ids=("cap.read",),
            derived_provider_id="neutral-provider",
            derived_integration_kind=_KIND,
            derived_safe_display_label="Neutral Connection",
            status=LiveAccessBindingStatusV1.ACTIVE,
            mutation_id="mutation-1",
            effective_revision=1,
            semantic_identity_hash="a" * 64,
            created_at=_NOW,
            updated_at=_NOW,
        )
        self.value = WorkspaceKnowledgeConfigurationV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=1,
            connection_attachments=(attachment,),
            indexed_sources=(),
            live_access_bindings=(binding,),
            query_policy=WorkspaceQueryPolicyV2(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                mode=mode,
                allowed_connection_refs=(
                    ("connection-1",)
                    if mode is not QueryPolicyModeV2.INDEXED_ONLY
                    else ()
                ),
                allowed_capability_ids=(
                    ("cap.read",)
                    if mode is not QueryPolicyModeV2.INDEXED_ONLY
                    else ()
                ),
                max_live_calls=(
                    1 if mode is not QueryPolicyModeV2.INDEXED_ONLY else 0
                ),
                max_total_duration_ms=30_000,
                max_result_items=50,
                max_result_bytes=1_048_576,
                live_result_retention=retention,
                mutation_id="mutation-1",
                effective_revision=1,
                updated_at=_NOW,
            ),
            updated_at=_NOW,
        )

    def get_configuration(self, **_: object) -> WorkspaceKnowledgeConfigurationV1:
        return self.value


class _Catalog:
    def list_capabilities(self, **_: object) -> tuple[Any, ...]:
        from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
            CapabilityEffectV1,
            LiveCapabilityDescriptorV1,
        )

        return (
            LiveCapabilityDescriptorV1(
                capability_id="cap.read",
                provider_id="neutral-provider",
                integration_kind=_KIND,
                effect=CapabilityEffectV1.READ,
                read_only=True,
                resource_scope_required=True,
                request_schema_ref="neutral.request.v1",
                result_schema_ref="neutral.result.v1",
                max_result_items=50,
                max_result_bytes=1_048_576,
            ),
        )


class _EnvelopeValidator:
    def validate_request_envelope(self, **kwargs: object) -> dict[str, object]:
        return {"item_key": "ITEM-1"}


class _ScopeValidator:
    def validate_resource_scope(
        self,
        *,
        binding: WorkspaceLiveAccessBinding,
        capability_id: str,
        validated_request: dict[str, object],
    ) -> ResolvedLiveResourceScopeV1:
        del capability_id, validated_request
        return ResolvedLiveResourceScopeV1(
            remote_resource_id=binding.remote_resource_id,
            scope_token="neutral-scope",
        )


class _IndexedRetriever:
    def __init__(self, evidence: tuple[IndexedWorkspaceEvidenceV1, ...]) -> None:
        self.evidence = evidence
        self.calls = 0

    async def retrieve(self, **_: object) -> tuple[IndexedWorkspaceEvidenceV1, ...]:
        self.calls += 1
        return self.evidence


class _LiveExecutor:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, *, run_id: str, call: Any, retention: LiveResultRetentionV1, **_: object):
        self.calls += 1
        receipt = None
        if retention is LiveResultRetentionV1.RECEIPT_ONLY:
            receipt = LiveExecutionReceiptV1(
                receipt_id="receipt-1",
                run_id=run_id,
                call_id=call.call_id,
                live_access_binding_id=call.live_access_binding_id,
                capability_id=call.capability_id,
                started_at=_NOW,
                completed_at=_NOW,
                item_count=1,
                byte_count=len(_CONTENT.encode()),
                content_hash=sha256(_CONTENT.encode()).hexdigest(),
                normalized_outcome="completed",
            )
        item = LiveCapabilityResultItemV1(
            remote_item_id="item-1",
            safe_display_name="Neutral live item",
            content=_CONTENT,
            content_hash=sha256(_CONTENT.encode()).hexdigest(),
            retrieved_at=_NOW,
            remote_updated_at=_NOW,
        )
        return LiveCapabilityExecutionResultV1(
            call_id=call.call_id,
            normalized_outcome=LiveExecutionOutcomeV1.COMPLETED,
            items=(item,),
            item_count=1,
            byte_count=len(_CONTENT.encode()),
            started_at=_NOW,
            completed_at=_NOW,
            receipt=receipt,
        )


def _indexed_evidence() -> IndexedWorkspaceEvidenceV1:
    content = "Indexed content."
    return IndexedWorkspaceEvidenceV1(
        evidence_id="idx:workspace-neutral:document-1:chunk-1",
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
    )


def _service(
    mode: QueryPolicyModeV2,
    retention: LiveResultRetentionV1,
    llm: _RecordingLLM,
) -> tuple[WorkspaceAskServiceV2, _IndexedRetriever, _LiveExecutor, WorkspaceAskRepository]:
    indexed = _IndexedRetriever((_indexed_evidence(),))
    live = _LiveExecutor()
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed,
        live_executor=live,  # type: ignore[arg-type]
        clock=lambda: _NOW,
        monotonic=lambda: 100.0,
    )
    repository = _Repository()
    store = InMemoryDocumentStore()
    ask_repository = WorkspaceAskRepository(store)
    configuration = _Configuration(mode, retention)
    service = WorkspaceAskServiceV2(
        workspace_service=_WorkspaceAuthority(),  # type: ignore[arg-type]
        workspace_repository=repository,  # type: ignore[arg-type]
        ask_repository=ask_repository,
        configuration_service=configuration,  # type: ignore[arg-type]
        capability_catalog=_Catalog(),  # type: ignore[arg-type]
        request_envelope_validator=_EnvelopeValidator(),  # type: ignore[arg-type]
        resource_scope_validator=_ScopeValidator(),
        orchestrator=orchestrator,
        llm_adapter=llm,
        clock=lambda: _NOW,
        run_id_factory=lambda: "run-1",
        plan_id_factory=lambda: "plan-1",
    )
    return service, indexed, live, ask_repository


def _command(
    mode: QueryPolicyModeV2,
    *,
    retention: LiveResultRetentionV1 = LiveResultRetentionV1.EPHEMERAL,
) -> WorkspaceAskCommandV2:
    del retention
    proposals = ()
    if mode in (QueryPolicyModeV2.LIVE_ONLY, QueryPolicyModeV2.HYBRID):
        from local_workspace_application.workspaces.hybrid_ask_policy import LiveCallProposalV1

        proposals = (
            LiveCallProposalV1(
                call_id="call-1",
                live_access_binding_id="binding-1",
                capability_id="cap.read",
                typed_capability_request={"item_key": "ITEM-1"},
            ),
        )
    return WorkspaceAskCommandV2(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        question="What is the grounded answer?",
        requested_mode=mode,
        audience_context=AudienceContextV1(
            audience=KnowledgeQueryAudienceV1.PERSONAL
        ),
        ordered_live_call_proposals=proposals,
        request_id="request-1",
    )


@pytest.mark.parametrize(
    ("mode", "expected_indexed", "expected_live"),
    [
        (QueryPolicyModeV2.INDEXED_ONLY, 1, 0),
        (QueryPolicyModeV2.LIVE_ONLY, 0, 1),
        (QueryPolicyModeV2.HYBRID, 1, 1),
    ],
)
def test_v2_modes_execute_once_and_reconstruct(
    mode: QueryPolicyModeV2,
    expected_indexed: int,
    expected_live: int,
) -> None:
    indexed_id = _indexed_evidence().evidence_id
    used_ids = (
        [indexed_id]
        if mode is QueryPolicyModeV2.INDEXED_ONLY
        else [_LIVE_ID]
        if mode is QueryPolicyModeV2.LIVE_ONLY
        else [indexed_id, _LIVE_ID]
    )
    llm = _RecordingLLM(used_ids)
    service, indexed, live, repository = _service(
        mode,
        LiveResultRetentionV1.EPHEMERAL,
        llm,
    )

    run = asyncio.run(service.ask(_command(mode)))

    assert run.status is AskRunStatus.COMPLETED
    assert indexed.calls == expected_indexed
    assert live.calls == expected_live
    assert llm.calls == 1
    assert run.citations
    if mode is QueryPolicyModeV2.HYBRID:
        assert {item.evidence_type.value for item in run.citations} == {"indexed", "live"}
    if mode is QueryPolicyModeV2.LIVE_ONLY:
        assert all(item.evidence_type.value == "live" for item in run.citations)
    assert all(item.evidence_type.value != "live" or "content" not in item.model_dump() for item in run.persisted_evidence)
    assert _CONTENT not in json.dumps(run.model_dump(mode="json"))

    reloaded = repository.get_run_v2(tenant_id=_TENANT, run_id=run.run_id)
    assert reloaded is not None
    assert reloaded.run_id == run.run_id
    assert reloaded.workspace_id == _WORKSPACE


def test_v2_receipt_only_persists_one_receipt_without_live_body() -> None:
    llm = _RecordingLLM([_LIVE_ID])
    service, _, live, repository = _service(
        QueryPolicyModeV2.LIVE_ONLY,
        LiveResultRetentionV1.RECEIPT_ONLY,
        llm,
    )

    run = asyncio.run(
        service.ask(
            _command(
                QueryPolicyModeV2.LIVE_ONLY,
                retention=LiveResultRetentionV1.RECEIPT_ONLY,
            )
        )
    )

    assert run.status is AskRunStatus.COMPLETED
    assert live.calls == 1
    assert len(run.execution_receipts) == 1
    assert run.citations[0].receipt_id == "receipt-1"
    assert _CONTENT not in json.dumps(repository.get_run_v2(
        tenant_id=_TENANT,
        run_id=run.run_id,
    ).model_dump(mode="json"))


def test_v2_unknown_model_evidence_id_fails_closed_and_persists() -> None:
    llm = _RecordingLLM(["live:unknown"])
    service, _, _, repository = _service(
        QueryPolicyModeV2.LIVE_ONLY,
        LiveResultRetentionV1.EPHEMERAL,
        llm,
    )

    run = asyncio.run(
        service.ask(_command(QueryPolicyModeV2.LIVE_ONLY))
    )

    assert run.status is AskRunStatus.FAILED
    assert run.error is not None
    assert run.error.code == "unknown_evidence_id"
    assert repository.get_run_v2(tenant_id=_TENANT, run_id=run.run_id) is not None

