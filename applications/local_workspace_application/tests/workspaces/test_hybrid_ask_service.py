# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Workspace Ask V2 application acceptance proof."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.hybrid_ask_execution import (
    KnowledgeQueryOrchestratorV1,
    LiveCapabilityExecutionResultV1,
    LiveCapabilityResultItemV1,
    LiveExecutionOutcomeV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    AskAudienceV1,
    EvidenceAdmissibilityStatusV1,
    IndexedWorkspaceEvidenceV1,
    LiveExecutionReceiptV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    HybridAskPolicyError,
    IndexedEvidenceRequirementV1,
    KnowledgeQueryAudienceV1,
    LiveCallProposalV1,
    LiveEvidenceRequirementV1,
    ProviderEvidencePlanV1,
    ResolvedLiveResourceScopeV1,
    compose_evidence_obligations,
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
from local_workspace_application.workspaces.knowledge_live_access_service import (
    WorkspaceLiveAccessRuntimeAuthority,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceSource,
)

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live.contracts import evidence_id_for_call
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_TENANT = "tenant-neutral"
_WORKSPACE = "workspace-neutral"
_KIND = IntegrationCategory.WIKI_KNOWLEDGE
_CONTENT = "Live result body exists only during synthesis."
_LIVE_ID = evidence_id_for_call(
    provider_id="neutral_provider",
    integration_kind=_KIND,
    source_kind="issues",
    capability_id="vendor.neutral_provider.issues.read",
    contract_version="1",
    live_access_binding_id="binding-1",
    connection_ref="connection-1",
    remote_resource_id=None,
    call_id="call-1",
    remote_item_id="item-1",
)


class _RecordingLLM(LLMAdapter):
    provider = "fake"
    model = "fake"

    def __init__(
        self,
        used_ids: list[str],
        *,
        status: str = "completed",
    ) -> None:
        super().__init__()
        self.used_ids = used_ids
        self.status = status
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
                    "status": self.status,
                    "answer": (
                        "The answer is grounded in the selected evidence."
                        if self.status == "completed"
                        else None
                    ),
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
            allowed_capability_ids=("vendor.neutral_provider.issues.read",),
            derived_provider_id="neutral_provider",
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
                    ("vendor.neutral_provider.issues.read",)
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
                capability_id="vendor.neutral_provider.issues.read",
                provider_id="neutral_provider",
                integration_kind=_KIND,
                source_kind="issues",
                contract_version="1",
                effect=CapabilityEffectV1.READ,
                read_only=True,
                resource_scope_required=True,
                request_schema_ref="schema://vendor-knowledge/live/neutral_provider/issues/read/request/v1",
                result_schema_ref="schema://vendor-knowledge/live/neutral_provider/issues/read/result/v1",
                max_result_items=50,
                max_result_bytes=1_048_576,
            ),
        )


class _EnvelopeValidator:
    def validate_request_envelope(self, **kwargs: object) -> BaseModel:
        class _Request(BaseModel):
            model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

            item_key: str

        return _Request(item_key="ITEM-1")


class _ScopeValidator:
    def validate_resource_scope(
        self,
        *,
        binding: WorkspaceLiveAccessBinding,
        capability_id: str,
        validated_request: BaseModel,
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
                provider_id=call.provider_id,
                source_kind=call.source_kind,
                capability_id=call.capability_id,
                contract_version=call.contract_version,
                started_at=_NOW,
                completed_at=_NOW,
                item_count=1,
                byte_count=len(_CONTENT.encode()),
                result_hash=sha256(_CONTENT.encode()).hexdigest(),
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
    required_evidence_obligations: tuple[Any, ...] = (),
) -> WorkspaceAskCommandV2:
    del retention
    proposals = ()
    if mode in (QueryPolicyModeV2.LIVE_ONLY, QueryPolicyModeV2.HYBRID):
        from local_workspace_application.workspaces.hybrid_ask_policy import (
            LiveCallProposalV1,
        )

        proposals = (
            LiveCallProposalV1(
                call_id="call-1",
                live_access_binding_id="binding-1",
                capability_id="vendor.neutral_provider.issues.read",
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
        required_evidence_obligations=required_evidence_obligations,
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


def test_v2_insufficient_evidence_persists_and_reconstructs() -> None:
    service, _, _, repository = _service(
        QueryPolicyModeV2.INDEXED_ONLY,
        LiveResultRetentionV1.EPHEMERAL,
        _RecordingLLM([], status="insufficient_evidence"),
    )

    run = asyncio.run(service.ask(_command(QueryPolicyModeV2.INDEXED_ONLY)))

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert run.citations == []
    reloaded = repository.get_run_v2(tenant_id=_TENANT, run_id=run.run_id)
    assert reloaded is not None
    assert reloaded.status is AskRunStatus.INSUFFICIENT_EVIDENCE


def test_v2_admissibility_satisfied_invokes_assembler_and_persists_result() -> None:
    indexed_id = _indexed_evidence().evidence_id
    llm = _RecordingLLM([indexed_id, _LIVE_ID])
    service, _, _, repository = _service(
        QueryPolicyModeV2.HYBRID,
        LiveResultRetentionV1.EPHEMERAL,
        llm,
    )

    run = asyncio.run(service.ask(_command(QueryPolicyModeV2.HYBRID)))

    assert run.status is AskRunStatus.COMPLETED
    assert llm.calls == 1
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.SATISFIED
    )
    obligation_ids = {item.requirement_id for item in run.required_evidence_obligations}
    assert obligation_ids == {"product:hybrid:indexed"}
    assert {item.evidence_type.value for item in run.citations} == {"indexed", "live"}
    reloaded = repository.get_run_v2(tenant_id=_TENANT, run_id=run.run_id)
    assert reloaded is not None
    assert reloaded.evidence_admissibility == run.evidence_admissibility
    assert reloaded.required_evidence_obligations == run.required_evidence_obligations


def test_v2_missing_required_indexed_evidence_skips_assembler() -> None:
    llm = _RecordingLLM([_indexed_evidence().evidence_id])
    indexed = _IndexedRetriever(())
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
    configuration = _Configuration(QueryPolicyModeV2.INDEXED_ONLY, LiveResultRetentionV1.EPHEMERAL)
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
        run_id_factory=lambda: "run-adm-1",
        plan_id_factory=lambda: "plan-adm-1",
    )
    obligations = (
        IndexedEvidenceRequirementV1(
            requirement_id="req-indexed",
            semantic_role="Indexed grounding",
        ),
    )

    run = asyncio.run(
        service.ask(
            _command(
                QueryPolicyModeV2.INDEXED_ONLY,
                required_evidence_obligations=obligations,
            )
        )
    )

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert llm.calls == 0
    assert run.answer is None
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.UNSATISFIED
    )


def test_v2_missing_required_live_evidence_skips_assembler() -> None:
    class _EmptyLiveExecutor(_LiveExecutor):
        async def execute(self, **_: object) -> LiveCapabilityExecutionResultV1:
            self.calls += 1
            return LiveCapabilityExecutionResultV1(
                call_id="call-1",
                normalized_outcome=LiveExecutionOutcomeV1.COMPLETED,
                items=(),
                item_count=0,
                byte_count=0,
                started_at=_NOW,
                completed_at=_NOW,
                receipt=None,
            )

    llm = _RecordingLLM([_LIVE_ID])
    indexed = _IndexedRetriever(())
    live = _EmptyLiveExecutor()
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed,
        live_executor=live,  # type: ignore[arg-type]
        clock=lambda: _NOW,
        monotonic=lambda: 100.0,
    )
    repository = _Repository()
    store = InMemoryDocumentStore()
    ask_repository = WorkspaceAskRepository(store)
    configuration = _Configuration(QueryPolicyModeV2.LIVE_ONLY, LiveResultRetentionV1.EPHEMERAL)
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
        run_id_factory=lambda: "run-adm-2",
        plan_id_factory=lambda: "plan-adm-2",
    )
    obligations = (
        LiveEvidenceRequirementV1(
            requirement_id="req-live",
            semantic_role="Live grounding",
            call_id="call-1",
        ),
    )

    run = asyncio.run(
        service.ask(
            _command(
                QueryPolicyModeV2.LIVE_ONLY,
                required_evidence_obligations=obligations,
            )
        )
    )

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert llm.calls == 0
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.UNSATISFIED
    )
    assert _CONTENT not in json.dumps(run.model_dump(mode="json"))


def test_v2_hybrid_product_obligations_derived_via_planning_path() -> None:
    llm = _RecordingLLM([_indexed_evidence().evidence_id, _LIVE_ID])
    service, _, _, _ = _service(
        QueryPolicyModeV2.HYBRID,
        LiveResultRetentionV1.EPHEMERAL,
        llm,
    )

    run = asyncio.run(service.ask(_command(QueryPolicyModeV2.HYBRID)))

    assert {item.requirement_id for item in run.required_evidence_obligations} == {
        "product:hybrid:indexed",
    }
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.SATISFIED
    )


def test_v2_hybrid_caller_cannot_remove_product_obligations() -> None:
    llm = _RecordingLLM([_indexed_evidence().evidence_id, _LIVE_ID])
    service, _, _, _ = _service(
        QueryPolicyModeV2.HYBRID,
        LiveResultRetentionV1.EPHEMERAL,
        llm,
    )

    run = asyncio.run(
        service.ask(
            _command(
                QueryPolicyModeV2.HYBRID,
                required_evidence_obligations=(),
            )
        )
    )

    assert "product:hybrid:indexed" in {
        item.requirement_id for item in run.required_evidence_obligations
    }


def test_v2_additive_caller_obligations_strengthen_product_contract() -> None:
    llm = _RecordingLLM([_indexed_evidence().evidence_id, _LIVE_ID])
    service, _, _, _ = _service(
        QueryPolicyModeV2.HYBRID,
        LiveResultRetentionV1.EPHEMERAL,
        llm,
    )
    additional = (
        IndexedEvidenceRequirementV1(
            requirement_id="caller:extra-indexed",
            semantic_role="Caller extra indexed proof",
        ),
    )

    run = asyncio.run(
        service.ask(
            _command(
                QueryPolicyModeV2.HYBRID,
                required_evidence_obligations=additional,
            )
        )
    )

    obligation_ids = {item.requirement_id for item in run.required_evidence_obligations}
    assert "product:hybrid:indexed" in obligation_ids
    assert "caller:extra-indexed" in obligation_ids


def test_v2_hybrid_provider_required_call_optional_planned_call_admissibility() -> None:
  """Core COMM-5C1-R2 proof: planned != mandatory for live evidence."""
  required_live_id = evidence_id_for_call(
      provider_id="neutral_provider",
      integration_kind=_KIND,
      source_kind="issues",
      capability_id="vendor.neutral_provider.issues.read",
      contract_version="1",
      live_access_binding_id="binding-1",
      connection_ref="connection-1",
      remote_resource_id=None,
      call_id="call-required",
      remote_item_id="item-1",
  )
  optional_live_id = evidence_id_for_call(
      provider_id="neutral_provider",
      integration_kind=_KIND,
      source_kind="issues",
      capability_id="vendor.neutral_provider.issues.read",
      contract_version="1",
      live_access_binding_id="binding-1",
      connection_ref="connection-1",
      remote_resource_id=None,
      call_id="call-optional",
      remote_item_id="item-1",
  )

  class _ProviderStrategy:
      def build_plan(
          self,
          *,
          configuration: WorkspaceKnowledgeConfigurationV1,
          request: object,
      ) -> ProviderEvidencePlanV1:
          del configuration, request
          return ProviderEvidencePlanV1(
              ordered_live_call_proposals=(
                  LiveCallProposalV1(
                      call_id="call-required",
                      live_access_binding_id="binding-1",
                      capability_id="vendor.neutral_provider.issues.read",
                      typed_capability_request={"item_key": "ITEM-1"},
                  ),
                  LiveCallProposalV1(
                      call_id="call-optional",
                      live_access_binding_id="binding-1",
                      capability_id="vendor.neutral_provider.issues.read",
                      typed_capability_request={"item_key": "ITEM-2"},
                  ),
              ),
              required_evidence_obligations=(
                  LiveEvidenceRequirementV1(
                      requirement_id="provider:live:call-required",
                      semantic_role="Authoritative required live call",
                      call_id="call-required",
                  ),
              ),
          )

      def build_expansion(self, **_: object) -> None:
          return None

      def coverage(self, **_: object) -> None:
          return None

  class _SelectiveLiveExecutor(_LiveExecutor):
      async def execute(
          self,
          *,
          run_id: str,
          call: Any,
          retention: LiveResultRetentionV1,
          **_: object,
      ) -> LiveCapabilityExecutionResultV1:
          if call.call_id == "call-optional":
              self.calls += 1
              return LiveCapabilityExecutionResultV1(
                  call_id=call.call_id,
                  normalized_outcome=LiveExecutionOutcomeV1.COMPLETED,
                  items=(),
                  item_count=0,
                  byte_count=0,
                  started_at=_NOW,
                  completed_at=_NOW,
                  receipt=None,
              )
          return await super().execute(
              run_id=run_id,
              call=call,
              retention=retention,
          )

  def _hybrid_service(
      llm: _RecordingLLM,
      live_executor: _LiveExecutor,
  ) -> WorkspaceAskServiceV2:
      repository = _Repository()
      store = InMemoryDocumentStore()
      ask_repository = WorkspaceAskRepository(store)
      configuration = _Configuration(
          QueryPolicyModeV2.HYBRID,
          LiveResultRetentionV1.EPHEMERAL,
      )
      hybrid_policy = WorkspaceQueryPolicyV2(
          tenant_id=_TENANT,
          workspace_id=_WORKSPACE,
          mode=QueryPolicyModeV2.HYBRID,
          allowed_connection_refs=("connection-1",),
          allowed_capability_ids=("vendor.neutral_provider.issues.read",),
          max_live_calls=2,
          max_total_duration_ms=30_000,
          max_result_items=50,
          max_result_bytes=1_048_576,
          live_result_retention=LiveResultRetentionV1.EPHEMERAL,
          mutation_id="mutation-1",
          effective_revision=1,
          updated_at=_NOW,
      )
      configuration.value = configuration.value.model_copy(
          update={"query_policy": hybrid_policy}
      )
      return WorkspaceAskServiceV2(
          workspace_service=_WorkspaceAuthority(),  # type: ignore[arg-type]
          workspace_repository=repository,  # type: ignore[arg-type]
          ask_repository=ask_repository,
          configuration_service=configuration,  # type: ignore[arg-type]
          capability_catalog=_Catalog(),  # type: ignore[arg-type]
          request_envelope_validator=_EnvelopeValidator(),  # type: ignore[arg-type]
          resource_scope_validator=_ScopeValidator(),
          orchestrator=KnowledgeQueryOrchestratorV1(
              indexed_retriever=_IndexedRetriever((_indexed_evidence(),)),
              live_executor=live_executor,  # type: ignore[arg-type]
              clock=lambda: _NOW,
              monotonic=lambda: 100.0,
          ),
          llm_adapter=llm,
          clock=lambda: _NOW,
          run_id_factory=lambda: "run-r2",
          plan_id_factory=lambda: "plan-r2",
          provider_strategy=_ProviderStrategy(),
      )

  command = WorkspaceAskCommandV2(
      tenant_id=_TENANT,
      workspace_id=_WORKSPACE,
      question="Required vs optional planned live calls?",
      requested_mode=QueryPolicyModeV2.HYBRID,
      audience_context=AudienceContextV1(
          audience=KnowledgeQueryAudienceV1.PERSONAL
      ),
      provider_request=object(),
      request_id="request-r2",
  )

  satisfied_llm = _RecordingLLM(
      [_indexed_evidence().evidence_id, required_live_id]
  )
  satisfied_run = asyncio.run(
      _hybrid_service(satisfied_llm, _SelectiveLiveExecutor()).ask(command)
  )
  assert satisfied_run.status is AskRunStatus.COMPLETED
  assert satisfied_llm.calls == 1
  assert satisfied_run.evidence_admissibility is not None
  assert (
      satisfied_run.evidence_admissibility.overall_status
      is EvidenceAdmissibilityStatusV1.SATISFIED
  )
  obligation_ids = {
      item.requirement_id for item in satisfied_run.required_evidence_obligations
  }
  assert obligation_ids == {
      "product:hybrid:indexed",
      "provider:live:call-required",
  }
  assert {item.evidence_type.value for item in satisfied_run.citations} == {
      "indexed",
      "live",
  }

  class _OptionalOnlyLiveExecutor(_LiveExecutor):
      async def execute(
          self,
          *,
          run_id: str,
          call: Any,
          retention: LiveResultRetentionV1,
          **_: object,
      ) -> LiveCapabilityExecutionResultV1:
          if call.call_id == "call-required":
              self.calls += 1
              return LiveCapabilityExecutionResultV1(
                  call_id=call.call_id,
                  normalized_outcome=LiveExecutionOutcomeV1.COMPLETED,
                  items=(),
                  item_count=0,
                  byte_count=0,
                  started_at=_NOW,
                  completed_at=_NOW,
                  receipt=None,
              )
          return await super().execute(
              run_id=run_id,
              call=call,
              retention=retention,
          )

  unsatisfied_llm = _RecordingLLM(
      [_indexed_evidence().evidence_id, optional_live_id]
  )
  unsatisfied_run = asyncio.run(
      _hybrid_service(unsatisfied_llm, _OptionalOnlyLiveExecutor()).ask(command)
  )
  assert unsatisfied_run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
  assert unsatisfied_llm.calls == 0
  assert unsatisfied_run.evidence_admissibility is not None
  assert (
      unsatisfied_run.evidence_admissibility.overall_status
      is EvidenceAdmissibilityStatusV1.UNSATISFIED
  )


def test_v2_compose_rejects_conflicting_duplicate_requirement_id() -> None:
    with pytest.raises(HybridAskPolicyError) as exc:
        compose_evidence_obligations(
            authoritative=(
                IndexedEvidenceRequirementV1(
                    requirement_id="dup",
                    semantic_role="Authoritative",
                ),
            ),
            additional=(
                IndexedEvidenceRequirementV1(
                    requirement_id="dup",
                    semantic_role="Caller duplicate",
                ),
            ),
        )
    assert exc.value.error_code == "duplicate_requirement_id"


def test_v2_indexed_only_without_product_contract_remains_obligation_free() -> None:
    llm = _RecordingLLM([_indexed_evidence().evidence_id])
    service, _, _, _ = _service(
        QueryPolicyModeV2.INDEXED_ONLY,
        LiveResultRetentionV1.EPHEMERAL,
        llm,
    )

    run = asyncio.run(service.ask(_command(QueryPolicyModeV2.INDEXED_ONLY)))

    assert run.required_evidence_obligations == ()
    assert run.status is AskRunStatus.COMPLETED


def test_v2_provider_strategy_supplies_authoritative_obligations() -> None:
    provider_live_id = evidence_id_for_call(
        provider_id="neutral_provider",
        integration_kind=_KIND,
        source_kind="issues",
        capability_id="vendor.neutral_provider.issues.read",
        contract_version="1",
        live_access_binding_id="binding-1",
        connection_ref="connection-1",
        remote_resource_id=None,
        call_id="provider-call-1",
        remote_item_id="item-1",
    )
    class _ProviderStrategy:
        def build_plan(
            self,
            *,
            configuration: WorkspaceKnowledgeConfigurationV1,
            request: object,
        ) -> ProviderEvidencePlanV1:
            del configuration, request
            return ProviderEvidencePlanV1(
                ordered_live_call_proposals=(
                    LiveCallProposalV1(
                        call_id="provider-call-1",
                        live_access_binding_id="binding-1",
                        capability_id="vendor.neutral_provider.issues.read",
                        typed_capability_request={"item_key": "ITEM-1"},
                    ),
                ),
                required_evidence_obligations=(
                    LiveEvidenceRequirementV1(
                        requirement_id="provider:live:provider-call-1",
                        semantic_role="Provider live proof",
                        call_id="provider-call-1",
                    ),
                ),
            )

        def build_expansion(self, **_: object) -> None:
            return None

        def coverage(self, **_: object) -> None:
            return None

    llm = _RecordingLLM([provider_live_id])
    repository = _Repository()
    store = InMemoryDocumentStore()
    ask_repository = WorkspaceAskRepository(store)
    configuration = _Configuration(
        QueryPolicyModeV2.LIVE_ONLY,
        LiveResultRetentionV1.EPHEMERAL,
    )
    service = WorkspaceAskServiceV2(
        workspace_service=_WorkspaceAuthority(),  # type: ignore[arg-type]
        workspace_repository=repository,  # type: ignore[arg-type]
        ask_repository=ask_repository,
        configuration_service=configuration,  # type: ignore[arg-type]
        capability_catalog=_Catalog(),  # type: ignore[arg-type]
        request_envelope_validator=_EnvelopeValidator(),  # type: ignore[arg-type]
        resource_scope_validator=_ScopeValidator(),
        orchestrator=KnowledgeQueryOrchestratorV1(
            indexed_retriever=_IndexedRetriever(()),
            live_executor=_LiveExecutor(),  # type: ignore[arg-type]
            clock=lambda: _NOW,
            monotonic=lambda: 100.0,
        ),
        llm_adapter=llm,
        clock=lambda: _NOW,
        run_id_factory=lambda: "run-provider-1",
        plan_id_factory=lambda: "plan-provider-1",
        provider_strategy=_ProviderStrategy(),
    )
    command = WorkspaceAskCommandV2(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        question="Provider-owned obligations?",
        requested_mode=QueryPolicyModeV2.LIVE_ONLY,
        audience_context=AudienceContextV1(
            audience=KnowledgeQueryAudienceV1.PERSONAL
        ),
        provider_request=object(),
        request_id="request-provider-1",
    )

    run = asyncio.run(service.ask(command))

    assert run.required_evidence_obligations[0].requirement_id == "provider:live:provider-call-1"
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.SATISFIED
    )


def test_v2_hybrid_missing_authoritative_evidence_skips_assembler() -> None:
    llm = _RecordingLLM([_indexed_evidence().evidence_id, _LIVE_ID])
    indexed = _IndexedRetriever(())
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
    configuration = _Configuration(QueryPolicyModeV2.HYBRID, LiveResultRetentionV1.EPHEMERAL)
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
        run_id_factory=lambda: "run-adm-hybrid",
        plan_id_factory=lambda: "plan-adm-hybrid",
    )

    run = asyncio.run(service.ask(_command(QueryPolicyModeV2.HYBRID)))

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert llm.calls == 0
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.UNSATISFIED
    )


def test_v2_controlled_execution_failure_persists_and_reconstructs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _, _, repository = _service(
        QueryPolicyModeV2.LIVE_ONLY,
        LiveResultRetentionV1.EPHEMERAL,
        _RecordingLLM([_LIVE_ID]),
    )

    async def fail_execute(**_: Any) -> Any:
        raise RuntimeError("controlled execution failure")

    monkeypatch.setattr(service._orchestrator, "execute", fail_execute)
    run = asyncio.run(service.ask(_command(QueryPolicyModeV2.LIVE_ONLY)))

    assert run.status is AskRunStatus.FAILED
    assert run.error is not None
    assert run.error.code == "live_execution_failed"
    reloaded = repository.get_run_v2(tenant_id=_TENANT, run_id=run.run_id)
    assert reloaded is not None
    assert reloaded.error is not None
    assert reloaded.error.code == "live_execution_failed"


@pytest.mark.parametrize(
    "case",
    [
        "citation_missing_evidence",
        "citation_evidence_type_mismatch",
        "citation_live_call_id_mismatch",
        "citation_provider_mismatch",
        "citation_unrelated_receipt",
        "receipt_run_id_mismatch",
        "receipt_unknown_live_call",
        "duplicate_receipt_call_id",
        "receipt_only_requires_call_receipt",
        "ephemeral_retention_forbids_receipts",
        "duplicate_evidence_id",
        "live_call_identity_mismatch",
        "receipt_binding_mismatch",
    ],
)
def test_v2_finalization_revalidates_and_persists_only_safe_failure(
    case: str,
) -> None:
    receipt_cases = {
        "citation_unrelated_receipt",
        "receipt_run_id_mismatch",
        "receipt_unknown_live_call",
        "duplicate_receipt_call_id",
        "receipt_only_requires_call_receipt",
        "ephemeral_retention_forbids_receipts",
        "receipt_binding_mismatch",
    }
    retention = (
        LiveResultRetentionV1.RECEIPT_ONLY
        if case in receipt_cases
        else LiveResultRetentionV1.EPHEMERAL
    )
    service, _, _, repository = _service(
        QueryPolicyModeV2.LIVE_ONLY,
        retention,
        _RecordingLLM([_LIVE_ID]),
    )
    run = asyncio.run(
        service.ask(
            _command(
                QueryPolicyModeV2.LIVE_ONLY,
                retention=retention,
            )
        )
    )
    live_citation = run.citations[0]
    live_evidence = next(
        item for item in run.persisted_evidence if item.evidence_type.value == "live"
    )
    receipt = run.execution_receipts[0] if run.execution_receipts else None

    if case == "citation_missing_evidence":
        update = {"citations": [live_citation.model_copy(update={"evidence_id": "live:missing"})]}
    elif case == "citation_evidence_type_mismatch":
        update = {"citations": [live_citation.model_copy(update={"evidence_type": "indexed"})]}
    elif case == "citation_live_call_id_mismatch":
        update = {"citations": [live_citation.model_copy(update={"call_id": "call-other"})]}
    elif case == "citation_provider_mismatch":
        update = {"citations": [live_citation.model_copy(update={"provider_id": "other-provider"})]}
    elif case == "citation_unrelated_receipt":
        update = {"citations": [live_citation.model_copy(update={"receipt_id": "receipt-other"})]}
    elif case == "receipt_run_id_mismatch":
        assert receipt is not None
        update = {
            "execution_receipts": [
                receipt.model_copy(update={"run_id": "run-other"})
            ]
        }
    elif case == "receipt_unknown_live_call":
        assert receipt is not None
        update = {
            "execution_receipts": [
                receipt.model_copy(update={"call_id": "call-other"})
            ]
        }
    elif case == "duplicate_receipt_call_id":
        assert receipt is not None
        update = {
            "execution_receipts": [
                receipt,
                receipt.model_copy(update={"receipt_id": "receipt-duplicate"}),
            ]
        }
    elif case == "receipt_only_requires_call_receipt":
        update = {"execution_receipts": []}
    elif case == "ephemeral_retention_forbids_receipts":
        assert receipt is not None
        update = {
            "live_result_retention": LiveResultRetentionV1.EPHEMERAL,
            "execution_receipts": [receipt],
        }
    elif case == "duplicate_evidence_id":
        update = {
            "persisted_evidence": [live_evidence, live_evidence.model_copy()]
        }
    elif case == "live_call_identity_mismatch":
        update = {
            "persisted_evidence": [
                live_evidence,
                live_evidence.model_copy(
                    update={
                        "evidence_id": "live:call-1:conflicting",
                        "provider_id": "other-provider",
                    }
                ),
            ]
        }
    else:
        assert case == "receipt_binding_mismatch"
        assert receipt is not None
        update = {
            "execution_receipts": [
                receipt.model_copy(update={"live_access_binding_id": "other-binding"})
            ]
        }

    finalized = service._finalize_run_model(run, update=update)

    assert finalized.status is AskRunStatus.FAILED
    assert finalized.error is not None
    assert finalized.error.code == "citation_validation_failed"
    assert finalized.citations == []
    assert finalized.persisted_evidence == []
    assert finalized.execution_receipts == []
    persisted = repository.get_run_v2(tenant_id=_TENANT, run_id=run.run_id)
    assert persisted is not None
    assert persisted == finalized
    assert persisted.error is not None
    assert persisted.error.message == "Evidence citations could not be verified."


class _MutableConfiguration(_Configuration):
    def disable_binding(self) -> None:
        binding = self.value.live_access_bindings[0]
        disabled = binding.model_copy(
            update={"status": LiveAccessBindingStatusV1.DISABLED}
        )
        self.value = self.value.model_copy(
            update={"live_access_bindings": (disabled,)}
        )


class _TenantConnectionPort:
    def get_connection(self, *, tenant_id: str, connection_ref: str) -> object:
        from intergrax.runtime.vendor_knowledge.tenant_connections import (
            SafeTenantConnectionV1,
            TenantConnectionAdministrativeStatus,
        )

        return SafeTenantConnectionV1(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id="neutral_provider",
            integration_kind=_KIND,
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            safe_display_name="Neutral Connection",
            configuration_version=1,
            connected_principal_ref="principal-1",
            created_at=_NOW,
            updated_at=_NOW,
        )


class _RevokingIndexedRetriever:
    def __init__(self, configuration: _MutableConfiguration) -> None:
        self.configuration = configuration
        self.calls = 0

    async def retrieve(self, **_: object) -> tuple[IndexedWorkspaceEvidenceV1, ...]:
        self.calls += 1
        self.configuration.disable_binding()
        return ()


class _EnvelopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    item_key: str


class _RecordingLiveHandler:
    provider_id = "neutral_provider"
    integration_kind = _KIND
    source_kind = "issues"
    capability_id = "vendor.neutral_provider.issues.read"
    contract_version = "1"
    request_schema_ref = "schema://vendor-knowledge/live/neutral_provider/issues/read/request/v1"
    result_schema_ref = "schema://vendor-knowledge/live/neutral_provider/issues/read/result/v1"
    expected_request_model = BaseModel

    def __init__(
        self,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.failure = failure
        self.calls = 0

    async def execute(self, *, integration: object, call: object, context: object):
        del integration, context
        self.calls += 1
        if self.failure is not None:
            raise self.failure
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
            receipt=None,
        )


class _RecordingIntegrationResolver:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, **_: object) -> object:
        self.calls += 1
        return object()


def _runtime_authority_service(
    *,
    indexed_retriever: object,
    handler: _RecordingLiveHandler,
    configuration: _MutableConfiguration,
    llm: _RecordingLLM,
) -> tuple[
    WorkspaceAskServiceV2,
    _RecordingLiveHandler,
    _RecordingIntegrationResolver,
    WorkspaceLiveAccessRuntimeAuthority,
]:
    from local_workspace_application.workspaces.hybrid_ask_execution import (
        LiveCapabilityExecutorV1,
        LiveCapabilityHandlerRegistryV1,
    )
    resolver = _RecordingIntegrationResolver()
    authority = WorkspaceLiveAccessRuntimeAuthority(
        configuration_service=configuration,  # type: ignore[arg-type]
        tenant_connection_port=_TenantConnectionPort(),  # type: ignore[arg-type]
        capability_catalog=_Catalog(),  # type: ignore[arg-type]
    )
    live_executor = LiveCapabilityExecutorV1(
        handler_registry=LiveCapabilityHandlerRegistryV1((handler,)),
        integration_resolver=resolver,
        runtime_authority=authority,
        monotonic=lambda: 100.0,
    )
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed_retriever,  # type: ignore[arg-type]
        live_executor=live_executor,
        clock=lambda: _NOW,
        monotonic=lambda: 100.0,
    )
    repository = _Repository()
    store = InMemoryDocumentStore()
    ask_repository = WorkspaceAskRepository(store)
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
        run_id_factory=lambda: "run-c2",
        plan_id_factory=lambda: "plan-c2",
    )
    return service, handler, resolver, authority


def test_v2_mid_flight_binding_revoke_insufficient_evidence_without_provider() -> None:
    llm = _RecordingLLM([_LIVE_ID])
    configuration = _MutableConfiguration(
        QueryPolicyModeV2.HYBRID,
        LiveResultRetentionV1.EPHEMERAL,
    )
    handler = _RecordingLiveHandler()
    service, handler, resolver, authority = _runtime_authority_service(
        indexed_retriever=_RevokingIndexedRetriever(configuration),
        handler=handler,
        configuration=configuration,
        llm=llm,
    )
    command = _command(
        QueryPolicyModeV2.HYBRID,
        required_evidence_obligations=(
            LiveEvidenceRequirementV1(
                requirement_id="required-live-call-1",
                semantic_role="Required live proof",
                call_id="call-1",
            ),
        ),
    )

    run = asyncio.run(service.ask(command))

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert run.answer is None
    assert llm.calls == 0
    assert handler.calls == 0
    assert resolver.calls == 0
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.UNSATISFIED
    )
    assert configuration.value.live_access_bindings[0].status is (
        LiveAccessBindingStatusV1.DISABLED
    )
    assert authority.is_usable(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        live_access_binding_id="binding-1",
        connection_ref="connection-1",
        capability_id="vendor.neutral_provider.issues.read",
    ) is False


def test_v2_mid_flight_binding_active_allows_provider_and_satisfies_admissibility() -> None:
    llm = _RecordingLLM([_LIVE_ID])
    configuration = _MutableConfiguration(
        QueryPolicyModeV2.LIVE_ONLY,
        LiveResultRetentionV1.EPHEMERAL,
    )
    handler = _RecordingLiveHandler()
    service, handler, resolver, authority = _runtime_authority_service(
        indexed_retriever=_IndexedRetriever(()),
        handler=handler,
        configuration=configuration,
        llm=llm,
    )
    command = _command(
        QueryPolicyModeV2.LIVE_ONLY,
        required_evidence_obligations=(
            LiveEvidenceRequirementV1(
                requirement_id="required-live-call-1",
                semantic_role="Required live proof",
                call_id="call-1",
            ),
        ),
    )

    run = asyncio.run(service.ask(command))

    assert handler.calls == 1
    assert resolver.calls == 1
    assert authority.is_usable(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        live_access_binding_id="binding-1",
        connection_ref="connection-1",
        capability_id="vendor.neutral_provider.issues.read",
    ) is True
    assert run.status is AskRunStatus.COMPLETED
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.SATISFIED
    )
    assert llm.calls == 1


def test_v2_governance_binding_unavailable_maps_to_insufficient_evidence() -> None:
    class _UnavailableBindingExecutor:
        async def execute(self, **kwargs: object) -> LiveCapabilityExecutionResultV1:
            call = kwargs["call"]
            return LiveCapabilityExecutionResultV1(
                call_id=call.call_id,
                normalized_outcome=LiveExecutionOutcomeV1.FAILED,
                item_count=0,
                byte_count=0,
                started_at=_NOW,
                completed_at=_NOW,
                error_code="live_binding_unavailable",
                receipt=None,
            )

    llm = _RecordingLLM([_LIVE_ID])
    indexed = _IndexedRetriever(())
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed,
        live_executor=_UnavailableBindingExecutor(),  # type: ignore[arg-type]
        clock=lambda: _NOW,
        monotonic=lambda: 100.0,
    )
    repository = _Repository()
    store = InMemoryDocumentStore()
    ask_repository = WorkspaceAskRepository(store)
    configuration = _Configuration(
        QueryPolicyModeV2.LIVE_ONLY,
        LiveResultRetentionV1.EPHEMERAL,
    )
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
        run_id_factory=lambda: "run-gov",
        plan_id_factory=lambda: "plan-gov",
    )
    command = _command(
        QueryPolicyModeV2.LIVE_ONLY,
        required_evidence_obligations=(
            LiveEvidenceRequirementV1(
                requirement_id="required-live-call-1",
                semantic_role="Required live proof",
                call_id="call-1",
            ),
        ),
    )

    run = asyncio.run(service.ask(command))

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert run.answer is None
    assert llm.calls == 0
    assert run.error is None


def test_v2_technical_provider_failure_remains_failed_with_runtime_authority() -> None:
    llm = _RecordingLLM([_LIVE_ID])
    configuration = _MutableConfiguration(
        QueryPolicyModeV2.LIVE_ONLY,
        LiveResultRetentionV1.EPHEMERAL,
    )
    handler = _RecordingLiveHandler(failure=RuntimeError("provider transport failed"))
    service, handler, resolver, _ = _runtime_authority_service(
        indexed_retriever=_IndexedRetriever(()),
        handler=handler,
        configuration=configuration,
        llm=llm,
    )

    run = asyncio.run(service.ask(_command(QueryPolicyModeV2.LIVE_ONLY)))

    assert handler.calls == 1
    assert resolver.calls == 1
    assert run.status is AskRunStatus.FAILED
    assert run.error is not None
    assert run.error.code == "live_execution_failed"
    assert llm.calls == 0

