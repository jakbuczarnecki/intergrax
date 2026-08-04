# © Artur Czarnecki. All rights reserved.

"""Hybrid Ask core contract, Query Policy V2 and Evidence Plan validation tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.ask_repository import (
    AskRunSchemaVersion,
    WorkspaceAskRepository,
    WorkspaceAskRepositoryError,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    AskAudienceV1,
    HybridAskIndexedRetrievalStatusV1,
    HybridAskLiveExecutionStatusV1,
    IndexedWorkspaceCitationV1,
    LiveExecutionReceiptV1,
    LiveWorkspaceCitationV1,
    PersistedIndexedEvidenceV2,
    PersistedLiveEvidenceProvenanceV2,
    WorkspaceAskRunV2,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    EffectiveLiveCallBudgetV1,
    EvidencePlanV1,
    HybridAskPolicyError,
    IndexedRetrievalDirectiveV1,
    KnowledgeQueryAudienceV1,
    LiveCallProposalV1,
    ResolvedLiveResourceScopeV1,
    resolve_effective_query_policy,
    validate_evidence_plan,
)
from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    normalize_update_query_policy_request_hash,
    query_policy_stage_manifest_hash,
    semantic_identity_hash_for_query_policy,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    LiveResultRetentionV1,
    QueryPolicyModeV1,
    QueryPolicyModeV2,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceKnowledgeConfigurationHead,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicy,
    WorkspaceQueryPolicyV2,
    parse_workspace_query_policy,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
    WorkspaceKnowledgeMutationRecoveryDispositionV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_configuration_validation import (
    validate_configuration_idempotency_hash,
)
from local_workspace_application.workspaces.knowledge_query_policy_handlers import (
    UpdateQueryPolicyMutationHandler,
    UpdateQueryPolicyMutationIntent,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_MUTATION = "mutation-1"
_SHA256 = "a" * 64
_IDEMPOTENCY = "b" * 64
_POLICY_HANDLER = UpdateQueryPolicyMutationHandler()


def _v1_policy(**overrides: object) -> WorkspaceQueryPolicy:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "mode": QueryPolicyModeV1.INDEXED_ONLY,
        "allowed_connection_refs": (),
        "allowed_capability_ids": (),
        "max_live_calls": 0,
        "max_total_duration_ms": 30_000,
        "max_result_items": 50,
        "max_result_bytes": 1_048_576,
        "live_result_retention": LiveResultRetentionV1.EPHEMERAL,
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceQueryPolicy(**payload)


def _v2_policy(**overrides: object) -> WorkspaceQueryPolicyV2:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "mode": QueryPolicyModeV2.HYBRID,
        "allowed_connection_refs": ("conn.live",),
        "allowed_capability_ids": ("cap.read",),
        "max_live_calls": 2,
        "max_total_duration_ms": 30_000,
        "max_result_items": 50,
        "max_result_bytes": 1_048_576,
        "live_result_retention": LiveResultRetentionV1.EPHEMERAL,
        "mutation_id": _MUTATION,
        "effective_revision": 2,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceQueryPolicyV2(**payload)


def _attachment(**overrides: object) -> WorkspaceConnectionAttachment:
    payload = {
        "attachment_id": "att-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": "conn.live",
        "safe_display_label": "Live Connection",
        "status": WorkspaceConnectionAttachmentStatusV1.ATTACHED,
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceConnectionAttachment(**payload)


def _live_binding(**overrides: object) -> WorkspaceLiveAccessBinding:
    payload = {
        "live_access_binding_id": "live-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": "conn.live",
        "allowed_capability_ids": ("cap.read",),
        "derived_provider_id": "provider-neutral",
        "derived_integration_kind": IntegrationCategory.WIKI_KNOWLEDGE,
        "derived_safe_display_label": "Neutral Provider",
        "status": LiveAccessBindingStatusV1.ACTIVE,
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "semantic_identity_hash": _SHA256,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceLiveAccessBinding(**payload)


def _configuration(
    *,
    revision: int = 1,
    query_policy: WorkspaceQueryPolicy | WorkspaceQueryPolicyV2 | None = None,
) -> WorkspaceKnowledgeConfigurationV1:
    return WorkspaceKnowledgeConfigurationV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=revision,
        connection_attachments=(_attachment(),),
        indexed_sources=(),
        live_access_bindings=(_live_binding(),),
        query_policy=query_policy,
        updated_at=_NOW,
    )


def _descriptor(**overrides: object) -> LiveCapabilityDescriptorV1:
    payload = {
        "capability_id": "cap.read",
        "provider_id": "provider-neutral",
        "integration_kind": IntegrationCategory.WIKI_KNOWLEDGE,
        "effect": CapabilityEffectV1.READ,
        "read_only": True,
        "resource_scope_required": True,
        "request_schema_ref": "test.request.v1",
        "result_schema_ref": "test.result.v1",
        "max_result_items": 25,
        "max_result_bytes": 512_000,
    }
    payload.update(overrides)
    return LiveCapabilityDescriptorV1(**payload)


class _FakeCatalog:
    def __init__(
        self,
        descriptors: dict[str, LiveCapabilityDescriptorV1 | tuple[LiveCapabilityDescriptorV1, ...]],
    ) -> None:
        self._descriptors = descriptors

    def list_capabilities(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        remote_resource_id: str | None,
    ) -> tuple[LiveCapabilityDescriptorV1, ...]:
        del tenant_id, remote_resource_id
        value = self._descriptors.get(connection_ref, ())
        if isinstance(value, LiveCapabilityDescriptorV1):
            return (value,)
        return value


class _FakeEnvelopeValidator:
    def validate_request_envelope(
        self,
        *,
        descriptor: LiveCapabilityDescriptorV1,
        typed_request: dict[str, Any],
    ) -> dict[str, Any]:
        if "item_key" not in typed_request:
            raise HybridAskPolicyError("live_request_invalid")
        return {"item_key": str(typed_request["item_key"]).strip()}


class _FakeScopeValidator:
    def validate_resource_scope(
        self,
        *,
        binding: WorkspaceLiveAccessBinding,
        capability_id: str,
        validated_request: dict[str, Any],
    ) -> ResolvedLiveResourceScopeV1:
        return ResolvedLiveResourceScopeV1(
            remote_resource_id=binding.remote_resource_id,
            scope_token=f"{binding.live_access_binding_id}:{validated_request['item_key']}",
        )


def _budget() -> EffectiveLiveCallBudgetV1:
    return EffectiveLiveCallBudgetV1(
        max_live_calls=2,
        max_total_duration_ms=30_000,
        max_result_items=50,
        max_result_bytes=1_048_576,
    )


def _indexed_only_plan() -> EvidencePlanV1:
    return EvidencePlanV1(
        plan_id="plan-1",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=1,
        mode=QueryPolicyModeV2.INDEXED_ONLY,
        indexed_retrieval_directive=IndexedRetrievalDirectiveV1(max_results=10),
        ordered_live_call_proposals=(),
        budget_snapshot=_budget(),
        audience_context=AudienceContextV1(audience=KnowledgeQueryAudienceV1.PERSONAL),
    )


def _live_plan() -> EvidencePlanV1:
    return EvidencePlanV1(
        plan_id="plan-2",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=1,
        mode=QueryPolicyModeV2.LIVE_ONLY,
        indexed_retrieval_directive=None,
        ordered_live_call_proposals=(
            LiveCallProposalV1(
                call_id="call-1",
                live_access_binding_id="live-1",
                capability_id="cap.read",
                typed_capability_request={"item_key": "ITEM-1"},
            ),
        ),
        budget_snapshot=_budget(),
        audience_context=AudienceContextV1(audience=KnowledgeQueryAudienceV1.PERSONAL),
    )


def _hybrid_plan() -> EvidencePlanV1:
    return EvidencePlanV1(
        plan_id="plan-3",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=1,
        mode=QueryPolicyModeV2.HYBRID,
        indexed_retrieval_directive=IndexedRetrievalDirectiveV1(max_results=10),
        ordered_live_call_proposals=(
            LiveCallProposalV1(
                call_id="call-1",
                live_access_binding_id="live-1",
                capability_id="cap.read",
                typed_capability_request={"item_key": "ITEM-1"},
            ),
        ),
        budget_snapshot=_budget(),
        audience_context=AudienceContextV1(audience=KnowledgeQueryAudienceV1.PERSONAL),
    )


def _v2_run(**overrides: object) -> WorkspaceAskRunV2:
    payload = {
        "run_id": "run-v2-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "question": "What changed?",
        "status": AskRunStatus.COMPLETED,
        "query_mode": QueryPolicyModeV2.HYBRID,
        "configuration_revision": 1,
        "plan_id": "plan-3",
        "answer": "Both sources agree.",
        "citations": [
            IndexedWorkspaceCitationV1(
                evidence_id="idx:ws-1:doc-1:chunk-1",
                safe_display_name="Indexed doc",
                excerpt="Indexed excerpt",
                retrieved_at=_NOW,
                document_id="doc-1",
                source_id="src-1",
                workspace_id=_WORKSPACE,
                source_path="/docs/a.txt",
                file_name="a.txt",
            ),
            LiveWorkspaceCitationV1(
                evidence_id="live:call-1:item-1",
                safe_display_name="Live item",
                retrieved_at=_NOW,
                provider_id="provider-neutral",
                connection_safe_label="Live Connection",
                capability_id="cap.read",
                call_id="call-1",
            ),
        ],
        "persisted_evidence": [
            PersistedIndexedEvidenceV2(
                evidence_id="idx:ws-1:doc-1:chunk-1",
                safe_display_name="Indexed doc",
                retrieved_at=_NOW,
                content_hash=_SHA256,
                audience=AskAudienceV1.PERSONAL,
                source_id="src-1",
                document_id="doc-1",
                chunk_id="chunk-1",
            ),
            PersistedLiveEvidenceProvenanceV2(
                evidence_id="live:call-1:item-1",
                safe_display_name="Live item",
                retrieved_at=_NOW,
                content_hash=_SHA256,
                audience=AskAudienceV1.PERSONAL,
                provider_id="provider-neutral",
                live_access_binding_id="live-1",
                connection_ref="conn.live",
                capability_id="cap.read",
                call_id="call-1",
            ),
        ],
        "indexed_retrieval_status": HybridAskIndexedRetrievalStatusV1.COMPLETED,
        "live_execution_status": HybridAskLiveExecutionStatusV1.COMPLETED,
        "created_at": _NOW,
        "completed_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceAskRunV2.model_validate(payload)


def test_v1_ask_run_still_round_trips() -> None:
    from local_workspace_application.tests.workspaces.test_ask_workspace_persistence import _run

    store = InMemoryDocumentStore()
    repo = WorkspaceAskRepository(store)
    run = _run()
    repo.put_run(run)
    loaded = repo.get_run(tenant_id="tenant-a", run_id="run-1")
    assert loaded is not None
    assert loaded.run_id == "run-1"


def test_v1_ask_row_without_version_marker_reads_as_v1() -> None:
    store = InMemoryDocumentStore()
    repo = WorkspaceAskRepository(store)
    raw = {
        "run_id": "legacy-run",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "question": "legacy",
        "status": "completed",
        "evidence": [],
        "answer": "ok",
        "citations": [],
        "created_at": _NOW.isoformat(),
        "completed_at": _NOW.isoformat(),
        "error": None,
    }
    store.put(
        DocumentRecord(
            partition_key=f"lkw.ask_run:{_TENANT}:ask_run",
            row_key="legacy-run",
            data=raw,
        )
    )
    version = repo.get_stored_run_schema_version(tenant_id=_TENANT, run_id="legacy-run")
    assert version is AskRunSchemaVersion.V1


def test_durable_live_evidence_rejects_content_and_excerpt() -> None:
    with pytest.raises(ValidationError):
        PersistedLiveEvidenceProvenanceV2.model_validate(
            {
                "evidence_id": "live:call-1:item-1",
                "safe_display_name": "Live item",
                "retrieved_at": _NOW,
                "content_hash": _SHA256,
                "audience": "personal",
                "provider_id": "provider-neutral",
                "live_access_binding_id": "live-1",
                "connection_ref": "conn.live",
                "capability_id": "cap.read",
                "call_id": "call-1",
                "content": "forbidden",
            }
        )
    with pytest.raises(ValidationError):
        PersistedLiveEvidenceProvenanceV2.model_validate(
            {
                "evidence_id": "live:call-1:item-1",
                "safe_display_name": "Live item",
                "retrieved_at": _NOW,
                "content_hash": _SHA256,
                "audience": "personal",
                "provider_id": "provider-neutral",
                "live_access_binding_id": "live-1",
                "connection_ref": "conn.live",
                "capability_id": "cap.read",
                "call_id": "call-1",
                "excerpt": "forbidden",
            }
        )


def test_v2_hybrid_run_round_trip_has_no_live_body() -> None:
    store = InMemoryDocumentStore()
    repo = WorkspaceAskRepository(store)
    run = _v2_run()
    repo.put_run_v2(run)
    loaded = repo.get_run_v2(tenant_id=_TENANT, run_id="run-v2-1")
    assert loaded is not None
    serialized = json.dumps(loaded.model_dump(mode="json"))
    assert '"content"' not in serialized


def test_unknown_ask_schema_version_fails_safely() -> None:
    store = InMemoryDocumentStore()
    repo = WorkspaceAskRepository(store)
    store.put(
        DocumentRecord(
            partition_key=f"lkw.ask_run:{_TENANT}:ask_run",
            row_key="bad-run",
            data={"run_schema_version": 99, "run_id": "bad-run"},
        )
    )
    with pytest.raises(WorkspaceAskRepositoryError) as exc:
        repo.get_run_any(tenant_id=_TENANT, run_id="bad-run")
    assert exc.value.error_code == "ask_run_schema_version_unknown"


def test_v2_query_policy_modes_and_invalid_combinations() -> None:
    _v2_policy(
        mode=QueryPolicyModeV2.INDEXED_ONLY,
        allowed_connection_refs=(),
        allowed_capability_ids=(),
        max_live_calls=0,
    )
    with pytest.raises(ValidationError):
        _v2_policy(
            mode=QueryPolicyModeV2.HYBRID,
            allowed_connection_refs=(),
            allowed_capability_ids=("cap.read",),
            max_live_calls=1,
        )


def test_query_policy_canonical_sorting_and_deduplication() -> None:
    policy = _v2_policy(
        allowed_connection_refs=("conn.b", "conn.a", "conn.b"),
        allowed_capability_ids=("cap.b", "cap.a", "cap.b"),
    )
    assert policy.allowed_connection_refs == ("conn.a", "conn.b")
    assert policy.allowed_capability_ids == ("cap.a", "cap.b")


def test_v1_and_v2_query_policy_coexist_in_repository() -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    v1 = _v1_policy(effective_revision=1)
    v2 = _v2_policy(effective_revision=2)
    assert repo.put_knowledge_query_policy_version_if_absent(v1)
    assert repo.put_knowledge_query_policy_version_if_absent(v2)
    versions = repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(versions) == 2
    assert isinstance(versions[0], WorkspaceQueryPolicy)
    assert isinstance(versions[1], WorkspaceQueryPolicyV2)


def test_parse_query_policy_version_discriminators() -> None:
    v1_data = _v1_policy().model_dump(mode="json")
    assert "policy_schema_version" not in v1_data
    assert isinstance(parse_workspace_query_policy(v1_data), WorkspaceQueryPolicy)
    v2_data = _v2_policy().model_dump(mode="json")
    assert v2_data["policy_schema_version"] == 2
    assert isinstance(parse_workspace_query_policy(v2_data), WorkspaceQueryPolicyV2)
    with pytest.raises(ValueError, match="query_policy_schema_version_unknown"):
        parse_workspace_query_policy({"policy_schema_version": 9})


def test_effective_policy_resolution_matrix() -> None:
    config = _configuration(revision=1, query_policy=None)
    effective = resolve_effective_query_policy(
        requested_mode=QueryPolicyModeV2.INDEXED_ONLY,
        configuration=config,
        configuration_revision=1,
    )
    assert effective.mode is QueryPolicyModeV2.INDEXED_ONLY

    with pytest.raises(HybridAskPolicyError) as exc:
        resolve_effective_query_policy(
            requested_mode=QueryPolicyModeV2.HYBRID,
            configuration=config,
            configuration_revision=1,
        )
    assert exc.value.error_code == "query_policy_required"

    v1_config = _configuration(revision=1, query_policy=_v1_policy())
    resolve_effective_query_policy(
        requested_mode=QueryPolicyModeV2.INDEXED_ONLY,
        configuration=v1_config,
        configuration_revision=1,
    )
    with pytest.raises(HybridAskPolicyError) as exc:
        resolve_effective_query_policy(
            requested_mode=QueryPolicyModeV2.HYBRID,
            configuration=v1_config,
            configuration_revision=1,
        )
    assert exc.value.error_code == "query_mode_not_allowed"

    v2_config = _configuration(revision=2, query_policy=_v2_policy(effective_revision=2))
    resolve_effective_query_policy(
        requested_mode=QueryPolicyModeV2.HYBRID,
        configuration=v2_config,
        configuration_revision=2,
    )
    with pytest.raises(HybridAskPolicyError) as exc:
        resolve_effective_query_policy(
            requested_mode=QueryPolicyModeV2.HYBRID,
            configuration=v2_config,
            configuration_revision=1,
        )
    assert exc.value.error_code == "configuration_revision_mismatch"


def test_evidence_plan_validation_paths() -> None:
    catalog = _FakeCatalog({"conn.live": _descriptor()})
    envelope = _FakeEnvelopeValidator()
    scope = _FakeScopeValidator()

    indexed_policy = _v2_policy(
        mode=QueryPolicyModeV2.INDEXED_ONLY,
        allowed_connection_refs=(),
        allowed_capability_ids=(),
        max_live_calls=0,
        effective_revision=1,
    )
    indexed_config = _configuration(revision=1, query_policy=indexed_policy)
    indexed_effective = resolve_effective_query_policy(
        requested_mode=QueryPolicyModeV2.INDEXED_ONLY,
        configuration=indexed_config,
        configuration_revision=1,
    )
    validate_evidence_plan(
        plan=_indexed_only_plan(),
        configuration=indexed_config,
        effective_policy=indexed_effective,
        capability_catalog=catalog,
        request_envelope_validator=envelope,
        resource_scope_validator=scope,
    )

    live_policy = _v2_policy(mode=QueryPolicyModeV2.LIVE_ONLY, effective_revision=1)
    live_config = _configuration(revision=1, query_policy=live_policy)
    live_effective = resolve_effective_query_policy(
        requested_mode=QueryPolicyModeV2.LIVE_ONLY,
        configuration=live_config,
        configuration_revision=1,
    )
    validated_live = validate_evidence_plan(
        plan=_live_plan(),
        configuration=live_config,
        effective_policy=live_effective,
        capability_catalog=catalog,
        request_envelope_validator=envelope,
        resource_scope_validator=scope,
    )
    assert validated_live.executable_live_calls[0].connection_ref == "conn.live"
    assert "provider_id" not in validated_live.executable_live_calls[0].validated_request

    hybrid_policy = _v2_policy(mode=QueryPolicyModeV2.HYBRID, effective_revision=1)
    hybrid_config = _configuration(revision=1, query_policy=hybrid_policy)
    hybrid_effective = resolve_effective_query_policy(
        requested_mode=QueryPolicyModeV2.HYBRID,
        configuration=hybrid_config,
        configuration_revision=1,
    )
    validate_evidence_plan(
        plan=_hybrid_plan(),
        configuration=hybrid_config,
        effective_policy=hybrid_effective,
        capability_catalog=catalog,
        request_envelope_validator=envelope,
        resource_scope_validator=scope,
    )


def test_evidence_plan_rejects_forbidden_model_fields_and_disabled_binding() -> None:
    with pytest.raises(ValidationError):
        LiveCallProposalV1(
            call_id="call-1",
            live_access_binding_id="live-1",
            capability_id="cap.read",
            typed_capability_request={"item_key": "ITEM-1"},
            connection_ref="conn.live",
        )
    disabled_binding = _live_binding(status=LiveAccessBindingStatusV1.DISABLED)
    config = WorkspaceKnowledgeConfigurationV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=1,
        connection_attachments=(_attachment(),),
        indexed_sources=(),
        live_access_bindings=(disabled_binding,),
        query_policy=_v2_policy(mode=QueryPolicyModeV2.LIVE_ONLY, effective_revision=1),
        updated_at=_NOW,
    )
    with pytest.raises(HybridAskPolicyError) as exc:
        validate_evidence_plan(
            plan=_live_plan(),
            configuration=config,
            effective_policy=resolve_effective_query_policy(
                requested_mode=QueryPolicyModeV2.LIVE_ONLY,
                configuration=config,
                configuration_revision=1,
            ),
            capability_catalog=_FakeCatalog({"conn.live": _descriptor()}),
            request_envelope_validator=_FakeEnvelopeValidator(),
            resource_scope_validator=_FakeScopeValidator(),
        )
    assert exc.value.error_code == "live_binding_unavailable"


def test_completed_v2_run_rejects_duplicate_and_unknown_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        _v2_run(
            persisted_evidence=[
                PersistedIndexedEvidenceV2(
                    evidence_id="idx:ws-1:doc-1:chunk-1",
                    safe_display_name="Indexed doc",
                    retrieved_at=_NOW,
                    content_hash=_SHA256,
                    audience=AskAudienceV1.PERSONAL,
                    source_id="src-1",
                    document_id="doc-1",
                ),
                PersistedIndexedEvidenceV2(
                    evidence_id="idx:ws-1:doc-1:chunk-1",
                    safe_display_name="Duplicate",
                    retrieved_at=_NOW,
                    content_hash=_SHA256,
                    audience=AskAudienceV1.PERSONAL,
                    source_id="src-2",
                    document_id="doc-2",
                ),
            ]
        )
    with pytest.raises(ValidationError):
        _v2_run(
            citations=[
                IndexedWorkspaceCitationV1(
                    evidence_id="idx:missing:doc:chunk",
                    safe_display_name="Missing",
                    retrieved_at=_NOW,
                    document_id="doc-1",
                    source_id="src-1",
                    workspace_id=_WORKSPACE,
                    source_path="/docs/a.txt",
                    file_name="a.txt",
                )
            ],
            persisted_evidence=[],
        )


def test_live_citation_rejects_excerpt_field() -> None:
    with pytest.raises(ValidationError):
        LiveWorkspaceCitationV1(
            evidence_id="live:call-1:item-1",
            safe_display_name="Live item",
            excerpt="secret",
            retrieved_at=_NOW,
            provider_id="provider-neutral",
            connection_safe_label="Live Connection",
            capability_id="cap.read",
            call_id="call-1",
        )


def test_serialized_hybrid_run_has_indexed_excerpt_but_no_live_excerpt() -> None:
    run = _v2_run()
    serialized = json.dumps(run.model_dump(mode="json"))
    assert '"excerpt": "Indexed excerpt"' in serialized
    live_citation = run.citations[1]
    assert isinstance(live_citation, LiveWorkspaceCitationV1)
    live_payload = live_citation.model_dump(mode="json")
    assert "excerpt" not in live_payload
    for item in run.model_dump(mode="json")["persisted_evidence"]:
        if item["evidence_type"] == "live":
            assert "content" not in item
            assert "excerpt" not in item


def test_mixed_v1_v2_ask_partition_listing() -> None:
    from local_workspace_application.tests.workspaces.test_ask_workspace_persistence import _run

    store = InMemoryDocumentStore()
    repo = WorkspaceAskRepository(store)
    v1_runs = [_run(run_id="run-v1-a"), _run(run_id="run-v1-b", workspace_id="ws-2")]
    v2_runs = [
        _v2_run(run_id="run-v2-a"),
        _v2_run(run_id="run-v2-b", workspace_id="ws-2"),
    ]
    for run in v1_runs:
        repo.put_run(run)
    for run in v2_runs:
        repo.put_run_v2(run)

    listed_v1 = {run.run_id for run in repo.list_runs(tenant_id=_TENANT)}
    listed_v2 = {run.run_id for run in repo.list_runs_v2(tenant_id=_TENANT)}
    assert listed_v1 == {"run-v1-a", "run-v1-b"}
    assert listed_v2 == {"run-v2-a", "run-v2-b"}


def test_descriptor_provider_mismatch_fails_closed() -> None:
    catalog = _FakeCatalog(
        {
            "conn.live": _descriptor(
                provider_id="other-provider",
                integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
            )
        }
    )
    config = _configuration(
        revision=1,
        query_policy=_v2_policy(mode=QueryPolicyModeV2.LIVE_ONLY, effective_revision=1),
    )
    with pytest.raises(HybridAskPolicyError) as exc:
        validate_evidence_plan(
            plan=_live_plan(),
            configuration=config,
            effective_policy=resolve_effective_query_policy(
                requested_mode=QueryPolicyModeV2.LIVE_ONLY,
                configuration=config,
                configuration_revision=1,
            ),
            capability_catalog=catalog,
            request_envelope_validator=_FakeEnvelopeValidator(),
            resource_scope_validator=_FakeScopeValidator(),
        )
    assert exc.value.error_code == "live_capability_unavailable"


def test_descriptor_integration_kind_mismatch_fails_closed() -> None:
    catalog = _FakeCatalog(
        {
            "conn.live": _descriptor(
                integration_kind=IntegrationCategory.ISSUE_TRACKER,
            )
        }
    )
    config = _configuration(
        revision=1,
        query_policy=_v2_policy(mode=QueryPolicyModeV2.LIVE_ONLY, effective_revision=1),
    )
    with pytest.raises(HybridAskPolicyError) as exc:
        validate_evidence_plan(
            plan=_live_plan(),
            configuration=config,
            effective_policy=resolve_effective_query_policy(
                requested_mode=QueryPolicyModeV2.LIVE_ONLY,
                configuration=config,
                configuration_revision=1,
            ),
            capability_catalog=catalog,
            request_envelope_validator=_FakeEnvelopeValidator(),
            resource_scope_validator=_FakeScopeValidator(),
        )
    assert exc.value.error_code == "live_capability_unavailable"


def test_live_binding_not_found_vs_unavailable() -> None:
    config = _configuration(
        revision=1,
        query_policy=_v2_policy(mode=QueryPolicyModeV2.LIVE_ONLY, effective_revision=1),
    )
    missing_plan = EvidencePlanV1(
        plan_id="plan-missing",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=1,
        mode=QueryPolicyModeV2.LIVE_ONLY,
        indexed_retrieval_directive=None,
        ordered_live_call_proposals=(
            LiveCallProposalV1(
                call_id="call-1",
                live_access_binding_id="missing-binding",
                capability_id="cap.read",
                typed_capability_request={"item_key": "ITEM-1"},
            ),
        ),
        budget_snapshot=_budget(),
        audience_context=AudienceContextV1(audience=KnowledgeQueryAudienceV1.PERSONAL),
    )
    with pytest.raises(HybridAskPolicyError) as exc:
        validate_evidence_plan(
            plan=missing_plan,
            configuration=config,
            effective_policy=resolve_effective_query_policy(
                requested_mode=QueryPolicyModeV2.LIVE_ONLY,
                configuration=config,
                configuration_revision=1,
            ),
            capability_catalog=_FakeCatalog({"conn.live": _descriptor()}),
            request_envelope_validator=_FakeEnvelopeValidator(),
            resource_scope_validator=_FakeScopeValidator(),
        )
    assert exc.value.error_code == "live_binding_not_found"


def _two_call_plan(*, reverse: bool = False) -> EvidencePlanV1:
    proposals = (
        LiveCallProposalV1(
            call_id="call-a",
            live_access_binding_id="live-1",
            capability_id="cap.a",
            typed_capability_request={"item_key": "A"},
        ),
        LiveCallProposalV1(
            call_id="call-b",
            live_access_binding_id="live-1",
            capability_id="cap.b",
            typed_capability_request={"item_key": "B"},
        ),
    )
    if reverse:
        proposals = tuple(reversed(proposals))
    return EvidencePlanV1(
        plan_id="plan-two-call",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=1,
        mode=QueryPolicyModeV2.LIVE_ONLY,
        indexed_retrieval_directive=None,
        ordered_live_call_proposals=proposals,
        budget_snapshot=_budget(),
        audience_context=AudienceContextV1(audience=KnowledgeQueryAudienceV1.PERSONAL),
    )


def _two_call_configuration() -> WorkspaceKnowledgeConfigurationV1:
    binding = _live_binding(
        allowed_capability_ids=("cap.a", "cap.b"),
    )
    return WorkspaceKnowledgeConfigurationV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=1,
        connection_attachments=(_attachment(),),
        indexed_sources=(),
        live_access_bindings=(binding,),
        query_policy=_v2_policy(
            mode=QueryPolicyModeV2.LIVE_ONLY,
            allowed_capability_ids=("cap.a", "cap.b"),
            max_live_calls=2,
            effective_revision=1,
        ),
        updated_at=_NOW,
    )


def test_per_call_budgets_are_independent_of_proposal_order() -> None:
    catalog = _FakeCatalog(
        {
            "conn.live": (
                _descriptor(
                    capability_id="cap.a",
                    max_result_items=10,
                    max_result_bytes=1_000,
                ),
                _descriptor(
                    capability_id="cap.b",
                    max_result_items=20,
                    max_result_bytes=2_000,
                ),
            )
        }
    )
    config = _two_call_configuration()
    effective = resolve_effective_query_policy(
        requested_mode=QueryPolicyModeV2.LIVE_ONLY,
        configuration=config,
        configuration_revision=1,
    )
    forward = validate_evidence_plan(
        plan=_two_call_plan(reverse=False),
        configuration=config,
        effective_policy=effective,
        capability_catalog=catalog,
        request_envelope_validator=_FakeEnvelopeValidator(),
        resource_scope_validator=_FakeScopeValidator(),
    )
    reverse = validate_evidence_plan(
        plan=_two_call_plan(reverse=True),
        configuration=config,
        effective_policy=effective,
        capability_catalog=catalog,
        request_envelope_validator=_FakeEnvelopeValidator(),
        resource_scope_validator=_FakeScopeValidator(),
    )
    forward_by_capability = {
        call.capability_id: call.effective_budget for call in forward.executable_live_calls
    }
    reverse_by_capability = {
        call.capability_id: call.effective_budget for call in reverse.executable_live_calls
    }
    assert forward_by_capability == reverse_by_capability
    assert forward_by_capability["cap.a"].max_result_items == 10
    assert forward_by_capability["cap.a"].max_result_bytes == 1_000
    assert forward_by_capability["cap.b"].max_result_items == 20
    assert forward_by_capability["cap.b"].max_result_bytes == 2_000
    assert forward.effective_budget.max_result_items == 50
    assert reverse.effective_budget.max_result_items == 50


@pytest.mark.parametrize(
    "mutation",
    [
        lambda: _v2_run(
            execution_receipts=[
                LiveExecutionReceiptV1(
                    receipt_id="receipt-1",
                    run_id="other-run",
                    call_id="call-1",
                    live_access_binding_id="live-1",
                    capability_id="cap.read",
                    started_at=_NOW,
                    completed_at=_NOW,
                    item_count=1,
                    byte_count=10,
                    content_hash=_SHA256,
                    normalized_outcome="ok",
                )
            ]
        ),
        lambda: _v2_run(
            citations=[
                IndexedWorkspaceCitationV1(
                    evidence_id="idx:ws-1:doc-1:chunk-1",
                    safe_display_name="Indexed doc",
                    excerpt="Indexed excerpt",
                    retrieved_at=_NOW,
                    document_id="doc-1",
                    source_id="src-1",
                    workspace_id=_WORKSPACE,
                    source_path="/docs/a.txt",
                    file_name="a.txt",
                ),
                LiveWorkspaceCitationV1(
                    evidence_id="live:call-1:item-1",
                    safe_display_name="Live item",
                    retrieved_at=_NOW,
                    provider_id="provider-neutral",
                    connection_safe_label="Live Connection",
                    capability_id="cap.read",
                    call_id="call-wrong",
                ),
            ]
        ),
        lambda: _v2_run(
            persisted_evidence=[
                PersistedIndexedEvidenceV2(
                    evidence_id="idx:ws-1:doc-1:chunk-1",
                    safe_display_name="Indexed doc",
                    retrieved_at=_NOW,
                    content_hash=_SHA256,
                    audience=AskAudienceV1.PERSONAL,
                    source_id="src-1",
                    document_id="doc-1",
                    chunk_id="chunk-1",
                ),
                PersistedLiveEvidenceProvenanceV2(
                    evidence_id="live:call-1:item-1",
                    safe_display_name="Live item",
                    retrieved_at=_NOW,
                    content_hash=_SHA256,
                    audience=AskAudienceV1.PERSONAL,
                    provider_id="provider-neutral",
                    live_access_binding_id="live-1",
                    connection_ref="conn.live",
                    capability_id="cap.read",
                    call_id="call-1",
                ),
                PersistedLiveEvidenceProvenanceV2(
                    evidence_id="live:call-2:item-2",
                    safe_display_name="Duplicate call",
                    retrieved_at=_NOW,
                    content_hash=_SHA256,
                    audience=AskAudienceV1.PERSONAL,
                    provider_id="provider-neutral",
                    live_access_binding_id="live-1",
                    connection_ref="conn.live",
                    capability_id="cap.read",
                    call_id="call-1",
                ),
            ]
        ),
    ],
)
def test_v2_run_receipt_integrity_mismatches_fail(mutation) -> None:
    with pytest.raises(ValidationError):
        mutation()


def test_nested_forbidden_model_controlled_fields_rejected() -> None:
    nested_cases = [
        {"nested": {"url": "https://evil"}},
        {"nested": {"provider_id": "evil"}},
        {"nested": {"connection_ref": "conn.evil"}},
        {"nested": {"headers": {"Authorization": "secret"}}},
        {"nested": {"jql": "project = SECRET"}},
        {"nested": {"sql": "SELECT * FROM secrets"}},
        {"nested": {"provider_client": {"token": "secret"}}},
        {"items": [{"url": "https://evil"}]},
    ]
    for typed_request in nested_cases:
        with pytest.raises(ValidationError):
            LiveCallProposalV1(
                call_id="call-1",
                live_access_binding_id="live-1",
                capability_id="cap.read",
                typed_capability_request=typed_request,
            )


def _workspace_record() -> Workspace:
    return Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Workspace",
        status=WorkspaceStatus.ACTIVE,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _v2_query_policy_intent() -> UpdateQueryPolicyMutationIntent:
    return UpdateQueryPolicyMutationIntent(
        mode=QueryPolicyModeV2.HYBRID,
        allowed_connection_refs=("conn.live",),
        allowed_capability_ids=("cap.read",),
        max_live_calls=2,
        max_total_duration_ms=30_000,
        max_result_items=50,
        max_result_bytes=1_048_576,
        live_result_retention=LiveResultRetentionV1.EPHEMERAL,
        policy_schema_version=2,
    )


def _policy_hashes(intent: UpdateQueryPolicyMutationIntent) -> tuple[str, str]:
    request = normalize_update_query_policy_request_hash(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        policy_schema_version=intent.policy_schema_version,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )
    semantic = semantic_identity_hash_for_query_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        policy_schema_version=intent.policy_schema_version,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )
    return request, semantic


def test_query_policy_v2_survives_committed_mutation_and_restart() -> None:
    validate_configuration_idempotency_hash(_IDEMPOTENCY)
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_workspace(_workspace_record())
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            updated_at=_NOW,
        )
    )
    lookup = ManagedWorkspaceService(repo)
    config_service = WorkspaceKnowledgeConfigurationService(repo, lookup)
    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        config_service,
        {WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY: _POLICY_HANDLER},
        clock=lambda: _NOW,
        mutation_id_factory=lambda: "mutation-v2-policy",
    )
    intent = _v2_query_policy_intent()
    request_hash, semantic_hash = _policy_hashes(intent)
    manifest_hash = query_policy_stage_manifest_hash(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        policy_schema_version=intent.policy_schema_version,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )
    mutation = WorkspaceKnowledgeMutationRecord(
        mutation_id="mutation-v2-policy",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY,
        idempotency_key_hash=_IDEMPOTENCY,
        normalized_request_hash=request_hash,
        semantic_identity_hash=semantic_hash,
        stage_manifest_hash=manifest_hash,
        target_revision=1,
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        result_entity_type="query_policy",
        result_entity_id="query-policy",
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    repo.replace_knowledge_configuration_head_if_match(
        expected=head,
        replacement=head.model_copy(
            update={
                "pending_revision": 1,
                "pending_mutation_id": mutation.mutation_id,
                "updated_at": _NOW,
            }
        ),
    )
    _POLICY_HANDLER.stage(
        repository=repo,
        mutation=mutation,
        target_revision=1,
        intent=intent,
        now=_NOW,
    )
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    projected = config_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert projected is not None
    assert isinstance(projected.query_policy, WorkspaceQueryPolicyV2)
    assert projected.query_policy.mode is QueryPolicyModeV2.HYBRID

    restarted_repo = ManagedWorkspaceRepository(store)
    restarted_lookup = ManagedWorkspaceService(restarted_repo)
    restarted_config = WorkspaceKnowledgeConfigurationService(restarted_repo, restarted_lookup)
    reloaded = restarted_config.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert reloaded is not None
    assert isinstance(reloaded.query_policy, WorkspaceQueryPolicyV2)
    assert reloaded.query_policy == projected.query_policy
