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
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    LiveResultRetentionV1,
    QueryPolicyModeV1,
    QueryPolicyModeV2,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicy,
    WorkspaceQueryPolicyV2,
    parse_workspace_query_policy,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_MUTATION = "mutation-1"
_SHA256 = "a" * 64


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
    def __init__(self, descriptors: dict[str, LiveCapabilityDescriptorV1]) -> None:
        self._descriptors = descriptors

    def get_descriptor(
        self,
        *,
        tenant_id: str,
        capability_id: str,
    ) -> LiveCapabilityDescriptorV1 | None:
        return self._descriptors.get(capability_id)


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
                excerpt="",
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
    catalog = _FakeCatalog({"cap.read": _descriptor()})
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
            capability_catalog=_FakeCatalog({"cap.read": _descriptor()}),
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
