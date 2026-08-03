# © Artur Czarnecki. All rights reserved.

"""Hash helpers for Workspace Knowledge Configuration mutations."""

from __future__ import annotations

import hashlib
import json

from intergrax.integrations.contracts.base import IntegrationCategory
from local_workspace_application.workspaces.connected_source_ids import (
    workspace_indexed_source_semantic_hash,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    KnowledgeAudienceEligibilityV1,
    LiveAccessBindingStatusV1,
    LiveResultRetentionV1,
    QueryPolicyModeV1,
    WorkspaceKnowledgeMutationOperationV1,
)

_QUERY_POLICY_ENTITY_ID = "query-policy"


def normalize_query_policy_string_tuple(
    value: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for item in value:
        trimmed = item.strip()
        if not trimmed:
            raise ValueError("blank_query_policy_tuple_value")
        if trimmed not in seen:
            seen.add(trimmed)
            normalized.append(trimmed)
    return tuple(sorted(normalized))


def _normalized_query_policy_fields(
    *,
    allowed_connection_refs: tuple[str, ...] | list[str],
    allowed_capability_ids: tuple[str, ...] | list[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        normalize_query_policy_string_tuple(allowed_connection_refs),
        normalize_query_policy_string_tuple(allowed_capability_ids),
    )


def _canonical_json(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_create_indexed_source_request_hash(
    *,
    tenant_id: str,
    workspace_id: str,
    knowledge_source_binding_ref: str,
    sync_mode: IndexedSourceSyncModeV1,
    audience_eligibility: IndexedSourceAudienceEligibilityV1,
) -> str:
    payload = _canonical_json(
        {
            "operation": WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE.value,
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "knowledge_source_binding_ref": knowledge_source_binding_ref.strip(),
            "sync_mode": sync_mode.value,
            "audience_eligibility": audience_eligibility.value,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def semantic_identity_hash_for_create_indexed_source(
    *,
    tenant_id: str,
    workspace_id: str,
    knowledge_source_binding_ref: str,
) -> str:
    return workspace_indexed_source_semantic_hash(
        tenant_id,
        workspace_id,
        knowledge_source_binding_ref,
    )


def normalize_request_hash(
    *,
    operation: WorkspaceKnowledgeMutationOperationV1,
    tenant_id: str,
    workspace_id: str,
    knowledge_source_binding_ref: str,
    sync_mode: IndexedSourceSyncModeV1,
    audience_eligibility: IndexedSourceAudienceEligibilityV1,
) -> str:
    if operation is not WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE:
        raise ValueError("unsupported_operation_for_request_hash")
    return normalize_create_indexed_source_request_hash(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        knowledge_source_binding_ref=knowledge_source_binding_ref,
        sync_mode=sync_mode,
        audience_eligibility=audience_eligibility,
    )


def semantic_identity_hash(
    *,
    operation: WorkspaceKnowledgeMutationOperationV1,
    tenant_id: str,
    workspace_id: str,
    knowledge_source_binding_ref: str,
) -> str:
    if operation is not WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE:
        raise ValueError("unsupported_operation_for_semantic_identity_hash")
    return semantic_identity_hash_for_create_indexed_source(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        knowledge_source_binding_ref=knowledge_source_binding_ref,
    )


def normalize_disable_indexed_source_request_hash(
    *,
    tenant_id: str,
    workspace_id: str,
    indexed_source_binding_id: str,
) -> str:
    payload = _canonical_json(
        {
            "operation": WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE.value,
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "indexed_source_binding_id": indexed_source_binding_id.strip(),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def semantic_identity_hash_for_disable_indexed_source(
    *,
    tenant_id: str,
    workspace_id: str,
    knowledge_source_binding_ref: str,
) -> str:
    return workspace_indexed_source_semantic_hash(
        tenant_id,
        workspace_id,
        knowledge_source_binding_ref,
    )


def normalize_live_access_capability_set(
    allowed_capability_ids: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for capability_id in allowed_capability_ids:
        trimmed = capability_id.strip()
        if not trimmed:
            raise ValueError("blank_capability_id")
        if trimmed not in seen:
            seen.add(trimmed)
            normalized.append(trimmed)
    if not normalized:
        raise ValueError("allowed_capability_ids_required")
    return tuple(sorted(normalized))


def normalize_live_access_remote_resource_id(remote_resource_id: str | None) -> str | None:
    if remote_resource_id is None:
        return None
    trimmed = remote_resource_id.strip()
    if not trimmed:
        raise ValueError("blank_remote_resource_id")
    return trimmed


def semantic_identity_hash_for_live_access_binding(
    *,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
    normalized_remote_resource_id: str | None,
    normalized_capability_set: tuple[str, ...],
) -> str:
    payload = _canonical_json(
        {
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "connection_ref": connection_ref.strip(),
            "normalized_remote_resource_id": normalized_remote_resource_id,
            "normalized_capability_set": list(normalized_capability_set),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def live_access_binding_id_from_semantic_hash(semantic_hash: str) -> str:
    if len(semantic_hash) < 32:
        raise ValueError("semantic_hash_invalid")
    return f"live:{semantic_hash[:32]}"


def normalize_create_live_access_binding_request_hash(
    *,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
    remote_resource_id: str | None,
    allowed_capability_ids: tuple[str, ...],
    audience_eligibility: KnowledgeAudienceEligibilityV1,
) -> str:
    normalized_resource_id = normalize_live_access_remote_resource_id(remote_resource_id)
    normalized_capabilities = normalize_live_access_capability_set(allowed_capability_ids)
    payload = _canonical_json(
        {
            "operation": WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING.value,
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "connection_ref": connection_ref.strip(),
            "remote_resource_id": normalized_resource_id,
            "allowed_capability_ids": list(normalized_capabilities),
            "audience_eligibility": audience_eligibility.value,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def live_access_binding_stage_manifest_hash(
    *,
    tenant_id: str,
    workspace_id: str,
    live_access_binding_id: str,
    connection_ref: str,
    remote_resource_id: str | None,
    allowed_capability_ids: tuple[str, ...],
    audience_eligibility: KnowledgeAudienceEligibilityV1,
    derived_provider_id: str,
    derived_integration_kind: IntegrationCategory,
    derived_resource_type: str | None,
    derived_safe_display_label: str,
) -> str:
    normalized_resource_id = normalize_live_access_remote_resource_id(remote_resource_id)
    normalized_capabilities = normalize_live_access_capability_set(allowed_capability_ids)
    payload = _canonical_json(
        {
            "operation": WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING.value,
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "live_access_binding_id": live_access_binding_id.strip(),
            "connection_ref": connection_ref.strip(),
            "normalized_remote_resource_id": normalized_resource_id,
            "normalized_capability_set": list(normalized_capabilities),
            "audience_eligibility": audience_eligibility.value,
            "derived_provider_id": derived_provider_id,
            "derived_integration_kind": derived_integration_kind.value,
            "derived_resource_type": derived_resource_type,
            "derived_safe_display_label": derived_safe_display_label,
            "expected_status": LiveAccessBindingStatusV1.ACTIVE.value,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_disable_live_access_binding_request_hash(
    *,
    tenant_id: str,
    workspace_id: str,
    live_access_binding_id: str,
) -> str:
    payload = _canonical_json(
        {
            "operation": WorkspaceKnowledgeMutationOperationV1.DISABLE_LIVE_ACCESS_BINDING.value,
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "live_access_binding_id": live_access_binding_id.strip(),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _query_policy_config_payload(
    *,
    mode: QueryPolicyModeV1,
    allowed_connection_refs: tuple[str, ...] | list[str],
    allowed_capability_ids: tuple[str, ...] | list[str],
    max_live_calls: int,
    max_total_duration_ms: int,
    max_result_items: int,
    max_result_bytes: int,
    live_result_retention: LiveResultRetentionV1,
) -> dict[str, object]:
    normalized_connection_refs, normalized_capability_ids = _normalized_query_policy_fields(
        allowed_connection_refs=allowed_connection_refs,
        allowed_capability_ids=allowed_capability_ids,
    )
    return {
        "mode": mode.value,
        "allowed_connection_refs": list(normalized_connection_refs),
        "allowed_capability_ids": list(normalized_capability_ids),
        "max_live_calls": max_live_calls,
        "max_total_duration_ms": max_total_duration_ms,
        "max_result_items": max_result_items,
        "max_result_bytes": max_result_bytes,
        "live_result_retention": live_result_retention.value,
    }


def normalize_update_query_policy_request_hash(
    *,
    tenant_id: str,
    workspace_id: str,
    mode: QueryPolicyModeV1,
    allowed_connection_refs: tuple[str, ...],
    allowed_capability_ids: tuple[str, ...],
    max_live_calls: int,
    max_total_duration_ms: int,
    max_result_items: int,
    max_result_bytes: int,
    live_result_retention: LiveResultRetentionV1,
) -> str:
    payload = _canonical_json(
        {
            "operation": WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY.value,
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            **_query_policy_config_payload(
                mode=mode,
                allowed_connection_refs=allowed_connection_refs,
                allowed_capability_ids=allowed_capability_ids,
                max_live_calls=max_live_calls,
                max_total_duration_ms=max_total_duration_ms,
                max_result_items=max_result_items,
                max_result_bytes=max_result_bytes,
                live_result_retention=live_result_retention,
            ),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def semantic_identity_hash_for_query_policy(
    *,
    tenant_id: str,
    workspace_id: str,
    mode: QueryPolicyModeV1,
    allowed_connection_refs: tuple[str, ...],
    allowed_capability_ids: tuple[str, ...],
    max_live_calls: int,
    max_total_duration_ms: int,
    max_result_items: int,
    max_result_bytes: int,
    live_result_retention: LiveResultRetentionV1,
) -> str:
    payload = _canonical_json(
        {
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            **_query_policy_config_payload(
                mode=mode,
                allowed_connection_refs=allowed_connection_refs,
                allowed_capability_ids=allowed_capability_ids,
                max_live_calls=max_live_calls,
                max_total_duration_ms=max_total_duration_ms,
                max_result_items=max_result_items,
                max_result_bytes=max_result_bytes,
                live_result_retention=live_result_retention,
            ),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def query_policy_stage_manifest_hash(
    *,
    tenant_id: str,
    workspace_id: str,
    mode: QueryPolicyModeV1,
    allowed_connection_refs: tuple[str, ...],
    allowed_capability_ids: tuple[str, ...],
    max_live_calls: int,
    max_total_duration_ms: int,
    max_result_items: int,
    max_result_bytes: int,
    live_result_retention: LiveResultRetentionV1,
) -> str:
    payload = _canonical_json(
        {
            "operation": WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY.value,
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "query_policy_entity_id": _QUERY_POLICY_ENTITY_ID,
            **_query_policy_config_payload(
                mode=mode,
                allowed_connection_refs=allowed_connection_refs,
                allowed_capability_ids=allowed_capability_ids,
                max_live_calls=max_live_calls,
                max_total_duration_ms=max_total_duration_ms,
                max_result_items=max_result_items,
                max_result_bytes=max_result_bytes,
                live_result_retention=live_result_retention,
            ),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
