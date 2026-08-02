# © Artur Czarnecki. All rights reserved.

"""Hash helpers for Workspace Knowledge Configuration mutations."""

from __future__ import annotations

import hashlib
import json

from local_workspace_application.workspaces.connected_source_ids import (
    workspace_indexed_source_semantic_hash,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    WorkspaceKnowledgeMutationOperationV1,
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
