# © Artur Czarnecki. All rights reserved.

"""Verified managed-workspace search evidence mapping (shared by Search and Ask)."""

from __future__ import annotations

from typing import Any

from local_workspace_application.serving.workspace_schemas import WorkspaceSearchHitV1
from local_workspace_application.workspaces.idempotency import normalize_source_path
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationOwnershipV1,
    RepositoryKnowledgeMaterializationVisibility,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


class SearchEvidenceIncompleteError(RuntimeError):
    """Platform result lacked required typed search evidence fields."""


def extract_search_summary(task_result: Any) -> dict[str, Any] | None:
    execution = getattr(task_result, "execution_result", None)
    if execution is not None:
        structured = getattr(execution, "structured_data", None)
        if isinstance(structured, dict):
            summary = structured.get("search_summary")
            if isinstance(summary, dict):
                return summary
    return None


def map_search_hits(
    *,
    repository: ManagedWorkspaceRepository,
    tenant_id: str,
    workspace_id: str,
    task_result: Any,
    limit: int,
) -> list[WorkspaceSearchHitV1]:
    summary = extract_search_summary(task_result)
    if summary is None:
        raise SearchEvidenceIncompleteError("search_summary_missing")
    evidence = summary.get("evidence")
    if not isinstance(evidence, list):
        raise SearchEvidenceIncompleteError("search_evidence_missing")

    hits: list[WorkspaceSearchHitV1] = []
    incomplete = False
    visibility_filtered = False
    visibility = RepositoryKnowledgeMaterializationVisibility(repository)
    visibility_cache: dict[str, bool] = {}
    for item in evidence:
        if not isinstance(item, dict):
            continue
        document_id = str(item.get("document_id") or "").strip()
        source_id = str(item.get("source_id") or "").strip()
        item_workspace_id = str(item.get("workspace_id") or workspace_id).strip()
        source_path = str(item.get("source_path") or "").strip()
        file_name = str(item.get("file_name") or "").strip()
        score_raw = item.get("score")
        snippet = str(item.get("snippet") or item.get("text") or "").strip()
        metadata = item.get("metadata")

        if not document_id or not source_path or not file_name:
            incomplete = True
            continue
        if not isinstance(score_raw, (int, float)):
            incomplete = True
            continue
        if not snippet:
            incomplete = True
            continue
        if item_workspace_id != workspace_id:
            continue

        try:
            ref = repository.get_document_ref(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                document_id=document_id,
            )
        except (TypeError, ValueError, AttributeError):
            visibility_filtered = True
            continue
        if ref is None:
            normalized_path = normalize_source_path(source_path)
            ref = next(
                (
                    candidate
                    for candidate in repository.list_document_refs(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                    )
                    if normalize_source_path(candidate.source_path) == normalized_path
                ),
                None,
            )
            if ref is None:
                continue
        if ref.tenant_id != tenant_id:
            continue
        if source_id and ref.source_id != source_id:
            continue
        if normalize_source_path(ref.source_path) != normalize_source_path(source_path):
            # Provenance mismatch — drop rather than fabricate.
            continue
        ownership = ref.materialization_ownership or KnowledgeMaterializationOwnershipV1.legacy(
            tenant_id=ref.tenant_id,
            workspace_id=ref.workspace_id,
            source_id=ref.source_id,
        )
        ownership_key = ownership.identity_scope
        if ownership_key not in visibility_cache:
            try:
                visibility_cache[ownership_key] = visibility.is_visible(
                    ownership=ownership,
                    document_id=ref.document_id,
                    content_hash=ref.content_hash,
                )
            except (TypeError, ValueError, AttributeError):
                visibility_cache[ownership_key] = False
        if not visibility_cache[ownership_key]:
            visibility_filtered = True
            continue

        hits.append(
            WorkspaceSearchHitV1(
                document_id=document_id,
                source_id=ref.source_id,
                workspace_id=workspace_id,
                source_path=ref.source_path,
                file_name=file_name,
                score=float(score_raw),
                snippet=snippet,
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
            )
        )
        if len(hits) >= limit:
            break

    if not hits and incomplete:
        raise SearchEvidenceIncompleteError("search_evidence_incomplete")
    if not hits and visibility_filtered:
        return []
    if not hits and evidence:
        raise SearchEvidenceIncompleteError("search_evidence_unverified")
    return hits
