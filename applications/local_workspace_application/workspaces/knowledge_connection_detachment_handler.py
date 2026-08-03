# © Artur Czarnecki. All rights reserved.

"""Detach Connection mutation handler (LKW-KNOWLEDGE-ACCESS-1D-1B)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, TypeVar

from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    connection_attachment_id,
    connection_attachment_semantic_identity_hash,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceLiveAccessBinding,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeExistingResult,
    WorkspaceKnowledgeStageInspection,
    WorkspaceKnowledgeStageStateV1,
    WorkspaceKnowledgeStagedResult,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_RESULT_ENTITY_TYPE = "connection_attachment"
_T = TypeVar("_T")
_INDEXED_STABLE = (
    "indexed_source_binding_id", "tenant_id", "workspace_id", "knowledge_source_binding_ref",
    "source_id", "sync_mode", "audience_eligibility", "semantic_identity_hash",
    "cached_safe_display_label", "created_at",
)
_LIVE_STABLE = (
    "live_access_binding_id", "tenant_id", "workspace_id", "connection_ref", "remote_resource_id",
    "allowed_capability_ids", "derived_provider_id", "derived_integration_kind",
    "derived_resource_type", "derived_safe_display_label", "audience_eligibility",
    "semantic_identity_hash", "created_at",
)


def _canonical_json(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_binding_ids(ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result = []
    for item in ids:
        trimmed = item.strip()
        if not trimmed:
            raise ValueError("binding_id_blank")
        if trimmed not in seen:
            seen.add(trimmed)
            result.append(trimmed)
    result.sort()
    return tuple(result)


def detach_connection_request_hash(*, tenant_id: str, workspace_id: str, connection_ref: str) -> str:
    payload = _canonical_json({
        "operation": WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION.value,
        "tenant_id": tenant_id.strip(), "workspace_id": workspace_id.strip(),
        "connection_ref": connection_ref.strip(),
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def detach_connection_stage_manifest_hash(
    *, attachment_id: str, connection_ref: str,
    indexed_source_binding_ids: tuple[str, ...], live_access_binding_ids: tuple[str, ...],
) -> str:
    payload = _canonical_json({
        "attachment_id": attachment_id, "connection_ref": connection_ref.strip(),
        "indexed_source_binding_ids": sorted({i.strip() for i in indexed_source_binding_ids if i.strip()}),
        "live_access_binding_ids": sorted({i.strip() for i in live_access_binding_ids if i.strip()}),
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DetachConnectionMutationIntent:
    attachment_id: str
    connection_ref: str
    indexed_source_binding_ids: tuple[str, ...]
    live_access_binding_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "indexed_source_binding_ids", _normalize_binding_ids(self.indexed_source_binding_ids))
        object.__setattr__(self, "live_access_binding_ids", _normalize_binding_ids(self.live_access_binding_ids))


class DetachConnectionMutationHandler:
    operation = WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION

    def find_existing_result(self, *, configuration: WorkspaceKnowledgeConfigurationV1, intent: object) -> WorkspaceKnowledgeExistingResult | None:
        if not isinstance(intent, DetachConnectionMutationIntent):
            raise ValueError("detach_connection_intent_required")
        attachment = next(
            (a for a in configuration.connection_attachments
             if a.attachment_id == intent.attachment_id and a.connection_ref == intent.connection_ref.strip()
             and a.status is WorkspaceConnectionAttachmentStatusV1.DETACHED),
            None,
        )
        if attachment is None:
            return None
        indexed = {b.indexed_source_binding_id: b for b in configuration.indexed_sources}
        live = {b.live_access_binding_id: b for b in configuration.live_access_bindings}
        if any(indexed.get(i) is None or indexed[i].status is not WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE for i in intent.indexed_source_binding_ids):
            return None
        if any(live.get(i) is None or live[i].status is not LiveAccessBindingStatusV1.UNAVAILABLE for i in intent.live_access_binding_ids):
            return None
        return WorkspaceKnowledgeExistingResult(result_entity_type=_RESULT_ENTITY_TYPE, result_entity_id=attachment.attachment_id)

    def stage(self, *, repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
              target_revision: int, intent: object, now: datetime) -> WorkspaceKnowledgeStagedResult:
        if not isinstance(intent, DetachConnectionMutationIntent):
            raise ValueError("detach_connection_intent_required")
        manifest = detach_connection_stage_manifest_hash(
            attachment_id=intent.attachment_id, connection_ref=intent.connection_ref,
            indexed_source_binding_ids=intent.indexed_source_binding_ids,
            live_access_binding_ids=intent.live_access_binding_ids,
        )
        if mutation.stage_manifest_hash != manifest:
            raise RuntimeError("detach_stage_manifest_mismatch")
        base = target_revision - 1
        prev_att = _select_prev(repository, mutation, base, repository.list_knowledge_connection_attachment_versions, "attachment_id", intent.attachment_id)
        if prev_att is None:
            raise RuntimeError("connection_attachment_previous_missing")
        _put_attachment(repository, WorkspaceConnectionAttachment(
            attachment_id=prev_att.attachment_id, tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
            connection_ref=prev_att.connection_ref, safe_display_label=prev_att.safe_display_label,
            status=WorkspaceConnectionAttachmentStatusV1.DETACHED, mutation_id=mutation.mutation_id,
            effective_revision=target_revision, created_at=prev_att.created_at, updated_at=now,
        ))
        for binding_id in intent.indexed_source_binding_ids:
            prev = _select_prev(repository, mutation, base, repository.list_knowledge_indexed_source_versions, "indexed_source_binding_id", binding_id)
            if prev is None or prev.status not in (WorkspaceIndexedSourceBindingStatusV1.ACTIVE, WorkspaceIndexedSourceBindingStatusV1.ERROR):
                raise RuntimeError("indexed_source_previous_invalid")
            _put_indexed(repository, prev.model_copy(update={
                "status": WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE, "mutation_id": mutation.mutation_id,
                "effective_revision": target_revision, "updated_at": now,
            }))
        for binding_id in intent.live_access_binding_ids:
            prev = _select_prev(repository, mutation, base, repository.list_knowledge_live_access_versions, "live_access_binding_id", binding_id)
            if prev is None or prev.status is not LiveAccessBindingStatusV1.ACTIVE or prev.connection_ref.strip() != intent.connection_ref.strip():
                raise RuntimeError("live_access_previous_invalid")
            _put_live(repository, prev.model_copy(update={
                "status": LiveAccessBindingStatusV1.UNAVAILABLE, "mutation_id": mutation.mutation_id,
                "effective_revision": target_revision, "updated_at": now,
            }))
        return WorkspaceKnowledgeStagedResult(result_entity_type=_RESULT_ENTITY_TYPE, result_entity_id=intent.attachment_id)

    def inspect_staged(self, *, repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord) -> WorkspaceKnowledgeStageInspection:
        conflict = WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT)
        if mutation.target_revision is None:
            return conflict
        base = mutation.target_revision - 1
        owned_att = _owned(repository, mutation, repository.list_knowledge_connection_attachment_versions)
        owned_idx = _owned(repository, mutation, repository.list_knowledge_indexed_source_versions)
        owned_live = _owned(repository, mutation, repository.list_knowledge_live_access_versions)
        if not owned_att and not owned_idx and not owned_live:
            return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.ABSENT)
        if len(owned_att) > 1:
            return conflict
        staged_att = owned_att[0] if owned_att else None
        idx_ids: list[str] = []
        for row in owned_idx:
            if sum(1 for r in owned_idx if r.indexed_source_binding_id == row.indexed_source_binding_id) > 1:
                return conflict
            if not _valid_indexed(repository, mutation, row, base):
                return conflict
            idx_ids.append(row.indexed_source_binding_id)
        live_ids: list[str] = []
        conn_ref = staged_att.connection_ref if staged_att else ""
        for row in owned_live:
            if sum(1 for r in owned_live if r.live_access_binding_id == row.live_access_binding_id) > 1:
                return conflict
            if not _valid_live(repository, mutation, row, base, conn_ref):
                return conflict
            live_ids.append(row.live_access_binding_id)
        if staged_att is None or not _valid_attachment(repository, mutation, staged_att, base):
            return conflict if staged_att else WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED)
        manifest = detach_connection_stage_manifest_hash(
            attachment_id=staged_att.attachment_id, connection_ref=staged_att.connection_ref,
            indexed_source_binding_ids=tuple(idx_ids), live_access_binding_ids=tuple(live_ids),
        )
        if mutation.stage_manifest_hash != manifest:
            return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED)
        if mutation.semantic_identity_hash != connection_attachment_semantic_identity_hash(
            tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id, connection_ref=staged_att.connection_ref,
        ):
            return conflict
        if mutation.result_entity_type not in (None, _RESULT_ENTITY_TYPE):
            return conflict
        if mutation.result_entity_id not in (None, staged_att.attachment_id):
            return conflict
        return WorkspaceKnowledgeStageInspection(
            state=WorkspaceKnowledgeStageStateV1.COMPLETE_VALID,
            result_entity_type=_RESULT_ENTITY_TYPE, result_entity_id=staged_att.attachment_id,
        )

    def cleanup_staged(self, *, repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
                       inspection: WorkspaceKnowledgeStageInspection) -> bool:
        if inspection.state is WorkspaceKnowledgeStageStateV1.ABSENT:
            return True
        if inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            return False
        if inspection.state not in (WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED, WorkspaceKnowledgeStageStateV1.COMPLETE_VALID):
            return False
        base = (mutation.target_revision or 0) - 1
        for row in _owned(repository, mutation, repository.list_knowledge_live_access_versions):
            if not _valid_live(repository, mutation, row, base, row.connection_ref) or not repository.delete_knowledge_live_access_version_if_match(row):
                return False
        for row in _owned(repository, mutation, repository.list_knowledge_indexed_source_versions):
            if not _valid_indexed(repository, mutation, row, base) or not repository.delete_knowledge_indexed_source_version_if_match(row):
                return False
        for row in _owned(repository, mutation, repository.list_knowledge_connection_attachment_versions):
            if not _valid_attachment(repository, mutation, row, base) or not repository.delete_knowledge_connection_attachment_version_if_match(row):
                return False
        return not (_owned(repository, mutation, repository.list_knowledge_connection_attachment_versions)
                    or _owned(repository, mutation, repository.list_knowledge_indexed_source_versions)
                    or _owned(repository, mutation, repository.list_knowledge_live_access_versions))


def _owned(repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
           list_fn: Callable[..., list[Any]]) -> list[Any]:
    return [v for v in list_fn(tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id)
            if v.mutation_id == mutation.mutation_id and v.effective_revision == mutation.target_revision]


def _select_prev(repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord, base_revision: int,
                 list_fn: Callable[..., list[_T]], id_field: str, entity_id: str) -> _T | None:
    matches = [v for v in list_fn(tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id)
               if getattr(v, id_field) == entity_id and v.effective_revision <= base_revision
               and v.tenant_id == mutation.tenant_id and v.workspace_id == mutation.workspace_id]
    if not matches:
        return None
    highest = max(matches, key=lambda v: v.effective_revision)
    if sum(1 for v in matches if v.effective_revision == highest.effective_revision) > 1:
        raise RuntimeError("duplicate_highest_version")
    return highest


def _put_attachment(repository: ManagedWorkspaceRepository, attachment: WorkspaceConnectionAttachment) -> None:
    expected = connection_attachment_id(tenant_id=attachment.tenant_id, workspace_id=attachment.workspace_id, connection_ref=attachment.connection_ref)
    if attachment.attachment_id != expected:
        raise RuntimeError("connection_attachment_identity_conflict")
    if repository.put_knowledge_connection_attachment_version_if_absent(attachment):
        return
    existing = repository.get_knowledge_connection_attachment_version(
        tenant_id=attachment.tenant_id, workspace_id=attachment.workspace_id,
        attachment_id=attachment.attachment_id, effective_revision=attachment.effective_revision,
    )
    if existing != attachment:
        raise RuntimeError("connection_attachment_stage_conflict")


def _put_indexed(repository: ManagedWorkspaceRepository, staged: WorkspaceIndexedSourceBinding) -> None:
    if repository.put_knowledge_indexed_source_version_if_absent(staged):
        return
    existing = repository.get_knowledge_indexed_source_version(
        tenant_id=staged.tenant_id, workspace_id=staged.workspace_id,
        indexed_source_binding_id=staged.indexed_source_binding_id, effective_revision=staged.effective_revision,
    )
    if existing != staged:
        raise RuntimeError("indexed_source_stage_conflict")


def _put_live(repository: ManagedWorkspaceRepository, staged: WorkspaceLiveAccessBinding) -> None:
    if repository.put_knowledge_live_access_version_if_absent(staged):
        return
    existing = repository.get_knowledge_live_access_version(
        tenant_id=staged.tenant_id, workspace_id=staged.workspace_id,
        live_access_binding_id=staged.live_access_binding_id, effective_revision=staged.effective_revision,
    )
    if existing != staged:
        raise RuntimeError("live_access_stage_conflict")


def _owned_row_ok(row: Any, mutation: WorkspaceKnowledgeMutationRecord) -> bool:
    return row.mutation_id == mutation.mutation_id and row.tenant_id == mutation.tenant_id and row.workspace_id == mutation.workspace_id and row.effective_revision == mutation.target_revision


def _valid_attachment(repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
                      staged: WorkspaceConnectionAttachment, base_revision: int) -> bool:
    if not _owned_row_ok(staged, mutation) or staged.status is not WorkspaceConnectionAttachmentStatusV1.DETACHED:
        return False
    if staged.attachment_id != connection_attachment_id(tenant_id=staged.tenant_id, workspace_id=staged.workspace_id, connection_ref=staged.connection_ref):
        return False
    prev = _select_prev(repository, mutation, base_revision, repository.list_knowledge_connection_attachment_versions, "attachment_id", staged.attachment_id)
    return prev is not None and prev.attachment_id == staged.attachment_id and prev.connection_ref == staged.connection_ref and prev.safe_display_label == staged.safe_display_label and prev.created_at == staged.created_at


def _valid_indexed(repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
                   staged: WorkspaceIndexedSourceBinding, base_revision: int) -> bool:
    if not _owned_row_ok(staged, mutation) or staged.status is not WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE:
        return False
    prev = _select_prev(repository, mutation, base_revision, repository.list_knowledge_indexed_source_versions, "indexed_source_binding_id", staged.indexed_source_binding_id)
    if prev is None or prev.status not in (WorkspaceIndexedSourceBindingStatusV1.ACTIVE, WorkspaceIndexedSourceBindingStatusV1.ERROR):
        return False
    return all(getattr(prev, f) == getattr(staged, f) for f in _INDEXED_STABLE)


def _valid_live(repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
                staged: WorkspaceLiveAccessBinding, base_revision: int, connection_ref: str) -> bool:
    if not _owned_row_ok(staged, mutation) or staged.status is not LiveAccessBindingStatusV1.UNAVAILABLE:
        return False
    if staged.connection_ref.strip() != connection_ref.strip():
        return False
    prev = _select_prev(repository, mutation, base_revision, repository.list_knowledge_live_access_versions, "live_access_binding_id", staged.live_access_binding_id)
    if prev is None or prev.status is not LiveAccessBindingStatusV1.ACTIVE or prev.connection_ref.strip() != connection_ref.strip():
        return False
    return all(getattr(prev, f) == getattr(staged, f) for f in _LIVE_STABLE)
