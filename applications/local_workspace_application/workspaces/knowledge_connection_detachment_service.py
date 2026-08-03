# © Artur Czarnecki. All rights reserved.

"""Workspace Connection Detachment domain service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypeVar

from local_workspace_application.workspaces.knowledge_access_service import TenantKnowledgeSourceBindingPort
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    connection_attachment_id, connection_attachment_semantic_identity_hash,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1, WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1, WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeConfigurationV1, WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationOutcomeV1, WorkspaceKnowledgeMutationStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine, WorkspaceKnowledgeConfigurationMutationError,
    WorkspaceKnowledgeMutationExecutionDispositionV1, WorkspaceKnowledgeMutationExecutionResult,
)
from local_workspace_application.workspaces.knowledge_configuration_service import WorkspaceKnowledgeConfigurationService
from local_workspace_application.workspaces.knowledge_connection_attachment_service import WorkspaceConnectionAttachmentError
from local_workspace_application.workspaces.knowledge_connection_detachment_handler import (
    DetachConnectionMutationIntent, detach_connection_request_hash, detach_connection_stage_manifest_hash,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_RESULT_TYPE = "connection_attachment"
_PORT_ERR = "connection_detach_dependency_resolution_failed"
_IDEMPOTENCY_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


def _validate_idempotency_key_hash(value: object) -> None:
    if not isinstance(value, str) or _IDEMPOTENCY_HASH_RE.fullmatch(value) is None:
        raise WorkspaceKnowledgeConfigurationMutationError("knowledge_configuration_idempotency_hash_invalid")


@dataclass(frozen=True, slots=True)
class DetachWorkspaceConnectionCommand:
    tenant_id: str
    workspace_id: str
    connection_ref: str
    expected_revision: int
    idempotency_key_hash: str


@dataclass(frozen=True, slots=True)
class DetachWorkspaceConnectionResult:
    attachment: WorkspaceConnectionAttachment
    configuration_revision: int
    disposition: WorkspaceKnowledgeMutationExecutionDispositionV1


def _incomplete() -> WorkspaceConnectionAttachmentError:
    return WorkspaceConnectionAttachmentError("connection_attachment_projection_incomplete")


def _highest(versions: list[Any], *, field: str, eid: str, rev: int) -> Any | None:
    m = [v for v in versions if getattr(v, field) == eid and v.effective_revision <= rev]
    if not m:
        return None
    top = max(m, key=lambda v: v.effective_revision)
    if sum(1 for v in m if v.effective_revision == top.effective_revision) > 1:
        raise _incomplete()
    return top


class WorkspaceConnectionDetachmentService:
    def __init__(
        self, *, configuration_service: WorkspaceKnowledgeConfigurationService,
        mutation_engine: WorkspaceKnowledgeConfigurationMutationEngine,
        tenant_binding_port: TenantKnowledgeSourceBindingPort, repository: ManagedWorkspaceRepository,
    ) -> None:
        self._configuration_service = configuration_service
        self._mutation_engine = mutation_engine
        self._tenant_binding_port = tenant_binding_port
        self._repository = repository

    def detach_connection(self, command: DetachWorkspaceConnectionCommand) -> DetachWorkspaceConnectionResult:
        _validate_idempotency_key_hash(command.idempotency_key_hash)
        tenant_id, workspace_id = command.tenant_id.strip(), command.workspace_id.strip()
        connection_ref = command.connection_ref.strip()
        if not connection_ref:
            raise WorkspaceConnectionAttachmentError("connection_attachment_not_found")
        attachment_id = connection_attachment_id(tenant_id=tenant_id, workspace_id=workspace_id, connection_ref=connection_ref)
        req_hash = detach_connection_request_hash(tenant_id=tenant_id, workspace_id=workspace_id, connection_ref=connection_ref)
        sem_hash = connection_attachment_semantic_identity_hash(tenant_id=tenant_id, workspace_id=workspace_id, connection_ref=connection_ref)
        existing = self._repository.get_knowledge_configuration_mutation(
            tenant_id=tenant_id, workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION,
            idempotency_key_hash=command.idempotency_key_hash,
        )
        if existing is not None and existing.normalized_request_hash != req_hash:
            raise WorkspaceKnowledgeConfigurationMutationError("configuration_idempotency_conflict")
        if existing and existing.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED and existing.normalized_request_hash == req_hash:
            result = self._mutation_engine.execute(
                tenant_id=tenant_id, workspace_id=workspace_id,
                operation=WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION,
                expected_revision=command.expected_revision, idempotency_key_hash=command.idempotency_key_hash,
                normalized_request_hash=req_hash, semantic_identity_hash=sem_hash,
                stage_manifest_hash=existing.stage_manifest_hash,
                intent=DetachConnectionMutationIntent(attachment_id=attachment_id, connection_ref=connection_ref, indexed_source_binding_ids=(), live_access_binding_ids=()),
            )
            if result.disposition is not WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY or result.mutation.status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED:
                raise _incomplete()
            return DetachWorkspaceConnectionResult(
                attachment=_hist_att(self._repository, result, attachment_id, tenant_id, workspace_id, connection_ref),
                configuration_revision=result.configuration_revision, disposition=result.disposition,
            )
        configuration = self._configuration_service.get_configuration(tenant_id=tenant_id, workspace_id=workspace_id)
        if configuration is None:
            raise WorkspaceConnectionAttachmentError("workspace_not_found")
        if _cur_att(configuration, attachment_id, tenant_id, workspace_id, connection_ref) is None:
            raise WorkspaceConnectionAttachmentError("connection_attachment_not_found")
        intent = DetachConnectionMutationIntent(
            attachment_id=attachment_id, connection_ref=connection_ref,
            indexed_source_binding_ids=_idx_deps(configuration, self._tenant_binding_port, tenant_id, connection_ref),
            live_access_binding_ids=_live_deps(configuration, connection_ref),
        )
        result = self._mutation_engine.execute(
            tenant_id=tenant_id, workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION,
            expected_revision=command.expected_revision, idempotency_key_hash=command.idempotency_key_hash,
            normalized_request_hash=req_hash, semantic_identity_hash=sem_hash,
            stage_manifest_hash=detach_connection_stage_manifest_hash(
                attachment_id=attachment_id, connection_ref=connection_ref,
                indexed_source_binding_ids=intent.indexed_source_binding_ids,
                live_access_binding_ids=intent.live_access_binding_ids,
            ),
            intent=intent,
        )
        attachment = _hist_att(self._repository, result, attachment_id, tenant_id, workspace_id, connection_ref)
        if result.disposition is not WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY:
            _verify_cascade(self._repository, result.configuration_revision, tenant_id, workspace_id, intent.indexed_source_binding_ids, intent.live_access_binding_ids)
        return DetachWorkspaceConnectionResult(attachment=attachment, configuration_revision=result.configuration_revision, disposition=result.disposition)


def _cur_att(cfg: WorkspaceKnowledgeConfigurationV1, aid: str, tid: str, wid: str, cref: str) -> WorkspaceConnectionAttachment | None:
    m = [a for a in cfg.connection_attachments if a.attachment_id == aid]
    if len(m) != 1:
        return None
    a = m[0]
    if a.tenant_id != tid or a.workspace_id != wid or a.connection_ref != cref:
        raise _incomplete()
    return a


def _idx_deps(cfg: WorkspaceKnowledgeConfigurationV1, port: TenantKnowledgeSourceBindingPort, tid: str, cref: str) -> tuple[str, ...]:
    out: list[str] = []
    for b in cfg.indexed_sources:
        if b.status not in (WorkspaceIndexedSourceBindingStatusV1.ACTIVE, WorkspaceIndexedSourceBindingStatusV1.ERROR):
            continue
        ref = b.knowledge_source_binding_ref.strip()
        try:
            conn = _port_conn(port.get_binding(tenant_id=tid, binding_id=ref), tenant_id=tid, binding_ref=ref)
        except Exception:
            raise WorkspaceConnectionAttachmentError(_PORT_ERR) from None
        if conn is None:
            raise WorkspaceConnectionAttachmentError(_PORT_ERR)
        if conn == cref:
            out.append(b.indexed_source_binding_id)
    out.sort()
    return tuple(out)


def _port_conn(tb: object, *, tenant_id: str, binding_ref: str) -> str | None:
    try:
        if tb is None:
            return None
        tid, bid, cref = tb.tenant_id, tb.binding_id, tb.connection_ref
        if not isinstance(tid, str) or not isinstance(bid, str) or not isinstance(cref, str):
            return None
        s = cref.strip()
        if tid != tenant_id or bid.strip() != binding_ref or not s:
            return None
        return s
    except (AttributeError, TypeError):
        return None


def _live_deps(cfg: WorkspaceKnowledgeConfigurationV1, cref: str) -> tuple[str, ...]:
    return tuple(sorted(b.live_access_binding_id for b in cfg.live_access_bindings if b.connection_ref.strip() == cref and b.status is LiveAccessBindingStatusV1.ACTIVE))


def _hist_att(repo: ManagedWorkspaceRepository, result: WorkspaceKnowledgeMutationExecutionResult, aid: str, tid: str, wid: str, cref: str) -> WorkspaceConnectionAttachment:
    m = result.mutation
    if m.normalized_request_hash != detach_connection_request_hash(tenant_id=tid, workspace_id=wid, connection_ref=cref):
        raise _incomplete()
    if m.semantic_identity_hash != connection_attachment_semantic_identity_hash(tenant_id=tid, workspace_id=wid, connection_ref=cref):
        raise _incomplete()
    if m.result_entity_type != _RESULT_TYPE or m.result_entity_id != aid or m.committed_revision != result.configuration_revision:
        raise _incomplete()
    att = _highest(repo.list_knowledge_connection_attachment_versions(tenant_id=tid, workspace_id=wid), field="attachment_id", eid=aid, rev=result.configuration_revision)
    if att is None or att.tenant_id != tid or att.workspace_id != wid or att.connection_ref != cref or att.status is not WorkspaceConnectionAttachmentStatusV1.DETACHED:
        raise _incomplete()
    if m.outcome is WorkspaceKnowledgeMutationOutcomeV1.APPLIED:
        if m.target_revision != m.committed_revision or att.effective_revision != m.target_revision or att.mutation_id != m.mutation_id:
            raise _incomplete()
    elif m.outcome is WorkspaceKnowledgeMutationOutcomeV1.EXISTING_RESULT:
        if m.target_revision is not None or att.effective_revision > m.committed_revision:
            raise _incomplete()
    else:
        raise _incomplete()
    return att


def _verify_cascade(repo: ManagedWorkspaceRepository, rev: int, tid: str, wid: str, idx_ids: tuple[str, ...], live_ids: tuple[str, ...]) -> None:
    iv = repo.list_knowledge_indexed_source_versions(tenant_id=tid, workspace_id=wid)
    for bid in idx_ids:
        row = _highest(iv, field="indexed_source_binding_id", eid=bid, rev=rev)
        if row is None or row.status is not WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE:
            raise _incomplete()
    lv = repo.list_knowledge_live_access_versions(tenant_id=tid, workspace_id=wid)
    for bid in live_ids:
        row = _highest(lv, field="live_access_binding_id", eid=bid, rev=rev)
        if row is None or row.status is not LiveAccessBindingStatusV1.UNAVAILABLE:
            raise _incomplete()
