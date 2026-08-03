# © Artur Czarnecki. All rights reserved.

"""Tests for Workspace Connection Detachment domain service and handler."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import SLACK_CONVERSATION_CHANNEL_PROVIDER_ID
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import SLACK_CONVERSATION_SOURCE_KIND
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBinding, KnowledgeSourceBindingStatus, KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.tenant_connections import SafeTenantConnectionV1, TenantConnectionAdministrativeStatus
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationHandler, connection_attachment_id, connection_attachment_semantic_identity_hash,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1, WorkspaceConnectionAttachment, WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBinding, WorkspaceIndexedSourceBindingStatusV1, WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord, WorkspaceKnowledgeMutationStatusV1,
    WorkspaceLiveAccessBinding,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine, WorkspaceKnowledgeConfigurationMutationError,
    WorkspaceKnowledgeMutationExecutionDispositionV1, WorkspaceKnowledgeMutationRecoveryDispositionV1,
    WorkspaceKnowledgeStageStateV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import WorkspaceKnowledgeConfigurationService
from local_workspace_application.workspaces.knowledge_connection_attachment_service import (
    AttachWorkspaceConnectionCommand, WorkspaceConnectionAttachmentError, WorkspaceConnectionAttachmentService,
)
from local_workspace_application.workspaces.knowledge_connection_detachment_handler import (
    DetachConnectionMutationHandler, DetachConnectionMutationIntent, detach_connection_request_hash,
    detach_connection_stage_manifest_hash,
)
from local_workspace_application.workspaces.knowledge_connection_detachment_service import (
    DetachWorkspaceConnectionCommand, WorkspaceConnectionDetachmentService,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceSource, WorkspaceSourceStatus, WorkspaceSourceType, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

pytestmark = pytest.mark.unit
_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_TENANT, _WORKSPACE = "tenant-a", "workspace-1"
_CONNECTION, _CONNECTION_OTHER = "conn.primary", "conn.other"
_SHA256, _SHA256_B = "a" * 64, "b" * 64
_LABEL = "Primary Connection"
_ATT_HANDLER, _DET_HANDLER = AttachConnectionMutationHandler(), DetachConnectionMutationHandler()
_ATTACHMENT_ID = connection_attachment_id(tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION)


class _FakeWorkspaceLookup:
    def __init__(self, workspaces: dict[tuple[str, str], Workspace]) -> None:
        self._workspaces = workspaces
    def require_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        w = self._workspaces.get((tenant_id, workspace_id))
        return w if w and w.tenant_id == tenant_id else None


class _FakeConnectionPort:
    def get_connection(self, *, tenant_id: str, connection_ref: str) -> SafeTenantConnectionV1 | None:
        if tenant_id == _TENANT and connection_ref == _CONNECTION:
            return SafeTenantConnectionV1(connection_ref=_CONNECTION, tenant_id=_TENANT, provider_id="provider.slack",
                integration_kind=IntegrationCategory.CONVERSATION_CHANNEL, safe_display_name=_LABEL,
                administrative_status=TenantConnectionAdministrativeStatus.ACTIVE, configuration_version=1,
                connected_principal_ref=None, created_at=_NOW, updated_at=_NOW)
        return None
    def list_connections(self, *, tenant_id: str, limit: int = 100, administrative_status=None):
        c = self.get_connection(tenant_id=tenant_id, connection_ref=_CONNECTION)
        return (c,) if c else ()


class _TenantBindingPort:
    def __init__(self, bindings: dict[str, KnowledgeSourceBinding] | None = None, *, fail: bool = False) -> None:
        self._bindings, self._fail, self.call_count = bindings or {}, fail, 0
    def get_binding(self, *, tenant_id: str, binding_id: str):
        self.call_count += 1
        if self._fail:
            raise RuntimeError("lookup_failed")
        return self._bindings.get(binding_id)


def _tenant_binding(**kw) -> KnowledgeSourceBinding:
    d = dict(binding_id="ksb-1", tenant_id=_TENANT, provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL, source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        connection_ref=_CONNECTION, safe_display_name="Binding",
        scope=KnowledgeSourceScope(remote_scope_id="scope", remote_scope_type="slack_conversation", safe_display_name="Binding", parameters={}),
        status=KnowledgeSourceBindingStatus.ACTIVE, configuration_version=1)
    d.update(kw)
    return KnowledgeSourceBinding(**d)


def _workspace() -> Workspace:
    return Workspace(workspace_id=_WORKSPACE, tenant_id=_TENANT, name="Workspace", status=WorkspaceStatus.ACTIVE, created_at=_NOW, updated_at=_NOW)


class _CountingRepo(ManagedWorkspaceRepository):
    def __init__(self, store) -> None:
        super().__init__(store)
        self.publication_count = 0
    def replace_knowledge_configuration_head_if_match(self, *, expected, replacement):
        if expected.pending_mutation_id and replacement.pending_mutation_id is None and replacement.committed_revision > expected.committed_revision:
            self.publication_count += 1
        return super().replace_knowledge_configuration_head_if_match(expected=expected, replacement=replacement)


def _build_stack(bindings: _TenantBindingPort | None = None, mutation_ids: list[str] | None = None, counting: bool = False):
    store = InMemoryDocumentStore()
    repo = _CountingRepo(store) if counting else ManagedWorkspaceRepository(store)
    repo.put_workspace(_workspace())
    lookup = _FakeWorkspaceLookup({(_TENANT, _WORKSPACE): _workspace()})
    config_service = WorkspaceKnowledgeConfigurationService(repo, lookup)
    ids = mutation_ids or ["mutation-1", "mutation-2", "mutation-3", "mutation-4", "mutation-5"]
    idx = {"i": 0}
    def _next_id() -> str:
        v = ids[idx["i"]]
        idx["i"] = min(idx["i"] + 1, len(ids) - 1)
        return v
    engine = WorkspaceKnowledgeConfigurationMutationEngine(repo, lookup, config_service,
        {WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION: _ATT_HANDLER, WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION: _DET_HANDLER},
        clock=lambda: _NOW, mutation_id_factory=_next_id)
    attach = WorkspaceConnectionAttachmentService(connection_port=_FakeConnectionPort(), configuration_service=config_service, mutation_engine=engine)
    detach = WorkspaceConnectionDetachmentService(configuration_service=config_service, mutation_engine=engine,
        tenant_binding_port=bindings or _TenantBindingPort(), repository=repo)
    return attach, detach, repo, engine, config_service


def _attach(svc, rev=0):
    return svc.attach_connection(AttachWorkspaceConnectionCommand(tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION, expected_revision=rev, idempotency_key_hash=_SHA256)).configuration_revision


def _detach_cmd(**kw):
    p = dict(tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION, expected_revision=1, idempotency_key_hash=_SHA256_B)
    p.update(kw)
    return DetachWorkspaceConnectionCommand(**p)


def _idx_row(bid, *, status, ksb_ref="ksb-1", revision=1):
    return WorkspaceIndexedSourceBinding(indexed_source_binding_id=bid, tenant_id=_TENANT, workspace_id=_WORKSPACE,
        knowledge_source_binding_ref=ksb_ref, source_id=f"source-{bid}", status=status, mutation_id="seed-mutation",
        effective_revision=revision, semantic_identity_hash=_SHA256, created_at=_NOW, updated_at=_NOW)


def _live_row(bid, *, status, connection_ref=_CONNECTION, revision=1):
    return WorkspaceLiveAccessBinding(live_access_binding_id=bid, tenant_id=_TENANT, workspace_id=_WORKSPACE,
        connection_ref=connection_ref, allowed_capability_ids=("cap.read",), derived_provider_id="provider-1",
        derived_integration_kind=IntegrationCategory.WIKI_KNOWLEDGE, derived_safe_display_label="Wiki", status=status,
        mutation_id="seed-mutation", effective_revision=revision, semantic_identity_hash=_SHA256, created_at=_NOW, updated_at=_NOW)


def _pending_head(repo, *, revision, mutation_id):
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head
    repo.replace_knowledge_configuration_head_if_match(expected=head, replacement=head.model_copy(update={"pending_revision": revision, "pending_mutation_id": mutation_id, "updated_at": _NOW}))


def _detach_mutation(repo, *, revision, manifest_ids, mutation_id="mutation-detach", idem=_SHA256_B, req_hash=None, sem_hash=None, manifest_hash=None):
    intent = DetachConnectionMutationIntent(attachment_id=_ATTACHMENT_ID, connection_ref=_CONNECTION,
        indexed_source_binding_ids=manifest_ids.get("indexed", ()), live_access_binding_ids=manifest_ids.get("live", ()))
    req = req_hash or detach_connection_request_hash(tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION)
    sem = sem_hash or connection_attachment_semantic_identity_hash(tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION)
    mh = manifest_hash or detach_connection_stage_manifest_hash(attachment_id=intent.attachment_id, connection_ref=intent.connection_ref,
        indexed_source_binding_ids=intent.indexed_source_binding_ids, live_access_binding_ids=intent.live_access_binding_ids)
    m = WorkspaceKnowledgeMutationRecord(mutation_id=mutation_id, tenant_id=_TENANT, workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION, idempotency_key_hash=idem,
        normalized_request_hash=req, semantic_identity_hash=sem, stage_manifest_hash=mh, target_revision=revision,
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED, result_entity_type="connection_attachment",
        result_entity_id=_ATTACHMENT_ID, created_at=_NOW, updated_at=_NOW)
    repo.put_knowledge_configuration_mutation_if_absent(m)
    _pending_head(repo, revision=revision, mutation_id=mutation_id)
    return m, intent


def _owned_rows(repo, mutation_id, revision):
    def _filter(lst):
        return [r for r in lst if r.mutation_id == mutation_id and r.effective_revision == revision]
    return (_filter(repo.list_knowledge_connection_attachment_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)),
            _filter(repo.list_knowledge_indexed_source_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)),
            _filter(repo.list_knowledge_live_access_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)))


@pytest.mark.parametrize("status", [WorkspaceConnectionAttachmentStatusV1.ATTACHED, WorkspaceConnectionAttachmentStatusV1.UNAVAILABLE])
def test_detach_variants(status) -> None:
    attach, detach, repo, _, _ = _build_stack()
    rev = _attach(attach)
    if status is WorkspaceConnectionAttachmentStatusV1.UNAVAILABLE:
        cfg = attach._configuration_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        prev = cfg.connection_attachments[0]
        repo.put_knowledge_connection_attachment_version_if_absent(WorkspaceConnectionAttachment(
            attachment_id=prev.attachment_id, tenant_id=prev.tenant_id, workspace_id=prev.workspace_id,
            connection_ref=prev.connection_ref, safe_display_label=prev.safe_display_label, status=status,
            mutation_id="mutation-unavailable", effective_revision=rev + 1, created_at=prev.created_at, updated_at=_NOW))
        head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        repo.replace_knowledge_configuration_head_if_match(expected=head, replacement=head.model_copy(update={"committed_revision": rev + 1, "updated_at": _NOW}))
        rev += 1
    result = detach.detach_connection(_detach_cmd(expected_revision=rev))
    assert result.attachment.status is WorkspaceConnectionAttachmentStatusV1.DETACHED


def test_existing_detached_and_committed_replay() -> None:
    attach, detach, _, _, _ = _build_stack()
    rev = _attach(attach)
    first = detach.detach_connection(_detach_cmd(expected_revision=rev))
    second = detach.detach_connection(_detach_cmd(expected_revision=first.configuration_revision, idempotency_key_hash="c" * 64))
    replay = detach.detach_connection(_detach_cmd(expected_revision=rev))
    assert second.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay.configuration_revision == first.configuration_revision


def test_missing_attachment() -> None:
    _, detach, repo, _, _ = _build_stack()
    with pytest.raises(WorkspaceConnectionAttachmentError, match="connection_attachment_not_found"):
        detach.detach_connection(_detach_cmd(expected_revision=0))
    assert not repo.list_knowledge_configuration_mutations(tenant_id=_TENANT, workspace_id=_WORKSPACE)


def _malicious_binding(**attrs):
    base = {"tenant_id": _TENANT, "binding_id": "ksb-1", "connection_ref": _CONNECTION}
    base.update(attrs)
    return type("Binding", (), base)()


@pytest.mark.parametrize("port", [
    _TenantBindingPort(bindings={}), _TenantBindingPort(fail=True),
    _TenantBindingPort(bindings={"ksb-1": _tenant_binding(tenant_id="other-tenant")}),
    _TenantBindingPort(bindings={"ksb-1": _tenant_binding(binding_id="other-id")}),
    _TenantBindingPort(bindings={"ksb-1": _malicious_binding(connection_ref="  ")}),
    _TenantBindingPort(bindings={"ksb-1": _malicious_binding(tenant_id=None)}),
    _TenantBindingPort(bindings={"ksb-1": _malicious_binding(binding_id=object())}),
    _TenantBindingPort(bindings={"ksb-1": _malicious_binding(connection_ref=object())}),
    _TenantBindingPort(bindings={"ksb-1": _malicious_binding(connection_ref=type("X", (), {"strip": lambda self: (_ for _ in ()).throw(RuntimeError("boom"))})())}),
])
def test_tenant_binding_resolution_failure(port) -> None:
    attach, detach, repo, _, _ = _build_stack(bindings=port)
    rev = _attach(attach)
    repo.put_knowledge_indexed_source_version_if_absent(_idx_row("idx-active", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE))
    with pytest.raises(WorkspaceConnectionAttachmentError, match="connection_detach_dependency_resolution_failed"):
        detach.detach_connection(_detach_cmd(expected_revision=rev))
    assert not any(m.operation is WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION for m in repo.list_knowledge_configuration_mutations(tenant_id=_TENANT, workspace_id=_WORKSPACE))


def test_indexed_and_live_cascade() -> None:
    bindings = _TenantBindingPort({"ksb-primary": _tenant_binding(binding_id="ksb-primary"), "ksb-other": _tenant_binding(binding_id="ksb-other", connection_ref=_CONNECTION_OTHER)})
    attach, detach, repo, _, cfg_svc = _build_stack(bindings=bindings)
    rev = _attach(attach)
    for row in [_idx_row("idx-active", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, ksb_ref="ksb-primary"),
                _idx_row("idx-error", status=WorkspaceIndexedSourceBindingStatusV1.ERROR, ksb_ref="ksb-primary"),
                _idx_row("idx-disabled", status=WorkspaceIndexedSourceBindingStatusV1.DISABLED, ksb_ref="ksb-primary"),
                _idx_row("idx-other", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, ksb_ref="ksb-other"),
                _live_row("live-active", status=LiveAccessBindingStatusV1.ACTIVE),
                _live_row("live-other", status=LiveAccessBindingStatusV1.ACTIVE, connection_ref=_CONNECTION_OTHER)]:
        (repo.put_knowledge_indexed_source_version_if_absent if hasattr(row, "indexed_source_binding_id") else repo.put_knowledge_live_access_version_if_absent)(row)
    result = detach.detach_connection(_detach_cmd(expected_revision=rev))
    cfg = cfg_svc.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    idx = {b.indexed_source_binding_id: b.status for b in cfg.indexed_sources}
    live = {b.live_access_binding_id: b.status for b in cfg.live_access_bindings}
    assert idx["idx-active"] is WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE
    assert idx["idx-error"] is WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE
    assert idx["idx-disabled"] is WorkspaceIndexedSourceBindingStatusV1.DISABLED
    assert idx["idx-other"] is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    assert live["live-active"] is LiveAccessBindingStatusV1.UNAVAILABLE
    assert result.configuration_revision == rev + 1


def test_non_destructive_detach() -> None:
    bindings = _TenantBindingPort({"ksb-primary": _tenant_binding(binding_id="ksb-primary")})
    attach, detach, repo, _, cfg_svc = _build_stack(bindings=bindings)
    rev = _attach(attach)
    repo.put_source_if_absent(WorkspaceSource(source_id="source-idx-active", tenant_id=_TENANT, workspace_id=_WORKSPACE,
        source_type=WorkspaceSourceType.CONNECTED_SOURCE, status=WorkspaceSourceStatus.REGISTERED, created_at=_NOW,
        knowledge_configuration_creation_mutation_id="seed-mutation", knowledge_configuration_visibility_revision=rev))
    repo.put_knowledge_indexed_source_version_if_absent(_idx_row("idx-active", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, ksb_ref="ksb-primary"))
    policy = cfg_svc.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE).query_policy
    detach.detach_connection(_detach_cmd(expected_revision=rev))
    cfg = cfg_svc.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert cfg.query_policy == policy
    assert repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id="source-idx-active")


def test_stage_manifest_hash_and_blank_rejection() -> None:
    a = detach_connection_stage_manifest_hash(attachment_id=_ATTACHMENT_ID, connection_ref=_CONNECTION,
        indexed_source_binding_ids=("idx-b", "idx-a"), live_access_binding_ids=("live-z", "live-a"))
    b = detach_connection_stage_manifest_hash(attachment_id=_ATTACHMENT_ID, connection_ref=_CONNECTION,
        indexed_source_binding_ids=("idx-a", "idx-b"), live_access_binding_ids=("live-a", "live-z"))
    assert a == b
    with pytest.raises(ValueError, match="binding_id_blank"):
        DetachConnectionMutationIntent(attachment_id=_ATTACHMENT_ID, connection_ref=_CONNECTION, indexed_source_binding_ids=(" ",), live_access_binding_ids=())


def test_replay_later_dependency_and_newer_attachment() -> None:
    bindings = _TenantBindingPort({"ksb-primary": _tenant_binding(binding_id="ksb-primary")})
    attach, detach, repo, _, _ = _build_stack(bindings=bindings)
    rev = _attach(attach)
    first = detach.detach_connection(_detach_cmd(expected_revision=rev))
    repo.put_knowledge_indexed_source_version_if_absent(_idx_row("idx-new", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, ksb_ref="ksb-primary", revision=first.configuration_revision + 1))
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    repo.replace_knowledge_configuration_head_if_match(expected=head, replacement=head.model_copy(update={"committed_revision": first.configuration_revision + 1, "updated_at": _NOW}))
    bindings._fail = True
    replay = detach.detach_connection(_detach_cmd(expected_revision=rev))
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay.configuration_revision == first.configuration_revision
    assert replay.attachment.effective_revision == first.configuration_revision
    assert bindings.call_count == 0
    cfg = attach._configuration_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    prev = next(a for a in cfg.connection_attachments if a.attachment_id == _ATTACHMENT_ID)
    repo.put_knowledge_connection_attachment_version_if_absent(WorkspaceConnectionAttachment(
        attachment_id=prev.attachment_id, tenant_id=prev.tenant_id, workspace_id=prev.workspace_id,
        connection_ref=prev.connection_ref, safe_display_label=prev.safe_display_label,
        status=WorkspaceConnectionAttachmentStatusV1.ATTACHED, mutation_id="mutation-reattach",
        effective_revision=first.configuration_revision + 2, created_at=prev.created_at, updated_at=_NOW))
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    repo.replace_knowledge_configuration_head_if_match(expected=head, replacement=head.model_copy(update={"committed_revision": first.configuration_revision + 2, "updated_at": _NOW}))
    replay2 = detach.detach_connection(_detach_cmd(expected_revision=rev))
    assert replay2.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay2.attachment.status is WorkspaceConnectionAttachmentStatusV1.DETACHED
    assert replay2.attachment.effective_revision == first.configuration_revision


def test_existing_result_replay_after_later_revision() -> None:
    attach, detach, repo, _, _ = _build_stack()
    rev = _attach(attach)
    detach.detach_connection(_detach_cmd(expected_revision=rev))
    noop = detach.detach_connection(_detach_cmd(expected_revision=rev + 1, idempotency_key_hash="c" * 64))
    assert noop.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT
    repo.put_knowledge_indexed_source_version_if_absent(_idx_row("idx-extra", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, revision=noop.configuration_revision + 1))
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    repo.replace_knowledge_configuration_head_if_match(expected=head, replacement=head.model_copy(update={"committed_revision": noop.configuration_revision + 1, "updated_at": _NOW}))
    replay = detach.detach_connection(_detach_cmd(expected_revision=rev, idempotency_key_hash="c" * 64))
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay.configuration_revision == noop.configuration_revision


def test_idempotency_conflict_before_deps() -> None:
    attach, detach, repo, _, _ = _build_stack()
    rev = _attach(attach)
    detach.detach_connection(_detach_cmd(expected_revision=rev))
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError, match="configuration_idempotency_conflict"):
        detach.detach_connection(_detach_cmd(expected_revision=rev, connection_ref=_CONNECTION_OTHER))


@pytest.mark.parametrize("field,wrong", [
    ("normalized_request_hash", "d" * 64), ("semantic_identity_hash", "e" * 64),
    ("stage_manifest_hash", "f" * 64), ("attachment_id", "wrong-id"),
])
def test_stage_identity_blocks_writes(field, wrong) -> None:
    attach, _, repo, _, _ = _build_stack(mutation_ids=["mutation-attach", "mutation-detach"])
    rev = _attach(attach)
    aid = wrong if field == "attachment_id" else _ATTACHMENT_ID
    kw = {}
    if field == "normalized_request_hash":
        kw["req_hash"] = wrong
    elif field == "semantic_identity_hash":
        kw["sem_hash"] = wrong
    elif field == "stage_manifest_hash":
        kw["manifest_hash"] = wrong
    mutation, _ = _detach_mutation(repo, revision=rev + 1, manifest_ids={}, **kw)
    intent = DetachConnectionMutationIntent(attachment_id=aid, connection_ref=_CONNECTION, indexed_source_binding_ids=(), live_access_binding_ids=())
    with pytest.raises(RuntimeError):
        _DET_HANDLER.stage(repository=repo, mutation=mutation, target_revision=rev + 1, intent=intent, now=_NOW)
    assert not any(_owned_rows(repo, "mutation-detach", rev + 1))


@pytest.mark.parametrize("field,state", [
    ("normalized_request_hash", WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT),
    ("semantic_identity_hash", WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT),
    ("manifest", WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED),
])
def test_recovery_identity(field, state) -> None:
    attach, _, repo, _, _ = _build_stack(mutation_ids=["mutation-attach", "mutation-detach"])
    rev = _attach(attach)
    kw = {}
    if field == "normalized_request_hash":
        kw["req_hash"] = "d" * 64
    elif field == "semantic_identity_hash":
        kw["sem_hash"] = "e" * 64
    mutation, intent = _detach_mutation(repo, revision=rev + 1, manifest_ids={"indexed": ("idx-missing",)}, **kw)
    cfg = attach._configuration_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    prev = cfg.connection_attachments[0]
    repo.put_knowledge_connection_attachment_version_if_absent(WorkspaceConnectionAttachment(
        attachment_id=prev.attachment_id, tenant_id=prev.tenant_id, workspace_id=prev.workspace_id,
        connection_ref=prev.connection_ref, safe_display_label=prev.safe_display_label,
        status=WorkspaceConnectionAttachmentStatusV1.DETACHED, mutation_id="mutation-detach",
        effective_revision=rev + 1, created_at=prev.created_at, updated_at=_NOW))
    assert _DET_HANDLER.inspect_staged(repository=repo, mutation=mutation).state is state


def test_complete_multi_record_recovery() -> None:
    bindings = _TenantBindingPort({"ksb-primary": _tenant_binding(binding_id="ksb-primary")})
    attach, _, repo, engine, cfg_svc = _build_stack(bindings=bindings, mutation_ids=["mutation-attach", "mutation-detach"], counting=True)
    rev = _attach(attach)
    cfg = cfg_svc.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    prev_att = cfg.connection_attachments[0]
    idx = _idx_row("idx-active", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, ksb_ref="ksb-primary", revision=rev)
    live = _live_row("live-active", status=LiveAccessBindingStatusV1.ACTIVE, revision=rev)
    repo.put_knowledge_indexed_source_version_if_absent(idx)
    repo.put_knowledge_live_access_version_if_absent(live)
    mutation, intent = _detach_mutation(repo, revision=rev + 1, manifest_ids={"indexed": ("idx-active",), "live": ("live-active",)})
    _DET_HANDLER.stage(repository=repo, mutation=mutation, target_revision=rev + 1, intent=intent, now=_NOW)
    assert _DET_HANDLER.inspect_staged(repository=repo, mutation=mutation).state is WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
    repo.publication_count = 0
    recovery = engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    assert recovery.mutation.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head.committed_revision == rev + 1 and head.pending_mutation_id is None
    cfg = cfg_svc.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert cfg.connection_attachments[0].status is WorkspaceConnectionAttachmentStatusV1.DETACHED
    assert {b.indexed_source_binding_id: b.status for b in cfg.indexed_sources}["idx-active"] is WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE
    assert {b.live_access_binding_id: b.status for b in cfg.live_access_bindings}["live-active"] is LiveAccessBindingStatusV1.UNAVAILABLE
    assert prev_att == repo.get_knowledge_connection_attachment_version(tenant_id=_TENANT, workspace_id=_WORKSPACE, attachment_id=prev_att.attachment_id, effective_revision=rev)
    assert repo.publication_count == 1


def test_partial_recovery_via_engine() -> None:
    bindings = _TenantBindingPort({"ksb-primary": _tenant_binding(binding_id="ksb-primary")})
    attach, _, repo, engine, cfg_svc = _build_stack(bindings=bindings, mutation_ids=["mutation-attach", "mutation-detach"])
    rev = _attach(attach)
    repo.put_knowledge_indexed_source_version_if_absent(_idx_row("idx-a", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, ksb_ref="ksb-primary", revision=rev))
    repo.put_knowledge_indexed_source_version_if_absent(_idx_row("idx-b", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, ksb_ref="ksb-primary", revision=rev))
    repo.put_knowledge_live_access_version_if_absent(_live_row("live-a", status=LiveAccessBindingStatusV1.ACTIVE, revision=rev))
    cfg_before = cfg_svc.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    mutation, intent = _detach_mutation(repo, revision=rev + 1, manifest_ids={"indexed": ("idx-a", "idx-b"), "live": ("live-a",)})
    prev = cfg_before.connection_attachments[0]
    repo.put_knowledge_connection_attachment_version_if_absent(WorkspaceConnectionAttachment(
        attachment_id=prev.attachment_id, tenant_id=prev.tenant_id, workspace_id=prev.workspace_id,
        connection_ref=prev.connection_ref, safe_display_label=prev.safe_display_label,
        status=WorkspaceConnectionAttachmentStatusV1.DETACHED, mutation_id="mutation-detach",
        effective_revision=rev + 1, created_at=prev.created_at, updated_at=_NOW))
    repo.put_knowledge_indexed_source_version_if_absent(_idx_row("idx-a", status=WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE, ksb_ref="ksb-primary", revision=rev + 1).model_copy(update={"mutation_id": "mutation-detach"}))
    assert _DET_HANDLER.inspect_staged(repository=repo, mutation=mutation).state is WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    recovery = engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    assert recovery.mutation.status is WorkspaceKnowledgeMutationStatusV1.ABORTED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head.committed_revision == rev and head.pending_mutation_id is None
    assert not any(_owned_rows(repo, "mutation-detach", rev + 1))
    assert cfg_svc.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE) == cfg_before


def test_ownership_conflict_blocks_cleanup() -> None:
    attach, _, repo, _, _ = _build_stack()
    rev = _attach(attach)
    mutation = WorkspaceKnowledgeMutationRecord(mutation_id="mutation-detach", tenant_id=_TENANT, workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION, idempotency_key_hash=_SHA256_B,
        normalized_request_hash="d" * 64, semantic_identity_hash="e" * 64, stage_manifest_hash="f" * 64,
        target_revision=rev + 1, status=WorkspaceKnowledgeMutationStatusV1.PREPARED, created_at=_NOW, updated_at=_NOW)
    repo.put_knowledge_connection_attachment_version_if_absent(WorkspaceConnectionAttachment(
        attachment_id=_ATTACHMENT_ID, tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION,
        safe_display_label="Tampered", status=WorkspaceConnectionAttachmentStatusV1.DETACHED,
        mutation_id="mutation-detach", effective_revision=rev + 1, created_at=_NOW, updated_at=_NOW))
    inspection = _DET_HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    assert not _DET_HANDLER.cleanup_staged(repository=repo, mutation=mutation, inspection=inspection)
