# © Artur Czarnecki. All rights reserved.

"""Workspace Knowledge Configuration mutation handlers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from local_workspace_application.workspaces.connected_source_ids import (
    connected_source_id,
    connected_source_id_from_semantic_hash,
    indexed_source_binding_id,
    indexed_source_binding_id_from_semantic_hash,
    workspace_indexed_source_semantic_hash,
)
from local_workspace_application.workspaces.connected_source_source_projection import (
    ConnectedSourceOriginValidationError,
    validate_connected_source_durable_origin,
)
from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    normalize_create_indexed_source_request_hash,
    normalize_disable_indexed_source_request_hash,
    semantic_identity_hash_for_create_indexed_source,
    semantic_identity_hash_for_disable_indexed_source,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeExistingResult,
    WorkspaceKnowledgeStageInspection,
    WorkspaceKnowledgeStageStateV1,
    WorkspaceKnowledgeStagedResult,
)
from local_workspace_application.workspaces.models import (
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_RESULT_ENTITY_TYPE = "indexed_source_binding"
_CONNECTION_ATTACHMENT_RESULT_TYPE = "connection_attachment"
_REACTIVATION_PREDECESSOR_STATUSES = frozenset(
    {
        WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
        WorkspaceIndexedSourceBindingStatusV1.DISABLED,
        WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE,
        WorkspaceIndexedSourceBindingStatusV1.ERROR,
    }
)
_DISABLE_PREDECESSOR_STATUSES = frozenset(
    {
        WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
        WorkspaceIndexedSourceBindingStatusV1.ERROR,
        WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE,
    }
)


def _canonical_json(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def connection_attachment_id(
    *,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
) -> str:
    payload = _canonical_json(
        {
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "connection_ref": connection_ref.strip(),
        }
    )
    return f"wca:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"


def connection_attachment_semantic_identity_hash(
    *,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
) -> str:
    payload = _canonical_json(
        {
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "connection_ref": connection_ref.strip(),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def connection_attachment_request_hash(
    *,
    connection_ref: str,
    safe_display_label: str,
) -> str:
    payload = _canonical_json(
        {
            "connection_ref": connection_ref.strip(),
            "safe_display_label": safe_display_label,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AttachConnectionMutationIntent:
    attachment_id: str
    connection_ref: str
    safe_display_label: str


class AttachConnectionMutationHandler:
    operation = WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION

    def find_existing_result(
        self,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        intent: object,
    ) -> WorkspaceKnowledgeExistingResult | None:
        if not isinstance(intent, AttachConnectionMutationIntent):
            raise ValueError("attach_connection_intent_required")
        connection_ref = intent.connection_ref.strip()
        for attachment in configuration.connection_attachments:
            if attachment.connection_ref != connection_ref:
                continue
            if attachment.status is not WorkspaceConnectionAttachmentStatusV1.ATTACHED:
                continue
            return WorkspaceKnowledgeExistingResult(
                result_entity_type=_CONNECTION_ATTACHMENT_RESULT_TYPE,
                result_entity_id=attachment.attachment_id,
            )
        return None

    def stage(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
        target_revision: int,
        intent: object,
        now: datetime,
    ) -> WorkspaceKnowledgeStagedResult:
        if not isinstance(intent, AttachConnectionMutationIntent):
            raise ValueError("attach_connection_intent_required")
        connection_ref = intent.connection_ref.strip()
        existing_versions = repository.list_knowledge_connection_attachment_versions(
            tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id,
        )
        earliest_created_at = now
        for version in existing_versions:
            if version.attachment_id != intent.attachment_id:
                continue
            if version.tenant_id != mutation.tenant_id:
                raise RuntimeError("connection_attachment_identity_conflict")
            if version.workspace_id != mutation.workspace_id:
                raise RuntimeError("connection_attachment_identity_conflict")
            if version.connection_ref != connection_ref:
                raise RuntimeError("connection_attachment_identity_conflict")
            if version.created_at < earliest_created_at:
                earliest_created_at = version.created_at

        attachment = WorkspaceConnectionAttachment(
            attachment_id=intent.attachment_id,
            tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id,
            connection_ref=connection_ref,
            safe_display_label=intent.safe_display_label,
            status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
            mutation_id=mutation.mutation_id,
            effective_revision=target_revision,
            created_at=earliest_created_at,
            updated_at=now,
        )
        if not repository.put_knowledge_connection_attachment_version_if_absent(attachment):
            existing = repository.get_knowledge_connection_attachment_version(
                tenant_id=mutation.tenant_id,
                workspace_id=mutation.workspace_id,
                attachment_id=attachment.attachment_id,
                effective_revision=target_revision,
            )
            if existing is None or not _is_equivalent_connection_attachment(
                existing,
                expected=attachment,
            ):
                raise RuntimeError("connection_attachment_stage_conflict")
        return WorkspaceKnowledgeStagedResult(
            result_entity_type=_CONNECTION_ATTACHMENT_RESULT_TYPE,
            result_entity_id=intent.attachment_id,
        )

    def inspect_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
    ) -> WorkspaceKnowledgeStageInspection:
        versions = repository.list_knowledge_connection_attachment_versions(
            tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id,
        )
        owned = [version for version in versions if version.mutation_id == mutation.mutation_id]
        if not owned:
            return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.ABSENT)
        if len(owned) != 1:
            return WorkspaceKnowledgeStageInspection(
                state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT,
            )
        return _inspect_owned_connection_attachment(mutation=mutation, staged=owned[0])

    def cleanup_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
        inspection: WorkspaceKnowledgeStageInspection,
    ) -> bool:
        if inspection.state is WorkspaceKnowledgeStageStateV1.ABSENT:
            return True
        if inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            return False
        if inspection.state is WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED:
            return False
        if inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        versions = repository.list_knowledge_connection_attachment_versions(
            tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id,
        )
        owned = [version for version in versions if version.mutation_id == mutation.mutation_id]
        if len(owned) != 1:
            return False
        staged = owned[0]
        reloaded_inspection = _inspect_owned_connection_attachment(
            mutation=mutation,
            staged=staged,
        )
        if reloaded_inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        if inspection.result_entity_type != _CONNECTION_ATTACHMENT_RESULT_TYPE:
            return False
        if inspection.result_entity_id != staged.attachment_id:
            return False
        if not repository.delete_knowledge_connection_attachment_version_if_match(staged):
            return False
        still_present = repository.get_knowledge_connection_attachment_version(
            tenant_id=staged.tenant_id,
            workspace_id=staged.workspace_id,
            attachment_id=staged.attachment_id,
            effective_revision=staged.effective_revision,
        )
        return still_present is None


def _inspect_owned_connection_attachment(
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    staged: WorkspaceConnectionAttachment,
) -> WorkspaceKnowledgeStageInspection:
    conflict = WorkspaceKnowledgeStageInspection(
        state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT,
    )
    if mutation.target_revision is None:
        return conflict
    if staged.mutation_id != mutation.mutation_id:
        return conflict
    if staged.tenant_id != mutation.tenant_id:
        return conflict
    if staged.workspace_id != mutation.workspace_id:
        return conflict
    if staged.effective_revision != mutation.target_revision:
        return conflict
    if staged.status is not WorkspaceConnectionAttachmentStatusV1.ATTACHED:
        return conflict
    expected_attachment_id = connection_attachment_id(
        tenant_id=staged.tenant_id,
        workspace_id=staged.workspace_id,
        connection_ref=staged.connection_ref,
    )
    if staged.attachment_id != expected_attachment_id:
        return conflict
    if mutation.semantic_identity_hash is None:
        return conflict
    expected_semantic = connection_attachment_semantic_identity_hash(
        tenant_id=staged.tenant_id,
        workspace_id=staged.workspace_id,
        connection_ref=staged.connection_ref,
    )
    if mutation.semantic_identity_hash != expected_semantic:
        return conflict
    expected_request = connection_attachment_request_hash(
        connection_ref=staged.connection_ref,
        safe_display_label=staged.safe_display_label,
    )
    if mutation.normalized_request_hash != expected_request:
        return conflict
    if mutation.result_entity_type is not None:
        if mutation.result_entity_type != _CONNECTION_ATTACHMENT_RESULT_TYPE:
            return conflict
    if mutation.result_entity_id is not None:
        if mutation.result_entity_id != staged.attachment_id:
            return conflict
    return WorkspaceKnowledgeStageInspection(
        state=WorkspaceKnowledgeStageStateV1.COMPLETE_VALID,
        result_entity_type=_CONNECTION_ATTACHMENT_RESULT_TYPE,
        result_entity_id=staged.attachment_id,
    )


def _is_equivalent_connection_attachment(
    actual: WorkspaceConnectionAttachment,
    *,
    expected: WorkspaceConnectionAttachment,
) -> bool:
    return (
        actual.attachment_id == expected.attachment_id
        and actual.tenant_id == expected.tenant_id
        and actual.workspace_id == expected.workspace_id
        and actual.connection_ref == expected.connection_ref
        and actual.safe_display_label == expected.safe_display_label
        and actual.status == expected.status
        and actual.mutation_id == expected.mutation_id
        and actual.effective_revision == expected.effective_revision
        and actual.created_at == expected.created_at
        and actual.updated_at == expected.updated_at
    )


@dataclass(frozen=True, slots=True)
class CreateIndexedSourceMutationIntent:
    knowledge_source_binding_ref: str
    sync_mode: IndexedSourceSyncModeV1 = IndexedSourceSyncModeV1.FULL
    audience_eligibility: IndexedSourceAudienceEligibilityV1 = (
        IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY
    )
    cached_safe_display_label: str | None = None

def _stage_conflict() -> WorkspaceKnowledgeStageInspection:
    return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT)

def _stage_valid(binding_id: str) -> WorkspaceKnowledgeStageInspection:
    return WorkspaceKnowledgeStageInspection(
        state=WorkspaceKnowledgeStageStateV1.COMPLETE_VALID,
        result_entity_type=_RESULT_ENTITY_TYPE,
        result_entity_id=binding_id,
    )

def _owned_indexed_bindings(
    repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
) -> list[WorkspaceIndexedSourceBinding]:
    try:
        all_bindings = repository.list_knowledge_indexed_source_versions(
            tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
        )
    except ValueError as exc:
        raise RuntimeError("indexed_source_list_invalid") from exc
    return [v for v in all_bindings if v.mutation_id == mutation.mutation_id]

def _owned_connected_sources(
    repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
) -> list[WorkspaceSource]:
    try:
        all_sources = repository.list_sources(tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id)
    except ValidationError as exc:
        raise RuntimeError("connected_source_list_invalid") from exc
    return [s for s in all_sources if s.knowledge_configuration_creation_mutation_id == mutation.mutation_id]

def _select_latest_indexed_predecessor(
    repository: ManagedWorkspaceRepository,
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    binding_id: str,
    base_revision: int,
) -> WorkspaceIndexedSourceBinding | None:
    versions = repository.list_knowledge_indexed_source_versions(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
    )
    matches = [
        version
        for version in versions
        if version.indexed_source_binding_id == binding_id
        and version.effective_revision <= base_revision
    ]
    if not matches:
        return None
    top_revision = max(version.effective_revision for version in matches)
    top_matches = [
        version for version in matches if version.effective_revision == top_revision
    ]
    if len(top_matches) > 1:
        raise RuntimeError("indexed_source_predecessor_ambiguous")
    return top_matches[0]

def _create_indexed_source_identity(
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    target_revision: int,
    intent: CreateIndexedSourceMutationIntent,
) -> tuple[str, str, str]:
    if mutation.operation is not WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE:
        raise RuntimeError("create_indexed_source_operation_required")
    if mutation.target_revision != target_revision:
        raise RuntimeError("create_indexed_source_target_revision_mismatch")
    binding_ref = intent.knowledge_source_binding_ref.strip()
    expected_request = normalize_create_indexed_source_request_hash(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        knowledge_source_binding_ref=binding_ref,
        sync_mode=intent.sync_mode,
        audience_eligibility=intent.audience_eligibility,
    )
    if mutation.normalized_request_hash != expected_request:
        raise RuntimeError("create_indexed_source_request_hash_mismatch")
    expected_semantic = semantic_identity_hash_for_create_indexed_source(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        knowledge_source_binding_ref=binding_ref,
    )
    if mutation.semantic_identity_hash != expected_semantic:
        raise RuntimeError("create_indexed_source_semantic_hash_mismatch")
    binding_id = indexed_source_binding_id(mutation.tenant_id, mutation.workspace_id, binding_ref)
    source_id = connected_source_id(mutation.tenant_id, mutation.workspace_id, binding_ref)
    if mutation.result_entity_type is not None and mutation.result_entity_type != _RESULT_ENTITY_TYPE:
        raise RuntimeError("create_indexed_source_result_type_mismatch")
    if mutation.result_entity_id is not None and mutation.result_entity_id != binding_id:
        raise RuntimeError("create_indexed_source_result_id_mismatch")
    return binding_id, source_id, expected_semantic

def _active_binding(
    *, binding_id: str, source_id: str, semantic_hash: str,
    mutation: WorkspaceKnowledgeMutationRecord, intent: CreateIndexedSourceMutationIntent,
    target_revision: int, now: datetime, created_at: datetime,
) -> WorkspaceIndexedSourceBinding:
    binding_ref = intent.knowledge_source_binding_ref.strip()
    return WorkspaceIndexedSourceBinding(
        indexed_source_binding_id=binding_id, tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id, knowledge_source_binding_ref=binding_ref,
        source_id=source_id, sync_mode=intent.sync_mode,
        status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
        audience_eligibility=intent.audience_eligibility, mutation_id=mutation.mutation_id,
        effective_revision=target_revision, semantic_identity_hash=semantic_hash,
        created_at=created_at, updated_at=now,
        cached_safe_display_label=intent.cached_safe_display_label,
    )

def _connected_source(
    *, source_id: str, mutation: WorkspaceKnowledgeMutationRecord, target_revision: int,
    created_at: datetime, status: WorkspaceSourceStatus = WorkspaceSourceStatus.REGISTERED,
) -> WorkspaceSource:
    return WorkspaceSource(
        source_id=source_id, workspace_id=mutation.workspace_id, tenant_id=mutation.tenant_id,
        source_type=WorkspaceSourceType.CONNECTED_SOURCE, path="", recursive=False, status=status,
        created_at=created_at, knowledge_configuration_creation_mutation_id=mutation.mutation_id,
        knowledge_configuration_visibility_revision=target_revision,
    )

def _is_compatible_connected_source(actual: WorkspaceSource, *, expected: WorkspaceSource) -> bool:
    return (
        actual.tenant_id == expected.tenant_id and actual.workspace_id == expected.workspace_id
        and actual.source_id == expected.source_id
        and actual.source_type is WorkspaceSourceType.CONNECTED_SOURCE
        and actual.path == "" and actual.recursive is False
        and actual.knowledge_configuration_creation_mutation_id
        == expected.knowledge_configuration_creation_mutation_id
        and actual.knowledge_configuration_visibility_revision
        == expected.knowledge_configuration_visibility_revision
    )

def _put_connected_source_if_absent(
    repository: ManagedWorkspaceRepository, *, source: WorkspaceSource,
    mutation: WorkspaceKnowledgeMutationRecord,
) -> None:
    if repository.put_source_if_absent(source):
        return
    existing = repository.get_source(
        tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id, source_id=source.source_id,
    )
    if existing is None or not _is_compatible_connected_source(existing, expected=source):
        raise RuntimeError("connected_source_stage_conflict")

def _assert_reactivation_predecessor(
    predecessor: WorkspaceIndexedSourceBinding, *, binding_id: str, source_id: str,
    binding_ref: str, semantic_hash: str,
) -> None:
    if predecessor.indexed_source_binding_id != binding_id:
        raise RuntimeError("indexed_source_binding_identity_conflict")
    if predecessor.source_id != source_id:
        raise RuntimeError("indexed_source_source_identity_conflict")
    if predecessor.knowledge_source_binding_ref != binding_ref:
        raise RuntimeError("indexed_source_binding_ref_conflict")
    if predecessor.semantic_identity_hash != semantic_hash:
        raise RuntimeError("indexed_source_semantic_identity_conflict")
    if predecessor.status not in _REACTIVATION_PREDECESSOR_STATUSES:
        raise RuntimeError("indexed_source_predecessor_transition_invalid")

def _validate_origin_or_raise(
    repository: ManagedWorkspaceRepository, *, mutation: WorkspaceKnowledgeMutationRecord,
    source_id: str, binding: WorkspaceIndexedSourceBinding, base_revision: int,
) -> None:
    try:
        validate_connected_source_durable_origin(
            repository=repository, tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id, source_id=source_id, binding=binding,
            committed_configuration_revision=base_revision,
        )
    except ConnectedSourceOriginValidationError as exc:
        raise RuntimeError(f"indexed_source_origin_invalid:{exc}") from exc

def _put_binding_if_absent(
    repository: ManagedWorkspaceRepository, binding: WorkspaceIndexedSourceBinding,
) -> None:
    if not repository.put_knowledge_indexed_source_version_if_absent(binding):
        raise RuntimeError("indexed_source_binding_stage_conflict")

def _confirm_indexed_cleanup(
    repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
) -> bool:
    for version in repository.list_knowledge_indexed_source_versions(
        tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
    ):
        if version.mutation_id == mutation.mutation_id:
            return False
    for source in repository.list_sources(
        tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
    ):
        if source.knowledge_configuration_creation_mutation_id == mutation.mutation_id:
            return False
    return True

def _delete_owned_indexed_rows(
    repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord, *,
    bindings: list[WorkspaceIndexedSourceBinding], sources: list[WorkspaceSource],
) -> bool:
    rev = mutation.target_revision
    for binding in bindings:
        if rev is not None and binding.effective_revision != rev:
            return False
        if not repository.delete_knowledge_indexed_source_version_if_match(binding):
            return False
    for source in sources:
        if rev is not None and source.knowledge_configuration_visibility_revision != rev:
            return False
        if not repository.delete_source_if_match(source):
            return False
    return _confirm_indexed_cleanup(repository, mutation)

def _intent_from_binding(binding: WorkspaceIndexedSourceBinding) -> CreateIndexedSourceMutationIntent:
    return CreateIndexedSourceMutationIntent(
        knowledge_source_binding_ref=binding.knowledge_source_binding_ref,
        sync_mode=binding.sync_mode,
        audience_eligibility=binding.audience_eligibility,
        cached_safe_display_label=binding.cached_safe_display_label,
    )

def _semantic_ids_or_none(mutation: WorkspaceKnowledgeMutationRecord) -> tuple[str, str] | None:
    try:
        return (
            indexed_source_binding_id_from_semantic_hash(mutation.semantic_identity_hash),
            connected_source_id_from_semantic_hash(mutation.semantic_identity_hash),
        )
    except ValueError:
        return None

def _prove_source_only_owned_partial(
    repository: ManagedWorkspaceRepository, *, mutation: WorkspaceKnowledgeMutationRecord,
    owned_source: WorkspaceSource,
) -> bool:
    if mutation.operation is not WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE or mutation.target_revision is None:
        return False
    ids = _semantic_ids_or_none(mutation)
    if ids is None:
        return False
    binding_id, source_id = ids
    if owned_source.source_id != source_id:
        return False
    if mutation.result_entity_type is not None and mutation.result_entity_type != _RESULT_ENTITY_TYPE:
        return False
    if mutation.result_entity_id is not None and mutation.result_entity_id != binding_id:
        return False
    expected = _connected_source(
        source_id=source_id, mutation=mutation, target_revision=mutation.target_revision,
        created_at=owned_source.created_at, status=WorkspaceSourceStatus.REGISTERED,
    )
    if owned_source != expected or owned_source.last_sync_at is not None:
        return False
    prior = repository.get_source(tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id, source_id=source_id)
    if prior is not None and prior.knowledge_configuration_creation_mutation_id != mutation.mutation_id:
        return False
    return True

def _prove_binding_only_owned_partial(
    repository: ManagedWorkspaceRepository, *, mutation: WorkspaceKnowledgeMutationRecord,
    owned_binding: WorkspaceIndexedSourceBinding,
) -> bool:
    if mutation.operation is not WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE or mutation.target_revision is None:
        return False
    ids = _semantic_ids_or_none(mutation)
    if ids is None:
        return False
    binding_id, source_id = ids
    if (
        owned_binding.indexed_source_binding_id != binding_id or owned_binding.source_id != source_id
        or owned_binding.tenant_id != mutation.tenant_id or owned_binding.workspace_id != mutation.workspace_id
        or owned_binding.status is not WorkspaceIndexedSourceBindingStatusV1.ACTIVE
        or owned_binding.effective_revision != mutation.target_revision
        or owned_binding.semantic_identity_hash != mutation.semantic_identity_hash
        or owned_binding.created_at != owned_binding.updated_at
        or not owned_binding.knowledge_source_binding_ref.strip()
    ):
        return False
    intent = _intent_from_binding(owned_binding)
    expected_request = normalize_create_indexed_source_request_hash(
        tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
        knowledge_source_binding_ref=intent.knowledge_source_binding_ref.strip(),
        sync_mode=intent.sync_mode, audience_eligibility=intent.audience_eligibility,
    )
    if mutation.normalized_request_hash != expected_request:
        return False
    if mutation.result_entity_type is not None and mutation.result_entity_type != _RESULT_ENTITY_TYPE:
        return False
    if mutation.result_entity_id is not None and mutation.result_entity_id != binding_id:
        return False
    if _select_latest_indexed_predecessor(
        repository, mutation=mutation, binding_id=binding_id, base_revision=mutation.target_revision - 1,
    ) is not None:
        return False
    return repository.get_source(
        tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id, source_id=source_id,
    ) is None

def _prove_initial_complete_stage(
    *, mutation: WorkspaceKnowledgeMutationRecord, owned_binding: WorkspaceIndexedSourceBinding,
    owned_source: WorkspaceSource, intent: CreateIndexedSourceMutationIntent,
    binding_id: str, source_id: str, semantic_hash: str, target_revision: int,
) -> bool:
    if mutation.operation is not WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE:
        return False
    if mutation.target_revision != target_revision:
        return False
    expected_request = normalize_create_indexed_source_request_hash(
        tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
        knowledge_source_binding_ref=intent.knowledge_source_binding_ref.strip(),
        sync_mode=intent.sync_mode, audience_eligibility=intent.audience_eligibility,
    )
    if mutation.normalized_request_hash != expected_request or mutation.semantic_identity_hash != semantic_hash:
        return False
    if mutation.result_entity_type is not None and mutation.result_entity_type != _RESULT_ENTITY_TYPE:
        return False
    if mutation.result_entity_id is not None and mutation.result_entity_id != binding_id:
        return False
    if owned_binding.status is not WorkspaceIndexedSourceBindingStatusV1.ACTIVE:
        return False
    expected_binding = _active_binding(
        binding_id=binding_id, source_id=source_id, semantic_hash=semantic_hash,
        mutation=mutation, intent=intent, target_revision=target_revision,
        now=owned_source.created_at, created_at=owned_source.created_at,
    )
    if owned_binding != expected_binding:
        return False
    expected_source = _connected_source(
        source_id=source_id, mutation=mutation, target_revision=target_revision,
        created_at=owned_source.created_at, status=owned_source.status,
    )
    return (
        _is_compatible_connected_source(owned_source, expected=expected_source)
        and owned_binding.source_id == owned_source.source_id
        and owned_binding.created_at == owned_source.created_at
    )

def _prove_incomplete_owned_partial(
    repository: ManagedWorkspaceRepository, *, mutation: WorkspaceKnowledgeMutationRecord,
    bindings: list[WorkspaceIndexedSourceBinding], sources: list[WorkspaceSource],
) -> bool:
    if bindings and sources:
        return False
    if bindings and not _prove_binding_only_owned_partial(repository, mutation=mutation, owned_binding=bindings[0]):
        return False
    if sources and not _prove_source_only_owned_partial(repository, mutation=mutation, owned_source=sources[0]):
        return False
    return True

def _inspect_create_indexed_staged(
    repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
) -> WorkspaceKnowledgeStageInspection:
    if mutation.target_revision is None:
        return _stage_conflict()
    try:
        owned_bindings = _owned_indexed_bindings(repository, mutation)
        owned_sources = _owned_connected_sources(repository, mutation)
    except RuntimeError:
        return _stage_conflict()
    if len(owned_bindings) > 1 or len(owned_sources) > 1:
        return _stage_conflict()
    owned_binding = owned_bindings[0] if owned_bindings else None
    owned_source = owned_sources[0] if owned_sources else None
    if owned_binding is None and owned_source is None:
        return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.ABSENT)
    if owned_binding is not None and owned_binding.effective_revision != mutation.target_revision:
        return _stage_conflict()
    if owned_source is not None and owned_source.knowledge_configuration_visibility_revision != mutation.target_revision:
        return _stage_conflict()
    if owned_binding is None:
        if _prove_source_only_owned_partial(repository, mutation=mutation, owned_source=owned_source):
            return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED)
        return _stage_conflict()
    try:
        binding_id, source_id, semantic_hash = _create_indexed_source_identity(
            mutation=mutation, target_revision=mutation.target_revision,
            intent=_intent_from_binding(owned_binding),
        )
    except RuntimeError:
        return _stage_conflict()
    if owned_binding.indexed_source_binding_id != binding_id or owned_binding.source_id != source_id:
        return _stage_conflict()
    base_revision = mutation.target_revision - 1
    try:
        predecessor = _select_latest_indexed_predecessor(
            repository, mutation=mutation, binding_id=binding_id, base_revision=base_revision,
        )
    except RuntimeError:
        return _stage_conflict()
    if predecessor is None:
        if owned_source is None:
            if _prove_binding_only_owned_partial(repository, mutation=mutation, owned_binding=owned_binding):
                return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED)
            return _stage_conflict()
        intent = _intent_from_binding(owned_binding)
        if not _prove_initial_complete_stage(
            mutation=mutation, owned_binding=owned_binding, owned_source=owned_source, intent=intent,
            binding_id=binding_id, source_id=source_id, semantic_hash=semantic_hash,
            target_revision=mutation.target_revision,
        ):
            return _stage_conflict()
        return _stage_valid(binding_id)
    if owned_source is not None:
        return _stage_conflict()
    intent = _intent_from_binding(owned_binding)
    try:
        _assert_reactivation_predecessor(
            predecessor, binding_id=binding_id, source_id=source_id,
            binding_ref=intent.knowledge_source_binding_ref.strip(), semantic_hash=semantic_hash,
        )
        validate_connected_source_durable_origin(
            repository=repository, tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
            source_id=source_id, binding=predecessor, committed_configuration_revision=base_revision,
        )
    except (RuntimeError, ConnectedSourceOriginValidationError):
        return _stage_conflict()
    expected = predecessor.model_copy(update={
        "sync_mode": intent.sync_mode, "audience_eligibility": intent.audience_eligibility,
        "cached_safe_display_label": intent.cached_safe_display_label,
        "status": WorkspaceIndexedSourceBindingStatusV1.ACTIVE, "mutation_id": mutation.mutation_id,
        "effective_revision": mutation.target_revision, "updated_at": owned_binding.updated_at,
    })
    if owned_binding != expected:
        return _stage_conflict()
    return _stage_valid(binding_id)

class CreateIndexedSourceMutationHandler:
    operation = WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE

    def find_existing_result(
        self, *, configuration: WorkspaceKnowledgeConfigurationV1, intent: object,
    ) -> WorkspaceKnowledgeExistingResult | None:
        if not isinstance(intent, CreateIndexedSourceMutationIntent):
            raise ValueError("create_indexed_source_intent_required")
        binding_ref = intent.knowledge_source_binding_ref.strip()
        semantic_hash = workspace_indexed_source_semantic_hash(
            configuration.tenant_id, configuration.workspace_id, binding_ref,
        )
        for binding in configuration.indexed_sources:
            if (
                binding.knowledge_source_binding_ref == binding_ref
                and binding.semantic_identity_hash == semantic_hash
                and binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
                and binding.sync_mode == intent.sync_mode
                and binding.audience_eligibility == intent.audience_eligibility
            ):
                return WorkspaceKnowledgeExistingResult(
                    result_entity_type=_RESULT_ENTITY_TYPE,
                    result_entity_id=binding.indexed_source_binding_id,
                )
        return None

    def stage(
        self, *, repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord, target_revision: int,
        intent: object, now: datetime,
    ) -> WorkspaceKnowledgeStagedResult:
        if not isinstance(intent, CreateIndexedSourceMutationIntent):
            raise ValueError("create_indexed_source_intent_required")
        binding_id, source_id, semantic_hash = _create_indexed_source_identity(
            mutation=mutation, target_revision=target_revision, intent=intent,
        )
        binding_ref = intent.knowledge_source_binding_ref.strip()
        base_revision = target_revision - 1
        predecessor = _select_latest_indexed_predecessor(
            repository, mutation=mutation, binding_id=binding_id, base_revision=base_revision,
        )
        if predecessor is None:
            _put_connected_source_if_absent(
                repository,
                source=_connected_source(
                    source_id=source_id, mutation=mutation, target_revision=target_revision,
                    created_at=now,
                ),
                mutation=mutation,
            )
        else:
            _assert_reactivation_predecessor(
                predecessor, binding_id=binding_id, source_id=source_id,
                binding_ref=binding_ref, semantic_hash=semantic_hash,
            )
            _validate_origin_or_raise(
                repository, mutation=mutation, source_id=source_id,
                binding=predecessor, base_revision=base_revision,
            )
        _put_binding_if_absent(
            repository,
            _active_binding(
                binding_id=binding_id, source_id=source_id, semantic_hash=semantic_hash,
                mutation=mutation, intent=intent, target_revision=target_revision,
                now=now, created_at=now if predecessor is None else predecessor.created_at,
            ),
        )
        return WorkspaceKnowledgeStagedResult(
            result_entity_type=_RESULT_ENTITY_TYPE, result_entity_id=binding_id,
        )

    def inspect_staged(
        self, *, repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
    ) -> WorkspaceKnowledgeStageInspection:
        return _inspect_create_indexed_staged(repository, mutation)

    def cleanup_staged(
        self, *, repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord, inspection: WorkspaceKnowledgeStageInspection,
    ) -> bool:
        if inspection.state is WorkspaceKnowledgeStageStateV1.ABSENT:
            return True
        if inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            return False
        current_inspection = _inspect_create_indexed_staged(repository, mutation)
        if current_inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            return False
        if inspection.state is WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED:
            if current_inspection.state is not WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED:
                return False
            bindings = _owned_indexed_bindings(repository, mutation)
            sources = _owned_connected_sources(repository, mutation)
            if not _prove_incomplete_owned_partial(
                repository, mutation=mutation, bindings=bindings, sources=sources,
            ):
                return False
            current_inspection = _inspect_create_indexed_staged(repository, mutation)
            if current_inspection.state is not WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED:
                return False
            bindings = _owned_indexed_bindings(repository, mutation)
            sources = _owned_connected_sources(repository, mutation)
            if not _prove_incomplete_owned_partial(repository, mutation=mutation, bindings=bindings, sources=sources):
                return False
            return _delete_owned_indexed_rows(repository, mutation, bindings=bindings, sources=sources)
        if current_inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        if inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        bindings = _owned_indexed_bindings(repository, mutation)
        sources = _owned_connected_sources(repository, mutation)
        if len(bindings) != 1:
            return False
        binding = bindings[0]
        if (
            mutation.target_revision is None or binding.effective_revision != mutation.target_revision
            or current_inspection.result_entity_id != binding.indexed_source_binding_id
        ):
            return False
        if not repository.delete_knowledge_indexed_source_version_if_match(binding):
            return False
        return _delete_owned_indexed_rows(repository, mutation, bindings=[], sources=sources)


@dataclass(frozen=True, slots=True)
class DisableIndexedSourceMutationIntent:
    indexed_source_binding_id: str
    knowledge_source_binding_ref: str

def _disable_indexed_source_identity(
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    target_revision: int,
    intent: DisableIndexedSourceMutationIntent,
) -> tuple[str, str]:
    if mutation.operation is not WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE:
        raise RuntimeError("disable_indexed_source_operation_required")
    if mutation.target_revision != target_revision:
        raise RuntimeError("disable_indexed_source_target_revision_mismatch")
    binding_id = intent.indexed_source_binding_id.strip()
    binding_ref = intent.knowledge_source_binding_ref.strip()
    expected_request = normalize_disable_indexed_source_request_hash(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        indexed_source_binding_id=binding_id,
    )
    if mutation.normalized_request_hash != expected_request:
        raise RuntimeError("disable_indexed_source_request_hash_mismatch")
    expected_semantic = semantic_identity_hash_for_disable_indexed_source(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        knowledge_source_binding_ref=binding_ref,
    )
    if mutation.semantic_identity_hash != expected_semantic:
        raise RuntimeError("disable_indexed_source_semantic_hash_mismatch")
    expected_binding_id = indexed_source_binding_id(
        mutation.tenant_id, mutation.workspace_id, binding_ref,
    )
    if binding_id != expected_binding_id:
        raise RuntimeError("disable_indexed_source_binding_id_mismatch")
    if mutation.result_entity_type is not None and mutation.result_entity_type != _RESULT_ENTITY_TYPE:
        raise RuntimeError("disable_indexed_source_result_type_mismatch")
    if mutation.result_entity_id is not None and mutation.result_entity_id != binding_id:
        raise RuntimeError("disable_indexed_source_result_id_mismatch")
    return binding_id, expected_semantic

def _inspect_disable_indexed_staged(
    repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
) -> WorkspaceKnowledgeStageInspection:
    owned_bindings = _owned_indexed_bindings(repository, mutation)
    if len(owned_bindings) > 1:
        return _stage_conflict()
    if mutation.target_revision is None:
        return _stage_conflict()
    if not owned_bindings:
        return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.ABSENT)
    staged = owned_bindings[0]
    if staged.effective_revision != mutation.target_revision:
        return _stage_conflict()
    try:
        binding_id, _ = _disable_indexed_source_identity(
            mutation=mutation, target_revision=mutation.target_revision,
            intent=DisableIndexedSourceMutationIntent(
                indexed_source_binding_id=staged.indexed_source_binding_id,
                knowledge_source_binding_ref=staged.knowledge_source_binding_ref,
            ),
        )
    except RuntimeError:
        return _stage_conflict()
    if staged.indexed_source_binding_id != binding_id:
        return _stage_conflict()
    base_revision = mutation.target_revision - 1
    try:
        predecessor = _select_latest_indexed_predecessor(
            repository,
            mutation=mutation,
            binding_id=binding_id,
            base_revision=base_revision,
        )
    except RuntimeError:
        return _stage_conflict()
    if predecessor is None or predecessor.status not in _DISABLE_PREDECESSOR_STATUSES:
        return _stage_conflict()
    expected = predecessor.model_copy(update={
        "status": WorkspaceIndexedSourceBindingStatusV1.DISABLED,
        "mutation_id": mutation.mutation_id,
        "effective_revision": mutation.target_revision,
        "updated_at": staged.updated_at,
    })
    if staged != expected:
        return _stage_conflict()
    return _stage_valid(binding_id)

class DisableIndexedSourceMutationHandler:
    operation = WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE

    def find_existing_result(
        self, *, configuration: WorkspaceKnowledgeConfigurationV1, intent: object,
    ) -> WorkspaceKnowledgeExistingResult | None:
        if not isinstance(intent, DisableIndexedSourceMutationIntent):
            raise ValueError("disable_indexed_source_intent_required")
        binding_id = intent.indexed_source_binding_id.strip()
        for binding in configuration.indexed_sources:
            if (
                binding.indexed_source_binding_id == binding_id
                and binding.status is WorkspaceIndexedSourceBindingStatusV1.DISABLED
            ):
                return WorkspaceKnowledgeExistingResult(
                    result_entity_type=_RESULT_ENTITY_TYPE, result_entity_id=binding_id,
                )
        return None

    def stage(
        self, *, repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord, target_revision: int,
        intent: object, now: datetime,
    ) -> WorkspaceKnowledgeStagedResult:
        if not isinstance(intent, DisableIndexedSourceMutationIntent):
            raise ValueError("disable_indexed_source_intent_required")
        binding_id, _ = _disable_indexed_source_identity(
            mutation=mutation, target_revision=target_revision, intent=intent,
        )
        predecessor = _select_latest_indexed_predecessor(
            repository, mutation=mutation, binding_id=binding_id, base_revision=target_revision - 1,
        )
        if predecessor is None:
            raise RuntimeError("disable_indexed_source_predecessor_missing")
        if predecessor.indexed_source_binding_id != binding_id:
            raise RuntimeError("disable_indexed_source_binding_identity_conflict")
        if predecessor.knowledge_source_binding_ref != intent.knowledge_source_binding_ref.strip():
            raise RuntimeError("disable_indexed_source_binding_ref_conflict")
        if predecessor.status not in _DISABLE_PREDECESSOR_STATUSES:
            raise RuntimeError("disable_indexed_source_predecessor_transition_invalid")
        binding = predecessor.model_copy(update={
            "status": WorkspaceIndexedSourceBindingStatusV1.DISABLED,
            "mutation_id": mutation.mutation_id,
            "effective_revision": target_revision,
            "updated_at": now,
        })
        if not repository.put_knowledge_indexed_source_version_if_absent(binding):
            raise RuntimeError("disable_indexed_source_stage_conflict")
        return WorkspaceKnowledgeStagedResult(
            result_entity_type=_RESULT_ENTITY_TYPE, result_entity_id=binding_id,
        )

    def inspect_staged(
        self, *, repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
    ) -> WorkspaceKnowledgeStageInspection:
        return _inspect_disable_indexed_staged(repository, mutation)

    def cleanup_staged(
        self, *, repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord, inspection: WorkspaceKnowledgeStageInspection,
    ) -> bool:
        if inspection.state is WorkspaceKnowledgeStageStateV1.ABSENT:
            return True
        if inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            return False
        current_inspection = _inspect_disable_indexed_staged(repository, mutation)
        if current_inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            return False
        if current_inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        if inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        owned = _owned_indexed_bindings(repository, mutation)
        if len(owned) != 1:
            return False
        staged = owned[0]
        if (
            mutation.target_revision is None
            or staged.effective_revision != mutation.target_revision
            or current_inspection.result_entity_id != staged.indexed_source_binding_id
        ):
            return False
        if not repository.delete_knowledge_indexed_source_version_if_match(staged):
            return False
        return _confirm_indexed_cleanup(repository, mutation)
