# © Artur Czarnecki. All rights reserved.

"""Workspace Knowledge Configuration mutation handlers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from local_workspace_application.workspaces.connected_source_ids import (
    connected_source_id,
    indexed_source_binding_id,
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
    return [
        v for v in repository.list_knowledge_indexed_source_versions(
            tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
        ) if v.mutation_id == mutation.mutation_id
    ]

def _owned_connected_sources(
    repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
) -> list[WorkspaceSource]:
    return [
        s for s in repository.list_sources(
            tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
        ) if s.knowledge_configuration_creation_mutation_id == mutation.mutation_id
    ]

def _indexed_binding_predecessor(
    repository: ManagedWorkspaceRepository, *, mutation: WorkspaceKnowledgeMutationRecord,
    binding_id: str, base_revision: int,
) -> WorkspaceIndexedSourceBinding | None:
    return repository.get_knowledge_indexed_source_version(
        tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
        indexed_source_binding_id=binding_id, effective_revision=base_revision,
    )

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

def _inspect_create_indexed_staged(
    repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
) -> WorkspaceKnowledgeStageInspection:
    if mutation.target_revision is None:
        return _stage_conflict()
    owned_bindings = _owned_indexed_bindings(repository, mutation)
    owned_sources = _owned_connected_sources(repository, mutation)
    if not owned_bindings and not owned_sources:
        return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.ABSENT)
    target_bindings = [b for b in owned_bindings if b.effective_revision == mutation.target_revision]
    if len(target_bindings) > 1 or len(owned_sources) > 1:
        return _stage_conflict()
    target_binding = target_bindings[0] if target_bindings else None
    owned_source = owned_sources[0] if owned_sources else None
    if target_binding is None:
        if owned_source is not None:
            return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED)
        return _stage_conflict()
    try:
        binding_id, source_id, _ = _create_indexed_source_identity(
            mutation=mutation, target_revision=mutation.target_revision,
            intent=CreateIndexedSourceMutationIntent(
                knowledge_source_binding_ref=target_binding.knowledge_source_binding_ref,
                sync_mode=target_binding.sync_mode,
                audience_eligibility=target_binding.audience_eligibility,
                cached_safe_display_label=target_binding.cached_safe_display_label,
            ),
        )
    except RuntimeError:
        return _stage_conflict()
    if (
        target_binding.indexed_source_binding_id != binding_id
        or target_binding.source_id != source_id
        or target_binding.status is not WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    ):
        return _stage_conflict()
    base_revision = mutation.target_revision - 1
    predecessor = _indexed_binding_predecessor(
        repository, mutation=mutation, binding_id=binding_id, base_revision=base_revision,
    )
    if predecessor is None:
        if owned_source is None:
            return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED)
        expected = _connected_source(
            source_id=source_id, mutation=mutation, target_revision=mutation.target_revision,
            created_at=owned_source.created_at, status=owned_source.status,
        )
        if not _is_compatible_connected_source(owned_source, expected=expected):
            return _stage_conflict()
        return _stage_valid(binding_id)
    if owned_source is not None:
        return _stage_conflict()
    try:
        validate_connected_source_durable_origin(
            repository=repository, tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id, source_id=source_id, binding=predecessor,
            committed_configuration_revision=base_revision,
        )
    except ConnectedSourceOriginValidationError:
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
        predecessor = _indexed_binding_predecessor(
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
        if inspection.state is WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED:
            bindings = _owned_indexed_bindings(repository, mutation)
            sources = _owned_connected_sources(repository, mutation)
            if bindings and sources:
                return False
            return _delete_owned_indexed_rows(
                repository, mutation, bindings=bindings, sources=sources,
            )
        if inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        bindings = _owned_indexed_bindings(repository, mutation)
        if len(bindings) != 1:
            return False
        binding = bindings[0]
        if (
            mutation.target_revision is None
            or binding.effective_revision != mutation.target_revision
            or inspection.result_entity_id != binding.indexed_source_binding_id
        ):
            return False
        if not repository.delete_knowledge_indexed_source_version_if_match(binding):
            return False
        return _delete_owned_indexed_rows(
            repository, mutation, bindings=[], sources=_owned_connected_sources(repository, mutation),
        )


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

def _disabled_binding(
    predecessor: WorkspaceIndexedSourceBinding, *, mutation: WorkspaceKnowledgeMutationRecord,
    target_revision: int, now: datetime,
) -> WorkspaceIndexedSourceBinding:
    return predecessor.model_copy(update={
        "status": WorkspaceIndexedSourceBindingStatusV1.DISABLED,
        "mutation_id": mutation.mutation_id,
        "effective_revision": target_revision,
        "updated_at": now,
    })

def _inspect_disable_indexed_staged(
    repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
) -> WorkspaceKnowledgeStageInspection:
    owned = _owned_indexed_bindings(repository, mutation)
    if not owned:
        return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.ABSENT)
    if len(owned) != 1 or mutation.target_revision is None:
        return _stage_conflict()
    staged = owned[0]
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
    if (
        staged.indexed_source_binding_id != binding_id
        or staged.status is not WorkspaceIndexedSourceBindingStatusV1.DISABLED
    ):
        return _stage_conflict()
    predecessor = _indexed_binding_predecessor(
        repository, mutation=mutation, binding_id=binding_id,
        base_revision=mutation.target_revision - 1,
    )
    if predecessor is None or predecessor.status not in _DISABLE_PREDECESSOR_STATUSES:
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
        predecessor = _indexed_binding_predecessor(
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
        binding = _disabled_binding(
            predecessor, mutation=mutation, target_revision=target_revision, now=now,
        )
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
        if inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        owned = _owned_indexed_bindings(repository, mutation)
        if len(owned) != 1:
            return False
        staged = owned[0]
        if (
            mutation.target_revision is None
            or staged.effective_revision != mutation.target_revision
            or inspection.result_entity_id != staged.indexed_source_binding_id
        ):
            return False
        if not repository.delete_knowledge_indexed_source_version_if_match(staged):
            return False
        return _confirm_indexed_cleanup(repository, mutation)
