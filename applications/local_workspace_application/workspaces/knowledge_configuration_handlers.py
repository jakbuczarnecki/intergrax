# © Artur Czarnecki. All rights reserved.

"""Workspace Knowledge Configuration mutation handlers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from local_workspace_application.workspaces.connected_source_ids import (
    connected_source_id,
    indexed_source_binding_id,
    workspace_indexed_source_semantic_hash,
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
        staged = owned[0]
        if mutation.target_revision is None:
            return WorkspaceKnowledgeStageInspection(
                state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT,
            )
        if staged.effective_revision != mutation.target_revision:
            return WorkspaceKnowledgeStageInspection(
                state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT,
            )
        if staged.tenant_id != mutation.tenant_id:
            return WorkspaceKnowledgeStageInspection(
                state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT,
            )
        if staged.workspace_id != mutation.workspace_id:
            return WorkspaceKnowledgeStageInspection(
                state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT,
            )
        if staged.status is not WorkspaceConnectionAttachmentStatusV1.ATTACHED:
            return WorkspaceKnowledgeStageInspection(
                state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT,
            )
        return WorkspaceKnowledgeStageInspection(
            state=WorkspaceKnowledgeStageStateV1.COMPLETE_VALID,
            result_entity_type=_CONNECTION_ATTACHMENT_RESULT_TYPE,
            result_entity_id=staged.attachment_id,
        )

    def cleanup_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
        inspection: WorkspaceKnowledgeStageInspection,
    ) -> bool:
        versions = repository.list_knowledge_connection_attachment_versions(
            tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id,
        )
        for version in versions:
            if version.mutation_id != mutation.mutation_id:
                continue
            if not repository.delete_knowledge_connection_attachment_version_if_match(version):
                return False
        remaining = repository.list_knowledge_connection_attachment_versions(
            tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id,
        )
        for version in remaining:
            if version.mutation_id == mutation.mutation_id:
                return False
        return True


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
    )


@dataclass(frozen=True, slots=True)
class CreateIndexedSourceMutationIntent:
    knowledge_source_binding_ref: str
    sync_mode: IndexedSourceSyncModeV1 = IndexedSourceSyncModeV1.FULL
    audience_eligibility: IndexedSourceAudienceEligibilityV1 = (
        IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY
    )
    cached_safe_display_label: str | None = None


class CreateIndexedSourceMutationHandler:
    operation = WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE

    def find_existing_result(
        self,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        intent: object,
    ) -> WorkspaceKnowledgeExistingResult | None:
        if not isinstance(intent, CreateIndexedSourceMutationIntent):
            raise ValueError("create_indexed_source_intent_required")
        binding_ref = intent.knowledge_source_binding_ref.strip()
        for binding in configuration.indexed_sources:
            if binding.knowledge_source_binding_ref != binding_ref:
                continue
            if binding.status is not WorkspaceIndexedSourceBindingStatusV1.ACTIVE:
                continue
            return WorkspaceKnowledgeExistingResult(
                result_entity_type=_RESULT_ENTITY_TYPE,
                result_entity_id=binding.indexed_source_binding_id,
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
        if not isinstance(intent, CreateIndexedSourceMutationIntent):
            raise ValueError("create_indexed_source_intent_required")
        binding_ref = intent.knowledge_source_binding_ref.strip()
        binding_id = indexed_source_binding_id(
            mutation.tenant_id,
            mutation.workspace_id,
            binding_ref,
        )
        source_id = connected_source_id(
            mutation.tenant_id,
            mutation.workspace_id,
            binding_ref,
        )
        semantic_hash = workspace_indexed_source_semantic_hash(
            mutation.tenant_id,
            mutation.workspace_id,
            binding_ref,
        )

        binding = WorkspaceIndexedSourceBinding(
            indexed_source_binding_id=binding_id,
            tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id,
            knowledge_source_binding_ref=binding_ref,
            source_id=source_id,
            sync_mode=intent.sync_mode,
            status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
            audience_eligibility=intent.audience_eligibility,
            mutation_id=mutation.mutation_id,
            effective_revision=target_revision,
            semantic_identity_hash=semantic_hash,
            created_at=now,
            updated_at=now,
            cached_safe_display_label=intent.cached_safe_display_label,
        )
        if not repository.put_knowledge_indexed_source_version_if_absent(binding):
            raise RuntimeError("indexed_source_binding_stage_conflict")

        source = WorkspaceSource(
            source_id=source_id,
            workspace_id=mutation.workspace_id,
            tenant_id=mutation.tenant_id,
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            path="",
            recursive=False,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=now,
            knowledge_configuration_creation_mutation_id=mutation.mutation_id,
            knowledge_configuration_visibility_revision=target_revision,
        )
        if not repository.put_source_if_absent(source):
            existing = repository.get_source(
                tenant_id=mutation.tenant_id,
                workspace_id=mutation.workspace_id,
                source_id=source_id,
            )
            if existing is None:
                raise RuntimeError("connected_source_stage_conflict")
            if not _is_compatible_connected_source(existing, expected=source):
                raise RuntimeError("connected_source_identity_conflict")

        return WorkspaceKnowledgeStagedResult(
            result_entity_type=_RESULT_ENTITY_TYPE,
            result_entity_id=binding_id,
        )

    def inspect_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
    ) -> WorkspaceKnowledgeStageInspection:
        versions = repository.list_knowledge_indexed_source_versions(
            tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id,
        )
        staged_binding: WorkspaceIndexedSourceBinding | None = None
        for version in versions:
            if version.mutation_id != mutation.mutation_id:
                continue
            if staged_binding is None or version.effective_revision > staged_binding.effective_revision:
                staged_binding = version

        if staged_binding is None:
            return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.ABSENT)

        source = repository.get_source(
            tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id,
            source_id=staged_binding.source_id,
        )
        if source is None:
            return WorkspaceKnowledgeStageInspection(
                state=WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED,
            )

        expected_source = WorkspaceSource(
            source_id=staged_binding.source_id,
            workspace_id=mutation.workspace_id,
            tenant_id=mutation.tenant_id,
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            path="",
            recursive=False,
            status=source.status,
            created_at=source.created_at,
            knowledge_configuration_creation_mutation_id=mutation.mutation_id,
            knowledge_configuration_visibility_revision=staged_binding.effective_revision,
        )
        if not _is_compatible_connected_source(source, expected=expected_source):
            return WorkspaceKnowledgeStageInspection(
                state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT,
            )

        return WorkspaceKnowledgeStageInspection(
            state=WorkspaceKnowledgeStageStateV1.COMPLETE_VALID,
            result_entity_type=_RESULT_ENTITY_TYPE,
            result_entity_id=staged_binding.indexed_source_binding_id,
        )

    def cleanup_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
        inspection: WorkspaceKnowledgeStageInspection,
    ) -> bool:
        versions = repository.list_knowledge_indexed_source_versions(
            tenant_id=mutation.tenant_id,
            workspace_id=mutation.workspace_id,
        )
        for version in versions:
            if version.mutation_id != mutation.mutation_id:
                continue
            if not repository.delete_knowledge_indexed_source_version_if_match(version):
                return False

        for version in versions:
            if version.mutation_id != mutation.mutation_id:
                continue
            source = repository.get_source(
                tenant_id=mutation.tenant_id,
                workspace_id=mutation.workspace_id,
                source_id=version.source_id,
            )
            if source is None:
                continue
            if source.knowledge_configuration_creation_mutation_id != mutation.mutation_id:
                return False
            if source.knowledge_configuration_visibility_revision != version.effective_revision:
                return False
            if not repository.delete_source_if_match(source):
                return False
        return True


def _is_compatible_connected_source(
    actual: WorkspaceSource,
    *,
    expected: WorkspaceSource,
) -> bool:
    return (
        actual.tenant_id == expected.tenant_id
        and actual.workspace_id == expected.workspace_id
        and actual.source_id == expected.source_id
        and actual.source_type is WorkspaceSourceType.CONNECTED_SOURCE
        and actual.path == ""
        and actual.recursive is False
        and actual.knowledge_configuration_creation_mutation_id
        == expected.knowledge_configuration_creation_mutation_id
        and actual.knowledge_configuration_visibility_revision
        == expected.knowledge_configuration_visibility_revision
    )
