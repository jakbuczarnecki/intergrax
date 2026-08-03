# © Artur Czarnecki. All rights reserved.

"""Workspace Live Access Binding mutation handlers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.integrations.contracts.base import IntegrationCategory
from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    live_access_binding_id_from_semantic_hash,
    normalize_create_live_access_binding_request_hash,
    normalize_disable_live_access_binding_request_hash,
    normalize_live_access_capability_set,
    normalize_live_access_remote_resource_id,
    semantic_identity_hash_for_live_access_binding,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    KnowledgeAudienceEligibilityV1,
    LiveAccessBindingStatusV1,
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

_RESULT_ENTITY_TYPE = "live_access_binding"
_REACTIVATION_PREDECESSOR_STATUSES = frozenset({
    LiveAccessBindingStatusV1.ACTIVE, LiveAccessBindingStatusV1.DISABLED,
    LiveAccessBindingStatusV1.UNAVAILABLE, LiveAccessBindingStatusV1.REVOKED,
})
_DISABLE_PREDECESSOR_STATUSES = frozenset({
    LiveAccessBindingStatusV1.ACTIVE, LiveAccessBindingStatusV1.UNAVAILABLE, LiveAccessBindingStatusV1.REVOKED,
})


def _stage_conflict() -> WorkspaceKnowledgeStageInspection:
    return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT)


def _stage_valid(binding_id: str) -> WorkspaceKnowledgeStageInspection:
    return WorkspaceKnowledgeStageInspection(
        state=WorkspaceKnowledgeStageStateV1.COMPLETE_VALID,
        result_entity_type=_RESULT_ENTITY_TYPE,
        result_entity_id=binding_id,
    )


def _owned_live_bindings(repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord):
    return [
        binding for binding in repository.list_knowledge_live_access_versions(
            tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
        ) if binding.mutation_id == mutation.mutation_id
    ]


def _select_latest_live_predecessor(
    repository: ManagedWorkspaceRepository, *, mutation: WorkspaceKnowledgeMutationRecord,
    binding_id: str, base_revision: int,
):
    matches = [
        version for version in repository.list_knowledge_live_access_versions(
            tenant_id=mutation.tenant_id, workspace_id=mutation.workspace_id,
        )
        if version.live_access_binding_id == binding_id and version.effective_revision <= base_revision
    ]
    if not matches:
        return None
    top_revision = max(version.effective_revision for version in matches)
    top_matches = [version for version in matches if version.effective_revision == top_revision]
    if len(top_matches) > 1:
        raise RuntimeError("live_access_predecessor_ambiguous")
    return top_matches[0]


@dataclass(frozen=True, slots=True)
class CreateLiveAccessBindingMutationIntent:
    connection_ref: str
    remote_resource_id: str | None
    allowed_capability_ids: tuple[str, ...]
    audience_eligibility: KnowledgeAudienceEligibilityV1
    derived_provider_id: str
    derived_integration_kind: IntegrationCategory
    derived_resource_type: str | None
    derived_safe_display_label: str


@dataclass(frozen=True, slots=True)
class DisableLiveAccessBindingMutationIntent:
    live_access_binding_id: str


def _intent_from_binding(binding: WorkspaceLiveAccessBinding) -> CreateLiveAccessBindingMutationIntent:
    return CreateLiveAccessBindingMutationIntent(
        connection_ref=binding.connection_ref, remote_resource_id=binding.remote_resource_id,
        allowed_capability_ids=binding.allowed_capability_ids, audience_eligibility=binding.audience_eligibility,
        derived_provider_id=binding.derived_provider_id, derived_integration_kind=binding.derived_integration_kind,
        derived_resource_type=binding.derived_resource_type, derived_safe_display_label=binding.derived_safe_display_label,
    )


def _create_live_access_identity(
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    target_revision: int,
    intent: CreateLiveAccessBindingMutationIntent,
) -> tuple[str, str, str | None, tuple[str, ...]]:
    if mutation.operation is not WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING:
        raise RuntimeError("create_live_access_binding_operation_required")
    if mutation.target_revision != target_revision:
        raise RuntimeError("create_live_access_binding_target_revision_mismatch")
    normalized_resource_id = normalize_live_access_remote_resource_id(intent.remote_resource_id)
    normalized_capabilities = normalize_live_access_capability_set(intent.allowed_capability_ids)
    expected_request = normalize_create_live_access_binding_request_hash(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        connection_ref=intent.connection_ref,
        remote_resource_id=normalized_resource_id,
        allowed_capability_ids=normalized_capabilities,
        audience_eligibility=intent.audience_eligibility,
    )
    if mutation.normalized_request_hash != expected_request:
        raise RuntimeError("create_live_access_binding_request_hash_mismatch")
    expected_semantic = semantic_identity_hash_for_live_access_binding(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        connection_ref=intent.connection_ref,
        normalized_remote_resource_id=normalized_resource_id,
        normalized_capability_set=normalized_capabilities,
    )
    if mutation.semantic_identity_hash != expected_semantic:
        raise RuntimeError("create_live_access_binding_semantic_hash_mismatch")
    binding_id = live_access_binding_id_from_semantic_hash(expected_semantic)
    if mutation.result_entity_type is not None and mutation.result_entity_type != _RESULT_ENTITY_TYPE:
        raise RuntimeError("create_live_access_binding_result_type_mismatch")
    if mutation.result_entity_id is not None and mutation.result_entity_id != binding_id:
        raise RuntimeError("create_live_access_binding_result_id_mismatch")
    return binding_id, expected_semantic, normalized_resource_id, normalized_capabilities


def _active_binding(
    *,
    binding_id: str,
    semantic_hash: str,
    mutation: WorkspaceKnowledgeMutationRecord,
    intent: CreateLiveAccessBindingMutationIntent,
    normalized_resource_id: str | None,
    normalized_capabilities: tuple[str, ...],
    target_revision: int,
    now: datetime,
    created_at: datetime,
) -> WorkspaceLiveAccessBinding:
    return WorkspaceLiveAccessBinding(
        live_access_binding_id=binding_id,
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        connection_ref=intent.connection_ref.strip(),
        remote_resource_id=normalized_resource_id,
        allowed_capability_ids=normalized_capabilities,
        derived_provider_id=intent.derived_provider_id,
        derived_integration_kind=intent.derived_integration_kind,
        derived_resource_type=intent.derived_resource_type,
        derived_safe_display_label=intent.derived_safe_display_label,
        status=LiveAccessBindingStatusV1.ACTIVE,
        audience_eligibility=intent.audience_eligibility,
        mutation_id=mutation.mutation_id,
        effective_revision=target_revision,
        semantic_identity_hash=semantic_hash,
        created_at=created_at,
        updated_at=now,
    )


def _assert_reactivation_predecessor(predecessor: WorkspaceLiveAccessBinding, *, binding_id: str, semantic_hash: str, connection_ref: str) -> None:
    if (
        predecessor.live_access_binding_id != binding_id or predecessor.semantic_identity_hash != semantic_hash
        or predecessor.connection_ref.strip() != connection_ref.strip()
        or predecessor.status not in _REACTIVATION_PREDECESSOR_STATUSES
    ):
        raise RuntimeError("live_access_predecessor_transition_invalid")


def _intent_from_binding(binding: WorkspaceLiveAccessBinding) -> CreateLiveAccessBindingMutationIntent:
    return CreateLiveAccessBindingMutationIntent(
        connection_ref=binding.connection_ref,
        remote_resource_id=binding.remote_resource_id,
        allowed_capability_ids=binding.allowed_capability_ids,
        audience_eligibility=binding.audience_eligibility,
        derived_provider_id=binding.derived_provider_id,
        derived_integration_kind=binding.derived_integration_kind,
        derived_resource_type=binding.derived_resource_type,
        derived_safe_display_label=binding.derived_safe_display_label,
    )


def _inspect_create_live_staged(repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord):
    if mutation.target_revision is None:
        return _stage_conflict()
    owned_bindings = _owned_live_bindings(repository, mutation)
    if len(owned_bindings) > 1:
        return _stage_conflict()
    if not owned_bindings:
        return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.ABSENT)
    staged = owned_bindings[0]
    if staged.effective_revision != mutation.target_revision:
        return _stage_conflict()
    try:
        intent = _intent_from_binding(staged)
        binding_id, semantic_hash, normalized_resource_id, normalized_capabilities = (
            _create_live_access_identity(
                mutation=mutation,
                target_revision=mutation.target_revision,
                intent=intent,
            )
        )
    except RuntimeError:
        return _stage_conflict()
    if staged.live_access_binding_id != binding_id:
        return _stage_conflict()
    base_revision = mutation.target_revision - 1
    try:
        predecessor = _select_latest_live_predecessor(
            repository,
            mutation=mutation,
            binding_id=binding_id,
            base_revision=base_revision,
        )
    except RuntimeError:
        return _stage_conflict()
    expected = _active_binding(
        binding_id=binding_id,
        semantic_hash=semantic_hash,
        mutation=mutation,
        intent=intent,
        normalized_resource_id=normalized_resource_id,
        normalized_capabilities=normalized_capabilities,
        target_revision=mutation.target_revision,
        now=staged.updated_at,
        created_at=staged.created_at if predecessor is None else predecessor.created_at,
    )
    if staged != expected:
        return _stage_conflict()
    return _stage_valid(binding_id)


class CreateLiveAccessBindingMutationHandler:
    operation = WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING

    def find_existing_result(self, *, configuration: WorkspaceKnowledgeConfigurationV1, intent: object):
        if not isinstance(intent, CreateLiveAccessBindingMutationIntent):
            raise ValueError("create_live_access_binding_intent_required")
        normalized_resource_id = normalize_live_access_remote_resource_id(intent.remote_resource_id)
        normalized_capabilities = normalize_live_access_capability_set(intent.allowed_capability_ids)
        semantic_hash = semantic_identity_hash_for_live_access_binding(
            tenant_id=configuration.tenant_id,
            workspace_id=configuration.workspace_id,
            connection_ref=intent.connection_ref,
            normalized_remote_resource_id=normalized_resource_id,
            normalized_capability_set=normalized_capabilities,
        )
        binding_id = live_access_binding_id_from_semantic_hash(semantic_hash)
        for binding in configuration.live_access_bindings:
            if binding.live_access_binding_id != binding_id:
                continue
            if binding.semantic_identity_hash != semantic_hash:
                continue
            if binding.status is not LiveAccessBindingStatusV1.ACTIVE:
                continue
            if binding.allowed_capability_ids != normalized_capabilities:
                continue
            if binding.audience_eligibility != intent.audience_eligibility:
                continue
            return WorkspaceKnowledgeExistingResult(
                result_entity_type=_RESULT_ENTITY_TYPE,
                result_entity_id=binding_id,
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
        if not isinstance(intent, CreateLiveAccessBindingMutationIntent):
            raise ValueError("create_live_access_binding_intent_required")
        binding_id, semantic_hash, normalized_resource_id, normalized_capabilities = (
            _create_live_access_identity(
                mutation=mutation,
                target_revision=target_revision,
                intent=intent,
            )
        )
        base_revision = target_revision - 1
        predecessor = _select_latest_live_predecessor(
            repository,
            mutation=mutation,
            binding_id=binding_id,
            base_revision=base_revision,
        )
        if predecessor is not None:
            _assert_reactivation_predecessor(
                predecessor,
                binding_id=binding_id,
                semantic_hash=semantic_hash,
                connection_ref=intent.connection_ref,
            )
        binding = _active_binding(
            binding_id=binding_id,
            semantic_hash=semantic_hash,
            mutation=mutation,
            intent=intent,
            normalized_resource_id=normalized_resource_id,
            normalized_capabilities=normalized_capabilities,
            target_revision=target_revision,
            now=now,
            created_at=now if predecessor is None else predecessor.created_at,
        )
        if not repository.put_knowledge_live_access_version_if_absent(binding):
            raise RuntimeError("live_access_binding_stage_conflict")
        return WorkspaceKnowledgeStagedResult(
            result_entity_type=_RESULT_ENTITY_TYPE,
            result_entity_id=binding_id,
        )

    def inspect_staged(self, *, repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord):
        return _inspect_create_live_staged(repository, mutation)

    def cleanup_staged(
        self, *, repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
        inspection: WorkspaceKnowledgeStageInspection,
    ) -> bool:
        if inspection.state is WorkspaceKnowledgeStageStateV1.ABSENT:
            return True
        if inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            return False
        current_inspection = _inspect_create_live_staged(repository, mutation)
        if current_inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            return False
        if current_inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        if inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        owned = _owned_live_bindings(repository, mutation)
        if len(owned) != 1:
            return False
        staged = owned[0]
        if (
            mutation.target_revision is None
            or staged.effective_revision != mutation.target_revision
            or current_inspection.result_entity_id != staged.live_access_binding_id
        ):
            return False
        return repository.delete_knowledge_live_access_version_if_match(staged)


def _disable_live_access_identity(
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    target_revision: int,
    intent: DisableLiveAccessBindingMutationIntent,
) -> str:
    if mutation.operation is not WorkspaceKnowledgeMutationOperationV1.DISABLE_LIVE_ACCESS_BINDING:
        raise RuntimeError("disable_live_access_binding_operation_required")
    if mutation.target_revision != target_revision:
        raise RuntimeError("disable_live_access_binding_target_revision_mismatch")
    binding_id = intent.live_access_binding_id.strip()
    expected_request = normalize_disable_live_access_binding_request_hash(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        live_access_binding_id=binding_id,
    )
    if mutation.normalized_request_hash != expected_request:
        raise RuntimeError("disable_live_access_binding_request_hash_mismatch")
    if mutation.result_entity_type is not None and mutation.result_entity_type != _RESULT_ENTITY_TYPE:
        raise RuntimeError("disable_live_access_binding_result_type_mismatch")
    if mutation.result_entity_id is not None and mutation.result_entity_id != binding_id:
        raise RuntimeError("disable_live_access_binding_result_id_mismatch")
    return binding_id


def _inspect_disable_live_staged(
    repository: ManagedWorkspaceRepository,
    mutation: WorkspaceKnowledgeMutationRecord,
) -> WorkspaceKnowledgeStageInspection:
    owned_bindings = _owned_live_bindings(repository, mutation)
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
        binding_id = _disable_live_access_identity(
            mutation=mutation,
            target_revision=mutation.target_revision,
            intent=DisableLiveAccessBindingMutationIntent(
                live_access_binding_id=staged.live_access_binding_id,
            ),
        )
    except RuntimeError:
        return _stage_conflict()
    if staged.live_access_binding_id != binding_id:
        return _stage_conflict()
    base_revision = mutation.target_revision - 1
    try:
        predecessor = _select_latest_live_predecessor(
            repository,
            mutation=mutation,
            binding_id=binding_id,
            base_revision=base_revision,
        )
    except RuntimeError:
        return _stage_conflict()
    if predecessor is None or predecessor.status not in _DISABLE_PREDECESSOR_STATUSES:
        return _stage_conflict()
    expected = predecessor.model_copy(
        update={
            "status": LiveAccessBindingStatusV1.DISABLED,
            "mutation_id": mutation.mutation_id,
            "effective_revision": mutation.target_revision,
            "updated_at": staged.updated_at,
        }
    )
    if staged != expected:
        return _stage_conflict()
    return _stage_valid(binding_id)


class DisableLiveAccessBindingMutationHandler:
    operation = WorkspaceKnowledgeMutationOperationV1.DISABLE_LIVE_ACCESS_BINDING

    def find_existing_result(self, *, configuration: WorkspaceKnowledgeConfigurationV1, intent: object):
        if not isinstance(intent, DisableLiveAccessBindingMutationIntent):
            raise ValueError("disable_live_access_binding_intent_required")
        binding_id = intent.live_access_binding_id.strip()
        for binding in configuration.live_access_bindings:
            if (
                binding.live_access_binding_id == binding_id
                and binding.status is LiveAccessBindingStatusV1.DISABLED
            ):
                return WorkspaceKnowledgeExistingResult(
                    result_entity_type=_RESULT_ENTITY_TYPE,
                    result_entity_id=binding_id,
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
        if not isinstance(intent, DisableLiveAccessBindingMutationIntent):
            raise ValueError("disable_live_access_binding_intent_required")
        binding_id = _disable_live_access_identity(
            mutation=mutation,
            target_revision=target_revision,
            intent=intent,
        )
        predecessor = _select_latest_live_predecessor(
            repository,
            mutation=mutation,
            binding_id=binding_id,
            base_revision=target_revision - 1,
        )
        if predecessor is None:
            raise RuntimeError("disable_live_access_binding_predecessor_missing")
        if predecessor.live_access_binding_id != binding_id:
            raise RuntimeError("disable_live_access_binding_identity_conflict")
        if predecessor.status not in _DISABLE_PREDECESSOR_STATUSES:
            raise RuntimeError("disable_live_access_binding_predecessor_transition_invalid")
        binding = predecessor.model_copy(
            update={
                "status": LiveAccessBindingStatusV1.DISABLED,
                "mutation_id": mutation.mutation_id,
                "effective_revision": target_revision,
                "updated_at": now,
            }
        )
        if not repository.put_knowledge_live_access_version_if_absent(binding):
            raise RuntimeError("disable_live_access_binding_stage_conflict")
        return WorkspaceKnowledgeStagedResult(
            result_entity_type=_RESULT_ENTITY_TYPE,
            result_entity_id=binding_id,
        )

    def inspect_staged(self, *, repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord):
        return _inspect_disable_live_staged(repository, mutation)

    def cleanup_staged(
        self, *, repository: ManagedWorkspaceRepository, mutation: WorkspaceKnowledgeMutationRecord,
        inspection: WorkspaceKnowledgeStageInspection,
    ) -> bool:
        if inspection.state is WorkspaceKnowledgeStageStateV1.ABSENT:
            return True
        if inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            return False
        current_inspection = _inspect_disable_live_staged(repository, mutation)
        if current_inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            return False
        if current_inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        if inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            return False
        owned = _owned_live_bindings(repository, mutation)
        if len(owned) != 1:
            return False
        staged = owned[0]
        if (
            mutation.target_revision is None
            or staged.effective_revision != mutation.target_revision
            or current_inspection.result_entity_id != staged.live_access_binding_id
        ):
            return False
        return repository.delete_knowledge_live_access_version_if_match(staged)
