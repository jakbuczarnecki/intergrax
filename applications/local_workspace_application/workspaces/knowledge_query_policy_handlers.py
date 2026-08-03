# © Artur Czarnecki. All rights reserved.

"""Workspace Query Policy mutation handlers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    normalize_update_query_policy_request_hash,
    query_policy_stage_manifest_hash,
    semantic_identity_hash_for_query_policy,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveResultRetentionV1,
    QueryPolicyModeV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceQueryPolicy,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeExistingResult,
    WorkspaceKnowledgeStageInspection,
    WorkspaceKnowledgeStageStateV1,
    WorkspaceKnowledgeStagedResult,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_RESULT_ENTITY_TYPE = "query_policy"
_QUERY_POLICY_ENTITY_ID = "query-policy"


@dataclass(frozen=True, slots=True)
class UpdateQueryPolicyMutationIntent:
    mode: QueryPolicyModeV1
    allowed_connection_refs: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    max_live_calls: int
    max_total_duration_ms: int
    max_result_items: int
    max_result_bytes: int
    live_result_retention: LiveResultRetentionV1


def _stage_conflict() -> WorkspaceKnowledgeStageInspection:
    return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT)


def _stage_valid() -> WorkspaceKnowledgeStageInspection:
    return WorkspaceKnowledgeStageInspection(
        state=WorkspaceKnowledgeStageStateV1.COMPLETE_VALID,
        result_entity_type=_RESULT_ENTITY_TYPE,
        result_entity_id=_QUERY_POLICY_ENTITY_ID,
    )


def _owned_query_policies(
    repository: ManagedWorkspaceRepository,
    mutation: WorkspaceKnowledgeMutationRecord,
) -> list[WorkspaceQueryPolicy]:
    try:
        return [
            policy
            for policy in repository.list_knowledge_query_policy_versions(
                tenant_id=mutation.tenant_id,
                workspace_id=mutation.workspace_id,
            )
            if policy.mutation_id == mutation.mutation_id
        ]
    except ValueError:
        raise RuntimeError("query_policy_owned_rows_unreadable") from None


def _intent_from_policy(policy: WorkspaceQueryPolicy) -> UpdateQueryPolicyMutationIntent:
    return UpdateQueryPolicyMutationIntent(
        mode=policy.mode,
        allowed_connection_refs=policy.allowed_connection_refs,
        allowed_capability_ids=policy.allowed_capability_ids,
        max_live_calls=policy.max_live_calls,
        max_total_duration_ms=policy.max_total_duration_ms,
        max_result_items=policy.max_result_items,
        max_result_bytes=policy.max_result_bytes,
        live_result_retention=policy.live_result_retention,
    )


def _policy_matches_intent(
    policy: WorkspaceQueryPolicy,
    intent: UpdateQueryPolicyMutationIntent,
) -> bool:
    return (
        policy.mode is intent.mode
        and policy.allowed_connection_refs == intent.allowed_connection_refs
        and policy.allowed_capability_ids == intent.allowed_capability_ids
        and policy.max_live_calls == intent.max_live_calls
        and policy.max_total_duration_ms == intent.max_total_duration_ms
        and policy.max_result_items == intent.max_result_items
        and policy.max_result_bytes == intent.max_result_bytes
        and policy.live_result_retention is intent.live_result_retention
    )


def _query_policy_request_hash(
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    intent: UpdateQueryPolicyMutationIntent,
) -> str:
    return normalize_update_query_policy_request_hash(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )


def _query_policy_semantic_hash(
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    intent: UpdateQueryPolicyMutationIntent,
) -> str:
    return semantic_identity_hash_for_query_policy(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )


def _query_policy_stage_manifest_hash(
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    intent: UpdateQueryPolicyMutationIntent,
) -> str:
    return query_policy_stage_manifest_hash(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )


def _stage_manifest_matches(
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    intent: UpdateQueryPolicyMutationIntent,
) -> bool:
    if mutation.stage_manifest_hash is None:
        return False
    expected = _query_policy_stage_manifest_hash(mutation=mutation, intent=intent)
    return mutation.stage_manifest_hash == expected


def _update_query_policy_identity(
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    target_revision: int,
    intent: UpdateQueryPolicyMutationIntent,
) -> None:
    if mutation.operation is not WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY:
        raise RuntimeError("query_policy_operation_required")
    if mutation.target_revision != target_revision:
        raise RuntimeError("query_policy_target_revision_mismatch")
    if mutation.normalized_request_hash != _query_policy_request_hash(
        mutation=mutation,
        intent=intent,
    ):
        raise RuntimeError("query_policy_request_hash_mismatch")
    if mutation.semantic_identity_hash != _query_policy_semantic_hash(
        mutation=mutation,
        intent=intent,
    ):
        raise RuntimeError("query_policy_semantic_hash_mismatch")
    if (
        mutation.result_entity_type is not None
        and mutation.result_entity_type != _RESULT_ENTITY_TYPE
    ):
        raise RuntimeError("query_policy_result_type_mismatch")
    if (
        mutation.result_entity_id is not None
        and mutation.result_entity_id != _QUERY_POLICY_ENTITY_ID
    ):
        raise RuntimeError("query_policy_result_id_mismatch")


def _expected_policy_row(
    *,
    mutation: WorkspaceKnowledgeMutationRecord,
    target_revision: int,
    intent: UpdateQueryPolicyMutationIntent,
    updated_at: datetime,
) -> WorkspaceQueryPolicy:
    return WorkspaceQueryPolicy(
        tenant_id=mutation.tenant_id,
        workspace_id=mutation.workspace_id,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
        mutation_id=mutation.mutation_id,
        effective_revision=target_revision,
        updated_at=updated_at,
    )


def _inspect_update_query_policy_staged(
    repository: ManagedWorkspaceRepository,
    mutation: WorkspaceKnowledgeMutationRecord,
) -> WorkspaceKnowledgeStageInspection:
    if mutation.target_revision is None:
        return _stage_conflict()
    try:
        owned_policies = _owned_query_policies(repository, mutation)
    except RuntimeError:
        return _stage_conflict()
    if len(owned_policies) > 1:
        return _stage_conflict()
    if not owned_policies:
        return WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.ABSENT)
    staged = owned_policies[0]
    if staged.effective_revision != mutation.target_revision:
        return _stage_conflict()
    try:
        intent = _intent_from_policy(staged)
        _update_query_policy_identity(
            mutation=mutation,
            target_revision=mutation.target_revision,
            intent=intent,
        )
    except RuntimeError:
        return _stage_conflict()
    if not _stage_manifest_matches(mutation=mutation, intent=intent):
        return _stage_conflict()
    if (
        staged.tenant_id != mutation.tenant_id
        or staged.workspace_id != mutation.workspace_id
    ):
        return _stage_conflict()
    expected = _expected_policy_row(
        mutation=mutation,
        target_revision=mutation.target_revision,
        intent=intent,
        updated_at=staged.updated_at,
    )
    if staged != expected:
        return _stage_conflict()
    return _stage_valid()


def _cleanup_staged_owned_row(
    *,
    repository: ManagedWorkspaceRepository,
    mutation: WorkspaceKnowledgeMutationRecord,
    inspection: WorkspaceKnowledgeStageInspection,
    reinspect,
) -> bool:
    if inspection.state is WorkspaceKnowledgeStageStateV1.ABSENT:
        return True
    if inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
        return False
    current_inspection = reinspect()
    if current_inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
        return False
    if current_inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
        return False
    if inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
        return False
    owned = _owned_query_policies(repository, mutation)
    if len(owned) != 1:
        return False
    staged = owned[0]
    if (
        mutation.target_revision is None
        or staged.effective_revision != mutation.target_revision
        or current_inspection.result_entity_id != _QUERY_POLICY_ENTITY_ID
    ):
        return False
    return repository.delete_knowledge_query_policy_version_if_match(staged)


class UpdateQueryPolicyMutationHandler:
    operation = WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY

    def find_existing_result(
        self,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        intent: object,
    ) -> WorkspaceKnowledgeExistingResult | None:
        if not isinstance(intent, UpdateQueryPolicyMutationIntent):
            raise ValueError("update_query_policy_intent_required")
        policy = configuration.query_policy
        if policy is None:
            return None
        if not _policy_matches_intent(policy, intent):
            return None
        return WorkspaceKnowledgeExistingResult(
            result_entity_type=_RESULT_ENTITY_TYPE,
            result_entity_id=_QUERY_POLICY_ENTITY_ID,
        )

    def stage(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
        target_revision: int,
        intent: object,
        now: datetime,
    ) -> WorkspaceKnowledgeStagedResult:
        if not isinstance(intent, UpdateQueryPolicyMutationIntent):
            raise ValueError("update_query_policy_intent_required")
        _update_query_policy_identity(
            mutation=mutation,
            target_revision=target_revision,
            intent=intent,
        )
        if not _stage_manifest_matches(mutation=mutation, intent=intent):
            raise RuntimeError("query_policy_stage_manifest_mismatch")
        policy = _expected_policy_row(
            mutation=mutation,
            target_revision=target_revision,
            intent=intent,
            updated_at=now,
        )
        if not repository.put_knowledge_query_policy_version_if_absent(policy):
            raise RuntimeError("query_policy_stage_conflict")
        return WorkspaceKnowledgeStagedResult(
            result_entity_type=_RESULT_ENTITY_TYPE,
            result_entity_id=_QUERY_POLICY_ENTITY_ID,
        )

    def inspect_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
    ) -> WorkspaceKnowledgeStageInspection:
        return _inspect_update_query_policy_staged(repository, mutation)

    def cleanup_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
        inspection: WorkspaceKnowledgeStageInspection,
    ) -> bool:
        return _cleanup_staged_owned_row(
            repository=repository,
            mutation=mutation,
            inspection=inspection,
            reinspect=lambda: _inspect_update_query_policy_staged(repository, mutation),
        )
