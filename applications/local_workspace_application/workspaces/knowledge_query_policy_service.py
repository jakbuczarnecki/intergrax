# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Workspace Query Policy lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import ValidationError

from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    normalize_update_query_policy_request_hash,
    query_policy_stage_manifest_hash,
    semantic_identity_hash_for_query_policy,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveResultRetentionV1,
    QueryPolicyModeV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationOutcomeV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
    WorkspaceQueryPolicy,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
    WorkspaceKnowledgeConfigurationMutationError,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
    WorkspaceKnowledgeMutationExecutionResult,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_configuration_validation import (
    validate_configuration_idempotency_hash,
)
from local_workspace_application.workspaces.knowledge_query_policy_handlers import (
    UpdateQueryPolicyMutationIntent,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_RESULT_TYPE = "query_policy"
_QUERY_POLICY_ENTITY_ID = "query-policy"
_VALIDATION_MUTATION_ID = "query-policy-validation-placeholder"
_VALIDATION_NOW = datetime(2000, 1, 1, 0, 0, 0, tzinfo=UTC)


class WorkspaceQueryPolicyError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class UpdateWorkspaceQueryPolicyCommand:
    tenant_id: str
    workspace_id: str
    mode: QueryPolicyModeV1
    allowed_connection_refs: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    max_live_calls: int
    max_total_duration_ms: int
    max_result_items: int
    max_result_bytes: int
    live_result_retention: LiveResultRetentionV1
    expected_revision: int
    idempotency_key_hash: str


@dataclass(frozen=True, slots=True)
class UpdateWorkspaceQueryPolicyResult:
    policy: WorkspaceQueryPolicy
    configuration_revision: int
    disposition: WorkspaceKnowledgeMutationExecutionDispositionV1
    updated_policy: bool
    mutation: WorkspaceKnowledgeMutationRecord


def _incomplete() -> WorkspaceQueryPolicyError:
    return WorkspaceQueryPolicyError("query_policy_projection_incomplete")


def _validate_and_normalize_intent(
    *,
    tenant_id: str,
    workspace_id: str,
    mode: object,
    allowed_connection_refs: tuple[str, ...],
    allowed_capability_ids: tuple[str, ...],
    max_live_calls: int,
    max_total_duration_ms: int,
    max_result_items: int,
    max_result_bytes: int,
    live_result_retention: object,
) -> UpdateQueryPolicyMutationIntent:
    if not isinstance(mode, QueryPolicyModeV1):
        raise WorkspaceQueryPolicyError("query_policy_mode_unsupported")
    if not isinstance(live_result_retention, LiveResultRetentionV1):
        raise WorkspaceQueryPolicyError("query_policy_invalid")
    try:
        validated = WorkspaceQueryPolicy(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            mode=mode,
            allowed_connection_refs=allowed_connection_refs,
            allowed_capability_ids=allowed_capability_ids,
            max_live_calls=max_live_calls,
            max_total_duration_ms=max_total_duration_ms,
            max_result_items=max_result_items,
            max_result_bytes=max_result_bytes,
            live_result_retention=live_result_retention,
            mutation_id=_VALIDATION_MUTATION_ID,
            effective_revision=1,
            updated_at=_VALIDATION_NOW,
        )
    except (ValidationError, ValueError):
        raise WorkspaceQueryPolicyError("query_policy_invalid") from None
    return UpdateQueryPolicyMutationIntent(
        mode=validated.mode,
        allowed_connection_refs=validated.allowed_connection_refs,
        allowed_capability_ids=validated.allowed_capability_ids,
        max_live_calls=validated.max_live_calls,
        max_total_duration_ms=validated.max_total_duration_ms,
        max_result_items=validated.max_result_items,
        max_result_bytes=validated.max_result_bytes,
        live_result_retention=validated.live_result_retention,
    )


def _policy_hashes(
    *,
    tenant_id: str,
    workspace_id: str,
    intent: UpdateQueryPolicyMutationIntent,
) -> tuple[str, str]:
    request_hash = normalize_update_query_policy_request_hash(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )
    semantic_hash = semantic_identity_hash_for_query_policy(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )
    return request_hash, semantic_hash


def _stage_manifest_hash(
    *,
    tenant_id: str,
    workspace_id: str,
    intent: UpdateQueryPolicyMutationIntent,
) -> str:
    return query_policy_stage_manifest_hash(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )


def _highest_policy(
    versions: list[WorkspaceQueryPolicy],
    *,
    revision: int,
) -> WorkspaceQueryPolicy | None:
    matches = [item for item in versions if item.effective_revision <= revision]
    if not matches:
        return None
    top = max(matches, key=lambda item: item.effective_revision)
    if sum(1 for item in matches if item.effective_revision == top.effective_revision) > 1:
        raise _incomplete()
    return top


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


def _resolve_historical_policy(
    repository: ManagedWorkspaceRepository,
    *,
    result: WorkspaceKnowledgeMutationExecutionResult,
    tenant_id: str,
    workspace_id: str,
    intent: UpdateQueryPolicyMutationIntent,
    request_hash: str,
    semantic_hash: str,
) -> WorkspaceQueryPolicy:
    mutation = result.mutation
    if (
        mutation.normalized_request_hash != request_hash
        or mutation.semantic_identity_hash != semantic_hash
        or mutation.result_entity_type != _RESULT_TYPE
        or mutation.result_entity_id != _QUERY_POLICY_ENTITY_ID
        or mutation.committed_revision != result.configuration_revision
    ):
        raise _incomplete()
    policy = _highest_policy(
        repository.list_knowledge_query_policy_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ),
        revision=result.configuration_revision,
    )
    if policy is None:
        raise _incomplete()
    if policy.tenant_id != tenant_id or policy.workspace_id != workspace_id:
        raise _incomplete()
    if not _policy_matches_intent(policy, intent):
        raise _incomplete()
    if mutation.outcome is WorkspaceKnowledgeMutationOutcomeV1.APPLIED:
        if (
            mutation.target_revision != mutation.committed_revision
            or policy.effective_revision != mutation.target_revision
            or policy.mutation_id != mutation.mutation_id
        ):
            raise _incomplete()
    elif mutation.outcome is WorkspaceKnowledgeMutationOutcomeV1.EXISTING_RESULT:
        if (
            mutation.target_revision is not None
            or policy.effective_revision > mutation.committed_revision
        ):
            raise _incomplete()
    else:
        raise _incomplete()
    return policy


def _updated_policy_flag(
    disposition: WorkspaceKnowledgeMutationExecutionDispositionV1,
) -> bool:
    return disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED


class WorkspaceQueryPolicyService:
    def __init__(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        configuration_service: WorkspaceKnowledgeConfigurationService,
        mutation_engine: WorkspaceKnowledgeConfigurationMutationEngine,
    ) -> None:
        self._repository = repository
        self._configuration_service = configuration_service
        self._mutation_engine = mutation_engine

    def update_query_policy(
        self,
        command: UpdateWorkspaceQueryPolicyCommand,
    ) -> UpdateWorkspaceQueryPolicyResult:
        validate_configuration_idempotency_hash(command.idempotency_key_hash)
        tenant_id = command.tenant_id.strip()
        workspace_id = command.workspace_id.strip()
        intent = _validate_and_normalize_intent(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            mode=command.mode,
            allowed_connection_refs=command.allowed_connection_refs,
            allowed_capability_ids=command.allowed_capability_ids,
            max_live_calls=command.max_live_calls,
            max_total_duration_ms=command.max_total_duration_ms,
            max_result_items=command.max_result_items,
            max_result_bytes=command.max_result_bytes,
            live_result_retention=command.live_result_retention,
        )
        request_hash, semantic_hash = _policy_hashes(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            intent=intent,
        )
        manifest_hash = _stage_manifest_hash(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            intent=intent,
        )
        existing = self._repository.get_knowledge_configuration_mutation(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY,
            idempotency_key_hash=command.idempotency_key_hash,
        )
        if existing is not None and existing.normalized_request_hash != request_hash:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_idempotency_conflict"
            )
        if (
            existing is not None
            and existing.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
        ):
            replay = self._mutation_engine.execute(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY,
                expected_revision=command.expected_revision,
                idempotency_key_hash=command.idempotency_key_hash,
                normalized_request_hash=request_hash,
                semantic_identity_hash=semantic_hash,
                stage_manifest_hash=existing.stage_manifest_hash,
                intent=intent,
            )
            if (
                replay.disposition
                is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
            ):
                policy = _resolve_historical_policy(
                    self._repository,
                    result=replay,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    intent=intent,
                    request_hash=request_hash,
                    semantic_hash=semantic_hash,
                )
                return UpdateWorkspaceQueryPolicyResult(
                    policy=policy,
                    configuration_revision=replay.configuration_revision,
                    disposition=replay.disposition,
                    updated_policy=False,
                    mutation=replay.mutation,
                )
        configuration = self._configuration_service.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            raise WorkspaceQueryPolicyError("workspace_not_found")
        result = self._mutation_engine.execute(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY,
            expected_revision=command.expected_revision,
            idempotency_key_hash=command.idempotency_key_hash,
            normalized_request_hash=request_hash,
            semantic_identity_hash=semantic_hash,
            stage_manifest_hash=manifest_hash,
            intent=intent,
        )
        policy = _resolve_historical_policy(
            self._repository,
            result=result,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            intent=intent,
            request_hash=request_hash,
            semantic_hash=semantic_hash,
        )
        return UpdateWorkspaceQueryPolicyResult(
            policy=policy,
            configuration_revision=result.configuration_revision,
            disposition=result.disposition,
            updated_policy=_updated_policy_flag(result.disposition),
            mutation=result.mutation,
        )
