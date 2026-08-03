# © Artur Czarnecki. All rights reserved.

"""Tests for WorkspaceQueryPolicyService and query policy mutation handlers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    normalize_update_query_policy_request_hash,
    query_policy_stage_manifest_hash,
    semantic_identity_hash_for_query_policy,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveResultRetentionV1,
    QueryPolicyModeV1,
    WorkspaceKnowledgeConfigurationHead,
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
    WorkspaceKnowledgeMutationRecoveryDispositionV1,
    WorkspaceKnowledgeStageStateV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_query_policy_handlers import (
    UpdateQueryPolicyMutationHandler,
    UpdateQueryPolicyMutationIntent,
)
from local_workspace_application.workspaces.knowledge_query_policy_service import (
    UpdateWorkspaceQueryPolicyCommand,
    WorkspaceQueryPolicyError,
    WorkspaceQueryPolicyService,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_TENANT, _WORKSPACE = "tenant-a", "workspace-1"
_CONN_A, _CONN_B = "conn.a", "conn.b"
_CAP_A, _CAP_B = "cap.read", "cap.search"
_SHA256_A, _SHA256_B, _SHA256_C, _SHA256_D, _SHA256_E = (
    "a" * 64,
    "b" * 64,
    "c" * 64,
    "d" * 64,
    "e" * 64,
)
_HANDLER = UpdateQueryPolicyMutationHandler()
_ENTITY_ID = "query-policy"


def _workspace() -> Workspace:
    return Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Workspace",
        status=WorkspaceStatus.ACTIVE,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _indexed_intent() -> UpdateQueryPolicyMutationIntent:
    return UpdateQueryPolicyMutationIntent(
        mode=QueryPolicyModeV1.INDEXED_ONLY,
        allowed_connection_refs=(),
        allowed_capability_ids=(),
        max_live_calls=0,
        max_total_duration_ms=30_000,
        max_result_items=50,
        max_result_bytes=1_048_576,
        live_result_retention=LiveResultRetentionV1.EPHEMERAL,
    )


def _live_intent() -> UpdateQueryPolicyMutationIntent:
    return UpdateQueryPolicyMutationIntent(
        mode=QueryPolicyModeV1.LIVE_ONLY,
        allowed_connection_refs=(_CONN_A,),
        allowed_capability_ids=(_CAP_A,),
        max_live_calls=3,
        max_total_duration_ms=60_000,
        max_result_items=100,
        max_result_bytes=2_097_152,
        live_result_retention=LiveResultRetentionV1.RECEIPT_ONLY,
    )


def _build_stack(*, mutation_ids: list[str] | None = None):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_workspace(_workspace())
    lookup = ManagedWorkspaceService(repo)
    config = WorkspaceKnowledgeConfigurationService(repo, lookup)
    ids = mutation_ids or [f"mutation-{i}" for i in range(1, 20)]
    idx = {"i": 0}

    def _next_id() -> str:
        value = ids[idx["i"]]
        idx["i"] = min(idx["i"] + 1, len(ids) - 1)
        return value

    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        config,
        {WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY: _HANDLER},
        clock=lambda: _NOW,
        mutation_id_factory=_next_id,
    )
    service = WorkspaceQueryPolicyService(
        repository=repo,
        configuration_service=config,
        mutation_engine=engine,
    )
    return service, repo, config, engine


def _cmd(**overrides: object) -> UpdateWorkspaceQueryPolicyCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "mode": QueryPolicyModeV1.INDEXED_ONLY,
        "allowed_connection_refs": (),
        "allowed_capability_ids": (),
        "max_live_calls": 0,
        "max_total_duration_ms": 30_000,
        "max_result_items": 50,
        "max_result_bytes": 1_048_576,
        "live_result_retention": LiveResultRetentionV1.EPHEMERAL,
        "expected_revision": 0,
        "idempotency_key_hash": _SHA256_A,
    }
    payload.update(overrides)
    return UpdateWorkspaceQueryPolicyCommand(**payload)


def _live_cmd(**overrides: object) -> UpdateWorkspaceQueryPolicyCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "mode": QueryPolicyModeV1.LIVE_ONLY,
        "allowed_connection_refs": (_CONN_A,),
        "allowed_capability_ids": (_CAP_A,),
        "max_live_calls": 3,
        "max_total_duration_ms": 60_000,
        "max_result_items": 100,
        "max_result_bytes": 2_097_152,
        "live_result_retention": LiveResultRetentionV1.RECEIPT_ONLY,
        "expected_revision": 0,
        "idempotency_key_hash": _SHA256_A,
    }
    payload.update(overrides)
    return UpdateWorkspaceQueryPolicyCommand(**payload)


def _hashes(intent: UpdateQueryPolicyMutationIntent) -> tuple[str, str]:
    request = normalize_update_query_policy_request_hash(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )
    semantic = semantic_identity_hash_for_query_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )
    return request, semantic


def _manifest(intent: UpdateQueryPolicyMutationIntent) -> str:
    return query_policy_stage_manifest_hash(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        mode=intent.mode,
        allowed_connection_refs=intent.allowed_connection_refs,
        allowed_capability_ids=intent.allowed_capability_ids,
        max_live_calls=intent.max_live_calls,
        max_total_duration_ms=intent.max_total_duration_ms,
        max_result_items=intent.max_result_items,
        max_result_bytes=intent.max_result_bytes,
        live_result_retention=intent.live_result_retention,
    )


def _policy_row(
    *,
    mutation_id: str,
    revision: int,
    intent: UpdateQueryPolicyMutationIntent | None = None,
    **overrides: object,
) -> WorkspaceQueryPolicy:
    resolved = intent or _indexed_intent()
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "mode": resolved.mode,
        "allowed_connection_refs": resolved.allowed_connection_refs,
        "allowed_capability_ids": resolved.allowed_capability_ids,
        "max_live_calls": resolved.max_live_calls,
        "max_total_duration_ms": resolved.max_total_duration_ms,
        "max_result_items": resolved.max_result_items,
        "max_result_bytes": resolved.max_result_bytes,
        "live_result_retention": resolved.live_result_retention,
        "mutation_id": mutation_id,
        "effective_revision": revision,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceQueryPolicy(**payload)


def _seed_head(repo, *, committed_revision: int = 0) -> None:
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=committed_revision,
            updated_at=_NOW,
        )
    )


def _pending_head(repo, *, revision: int, mutation_id: str) -> None:
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    if head is None:
        _seed_head(repo, committed_revision=max(0, revision - 1))
        head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    repo.replace_knowledge_configuration_head_if_match(
        expected=head,
        replacement=head.model_copy(
            update={
                "pending_revision": revision,
                "pending_mutation_id": mutation_id,
                "updated_at": _NOW,
            }
        ),
    )


_MANIFEST_DEFAULT = object()


def _create_mutation(
    repo,
    *,
    revision: int,
    intent: UpdateQueryPolicyMutationIntent,
    mutation_id: str = "mutation-policy",
    idempotency_key_hash: str = _SHA256_A,
    stage_manifest_hash: str | None | object = _MANIFEST_DEFAULT,
) -> WorkspaceKnowledgeMutationRecord:
    request_hash, semantic_hash = _hashes(intent)
    if stage_manifest_hash is _MANIFEST_DEFAULT:
        manifest = _manifest(intent)
    else:
        manifest = stage_manifest_hash
    mutation = WorkspaceKnowledgeMutationRecord(
        mutation_id=mutation_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY,
        idempotency_key_hash=idempotency_key_hash,
        normalized_request_hash=request_hash,
        semantic_identity_hash=semantic_hash,
        stage_manifest_hash=manifest,
        target_revision=revision,
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        result_entity_type="query_policy",
        result_entity_id=_ENTITY_ID,
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    _pending_head(repo, revision=revision, mutation_id=mutation_id)
    return mutation


def _policy_partition() -> str:
    return f"lkw.managed_workspace:{_TENANT}:knowledge_configuration_query_policy"


def _policy_row_key(revision: int) -> str:
    return f"{_WORKSPACE}:{_ENTITY_ID}:rev:{revision:020d}"


def _patch_store_field(repo, *, partition: str, row_key: str, updates: dict) -> None:
    record = repo.document_store.get(partition, row_key)
    assert record is not None
    data = dict(record.data)
    data.update({k: v.value if hasattr(v, "value") else v for k, v in updates.items()})
    repo.document_store.put(DocumentRecord(partition_key=partition, row_key=row_key, data=data))


def _replace_owned_policy(repo, mutation: WorkspaceKnowledgeMutationRecord, **updates: object) -> None:
    policy = next(
        item
        for item in repo.list_knowledge_query_policy_versions(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        if item.mutation_id == mutation.mutation_id
    )
    row_key = _policy_row_key(policy.effective_revision)
    if {"tenant_id", "workspace_id", "mode"} & updates.keys():
        _patch_store_field(repo, partition=_policy_partition(), row_key=row_key, updates=updates)
        return
    repo.delete_knowledge_query_policy_version_if_match(policy)
    repo.put_knowledge_query_policy_version_if_absent(policy.model_copy(update=updates))


def _assert_no_policy_side_effects(repo: ManagedWorkspaceRepository) -> None:
    assert repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE) == []
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is None or head.committed_revision == 0


# --- Validation and normalization ---


def test_valid_indexed_only_policy() -> None:
    service, repo, config, _ = _build_stack()
    result = service.update_query_policy(_cmd())
    assert result.updated_policy is True
    assert result.policy.mode is QueryPolicyModeV1.INDEXED_ONLY
    assert config.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE).query_policy == result.policy
    assert len(repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)) == 1


def test_valid_live_only_policy() -> None:
    service, _, config, _ = _build_stack()
    result = service.update_query_policy(_live_cmd())
    assert result.policy.mode is QueryPolicyModeV1.LIVE_ONLY
    assert result.policy.allowed_connection_refs == (_CONN_A,)
    assert config.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE).query_policy == result.policy


def test_connection_refs_sorted_and_deduplicated() -> None:
    service, _, _, _ = _build_stack()
    result = service.update_query_policy(
        _live_cmd(allowed_connection_refs=(f" {_CONN_B} ", _CONN_A, _CONN_A))
    )
    assert result.policy.allowed_connection_refs == (_CONN_A, _CONN_B)


def test_capability_ids_sorted_and_deduplicated() -> None:
    service, _, _, _ = _build_stack()
    result = service.update_query_policy(
        _live_cmd(allowed_capability_ids=(f" {_CAP_B} ", _CAP_A, _CAP_A))
    )
    assert result.policy.allowed_capability_ids == (_CAP_A, _CAP_B)


def test_unsupported_mode_rejected() -> None:
    service, repo, _, _ = _build_stack()
    with pytest.raises(WorkspaceQueryPolicyError, match="query_policy_mode_unsupported"):
        service.update_query_policy(_cmd(mode="hybrid"))  # type: ignore[arg-type]
    _assert_no_policy_side_effects(repo)


@pytest.mark.parametrize(
    ("overrides",),
    [
        ({"allowed_connection_refs": (_CONN_A,)},),
        ({"allowed_capability_ids": (_CAP_A,)},),
        ({"max_live_calls": 1},),
        ({"live_result_retention": LiveResultRetentionV1.RECEIPT_ONLY},),
    ],
)
def test_indexed_only_cross_field_rejected(overrides: dict) -> None:
    service, repo, _, _ = _build_stack()
    with pytest.raises(WorkspaceQueryPolicyError, match="query_policy_invalid"):
        service.update_query_policy(_cmd(**overrides))
    _assert_no_policy_side_effects(repo)


@pytest.mark.parametrize(
    ("overrides",),
    [
        ({"allowed_connection_refs": ()},),
        ({"allowed_capability_ids": ()},),
        ({"max_live_calls": 0},),
    ],
)
def test_live_only_cross_field_rejected(overrides: dict) -> None:
    service, repo, _, _ = _build_stack()
    with pytest.raises(WorkspaceQueryPolicyError, match="query_policy_invalid"):
        service.update_query_policy(_live_cmd(**overrides))
    _assert_no_policy_side_effects(repo)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_live_calls", -1),
        ("max_live_calls", 51),
        ("max_total_duration_ms", 0),
        ("max_total_duration_ms", 300_001),
        ("max_result_items", 0),
        ("max_result_items", 501),
        ("max_result_bytes", 0),
        ("max_result_bytes", 16_777_217),
    ],
)
def test_numeric_bounds_rejected(field: str, value: int) -> None:
    service, repo, _, _ = _build_stack()
    with pytest.raises(WorkspaceQueryPolicyError, match="query_policy_invalid"):
        service.update_query_policy(_live_cmd(**{field: value}))
    _assert_no_policy_side_effects(repo)


def test_semantic_hash_ignores_allowlist_order() -> None:
    first = semantic_identity_hash_for_query_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        mode=QueryPolicyModeV1.LIVE_ONLY,
        allowed_connection_refs=(_CONN_B, _CONN_A),
        allowed_capability_ids=(_CAP_B, _CAP_A),
        max_live_calls=2,
        max_total_duration_ms=30_000,
        max_result_items=50,
        max_result_bytes=1_048_576,
        live_result_retention=LiveResultRetentionV1.EPHEMERAL,
    )
    second = semantic_identity_hash_for_query_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        mode=QueryPolicyModeV1.LIVE_ONLY,
        allowed_connection_refs=(_CONN_A, _CONN_B),
        allowed_capability_ids=(_CAP_A, _CAP_B),
        max_live_calls=2,
        max_total_duration_ms=30_000,
        max_result_items=50,
        max_result_bytes=1_048_576,
        live_result_retention=LiveResultRetentionV1.EPHEMERAL,
    )
    assert first == second


# --- Lifecycle ---


def test_first_policy_creation() -> None:
    service, repo, config, _ = _build_stack()
    result = service.update_query_policy(_cmd())
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert result.configuration_revision == 1
    assert config.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE).query_policy is not None
    assert len(repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)) == 1


def test_replacement_indexed_to_live() -> None:
    service, repo, config, _ = _build_stack()
    first = service.update_query_policy(_cmd())
    second = service.update_query_policy(
        _live_cmd(expected_revision=first.configuration_revision, idempotency_key_hash=_SHA256_B)
    )
    assert second.updated_policy is True
    assert second.policy.mode is QueryPolicyModeV1.LIVE_ONLY
    versions = repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(versions) == 2
    assert config.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE).query_policy == second.policy


def test_replacement_live_to_indexed() -> None:
    service, repo, _, _ = _build_stack()
    first = service.update_query_policy(_live_cmd())
    second = service.update_query_policy(
        _cmd(expected_revision=first.configuration_revision, idempotency_key_hash=_SHA256_B)
    )
    assert second.policy.mode is QueryPolicyModeV1.INDEXED_ONLY
    revisions = sorted(v.effective_revision for v in repo.list_knowledge_query_policy_versions(
        tenant_id=_TENANT, workspace_id=_WORKSPACE,
    ))
    assert revisions == [1, 2]


def test_same_logical_entity_id_across_revisions() -> None:
    service, repo, _, _ = _build_stack()
    first = service.update_query_policy(_cmd())
    service.update_query_policy(
        _live_cmd(expected_revision=first.configuration_revision, idempotency_key_hash=_SHA256_B)
    )
    assert all(
        v.tenant_id == _TENANT and v.workspace_id == _WORKSPACE
        for v in repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    )


def test_prior_revision_preserved() -> None:
    service, repo, _, _ = _build_stack()
    first = service.update_query_policy(_cmd())
    service.update_query_policy(
        _live_cmd(expected_revision=first.configuration_revision, idempotency_key_hash=_SHA256_B)
    )
    first_revision = repo.get_knowledge_query_policy_version(
        tenant_id=_TENANT, workspace_id=_WORKSPACE, effective_revision=1,
    )
    assert first_revision is not None
    assert first_revision.mode is QueryPolicyModeV1.INDEXED_ONLY


def test_configuration_revision_increments_once_per_applied_update() -> None:
    service, _, _, _ = _build_stack()
    first = service.update_query_policy(_cmd())
    second = service.update_query_policy(
        _live_cmd(expected_revision=first.configuration_revision, idempotency_key_hash=_SHA256_B)
    )
    assert first.configuration_revision == 1
    assert second.configuration_revision == 2


def test_projection_selects_highest_committed_policy() -> None:
    service, repo, config, _ = _build_stack()
    first = service.update_query_policy(_cmd())
    second = service.update_query_policy(
        _live_cmd(expected_revision=first.configuration_revision, idempotency_key_hash=_SHA256_B)
    )
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    repo.replace_knowledge_configuration_head_if_match(
        expected=head,
        replacement=head.model_copy(update={"committed_revision": 1, "updated_at": _NOW}),
    )
    projected = config.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert projected is not None
    assert projected.query_policy is not None
    assert projected.query_policy.effective_revision == 1
    assert projected.query_policy.mode is QueryPolicyModeV1.INDEXED_ONLY
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    repo.replace_knowledge_configuration_head_if_match(
        expected=head,
        replacement=head.model_copy(update={"committed_revision": second.configuration_revision, "updated_at": _NOW}),
    )
    projected = config.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert projected is not None
    assert projected.query_policy == second.policy


def test_policy_update_does_not_change_unrelated_configuration_children() -> None:
    service, repo, config, _ = _build_stack()
    before = config.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert before is not None
    result = service.update_query_policy(_cmd())
    after = config.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert after is not None
    assert after.connection_attachments == before.connection_attachments
    assert after.indexed_sources == before.indexed_sources
    assert after.live_access_bindings == before.live_access_bindings
    assert after.query_policy == result.policy
    assert repo.list_sources(tenant_id=_TENANT, workspace_id=_WORKSPACE) == []


# --- Idempotency ---


def test_committed_replay() -> None:
    service, _, _, _ = _build_stack()
    first = service.update_query_policy(_cmd())
    replay = service.update_query_policy(
        _cmd(expected_revision=first.configuration_revision, idempotency_key_hash=_SHA256_A)
    )
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay.updated_policy is False
    assert replay.policy == first.policy


def test_idempotency_conflict_before_revision_mismatch() -> None:
    service, _, _, _ = _build_stack()
    service.update_query_policy(_cmd())
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        service.update_query_policy(
            _live_cmd(expected_revision=0, idempotency_key_hash=_SHA256_A)
        )
    assert exc.value.error_code == "configuration_idempotency_conflict"


def test_semantic_noop_with_different_key() -> None:
    service, repo, _, _ = _build_stack()
    first = service.update_query_policy(_cmd())
    noop = service.update_query_policy(
        _cmd(expected_revision=first.configuration_revision, idempotency_key_hash=_SHA256_B)
    )
    assert noop.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT
    assert noop.updated_policy is False
    assert len(repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)) == 1
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.committed_revision == first.configuration_revision


def test_semantic_noop_mutation_outcome() -> None:
    service, _, _, _ = _build_stack()
    first = service.update_query_policy(_cmd())
    noop = service.update_query_policy(
        _cmd(expected_revision=first.configuration_revision, idempotency_key_hash=_SHA256_B)
    )
    assert noop.mutation.outcome is WorkspaceKnowledgeMutationOutcomeV1.EXISTING_RESULT
    assert noop.mutation.target_revision is None
    assert noop.mutation.committed_revision == first.configuration_revision


# --- Recovery ---


def test_complete_prepared_recovery_commits() -> None:
    _, repo, config, engine = _build_stack(mutation_ids=["mutation-policy"])
    _seed_head(repo)
    intent = _indexed_intent()
    mutation = _create_mutation(repo, revision=1, intent=intent)
    _HANDLER.stage(
        repository=repo,
        mutation=mutation,
        target_revision=mutation.target_revision,
        intent=intent,
        now=_NOW,
    )
    recovery = engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.committed_revision == 1
    assert config.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE).query_policy is not None


def test_missing_manifest_fail_closed() -> None:
    _, repo, _, engine = _build_stack(mutation_ids=["mutation-policy"])
    _seed_head(repo)
    intent = _indexed_intent()
    mutation = _create_mutation(repo, revision=1, intent=intent, stage_manifest_hash=None)
    repo.put_knowledge_query_policy_version_if_absent(
        _policy_row(mutation_id=mutation.mutation_id, revision=1, intent=intent)
    )
    assert _HANDLER.inspect_staged(repository=repo, mutation=mutation).state is (
        WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError, match="configuration_recovery_required"):
        engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_stage_manifest_mismatch_blocks_stage() -> None:
    _, repo, _, _ = _build_stack(mutation_ids=["mutation-policy"])
    _seed_head(repo)
    intent = _indexed_intent()
    mutation = _create_mutation(repo, revision=1, intent=intent, stage_manifest_hash="f" * 64)
    with pytest.raises(RuntimeError, match="query_policy_stage_manifest_mismatch"):
        _HANDLER.stage(
            repository=repo,
            mutation=mutation,
            target_revision=mutation.target_revision,
            intent=intent,
            now=_NOW,
        )
    assert repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE) == []


@pytest.mark.parametrize(
    "field, corrupt_value",
    [
        ("mode", QueryPolicyModeV1.INDEXED_ONLY),
        ("allowed_connection_refs", (_CONN_B,)),
        ("allowed_capability_ids", (_CAP_B,)),
        ("max_live_calls", 9),
        ("max_total_duration_ms", 90_000),
        ("max_result_items", 99),
        ("max_result_bytes", 3_000_000),
        ("live_result_retention", LiveResultRetentionV1.EPHEMERAL),
        ("tenant_id", "tenant-other"),
        ("workspace_id", "workspace-other"),
    ],
)
def test_corrupt_staged_field_blocks_recovery(field: str, corrupt_value: object) -> None:
    service, repo, _, engine = _build_stack(mutation_ids=["mutation-1", "mutation-2"])
    service.update_query_policy(_cmd())
    intent = _live_intent()
    mutation = _create_mutation(
        repo,
        revision=2,
        intent=intent,
        mutation_id="mutation-2",
        idempotency_key_hash=_SHA256_B,
    )
    _HANDLER.stage(
        repository=repo,
        mutation=mutation,
        target_revision=mutation.target_revision,
        intent=intent,
        now=_NOW,
    )
    _replace_owned_policy(repo, mutation, **{field: corrupt_value})
    assert _HANDLER.inspect_staged(repository=repo, mutation=mutation).state is (
        WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError, match="configuration_recovery_required"):
        engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.pending_mutation_id == mutation.mutation_id
    reloaded = repo.get_knowledge_configuration_mutation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY,
        idempotency_key_hash=mutation.idempotency_key_hash,
    )
    assert reloaded is not None
    if field in {"tenant_id", "workspace_id"}:
        assert reloaded.status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED
    else:
        assert reloaded.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED
    assert repo.get_knowledge_query_policy_version(
        tenant_id=_TENANT, workspace_id=_WORKSPACE, effective_revision=1,
    ) is not None


def test_duplicate_owned_rows_conflict() -> None:
    _, repo, _, engine = _build_stack(mutation_ids=["mutation-policy"])
    _seed_head(repo)
    intent = _indexed_intent()
    mutation = _create_mutation(repo, revision=1, intent=intent)
    repo.put_knowledge_query_policy_version_if_absent(
        _policy_row(mutation_id=mutation.mutation_id, revision=1, intent=intent)
    )
    repo.put_knowledge_query_policy_version_if_absent(
        _policy_row(mutation_id=mutation.mutation_id, revision=2, intent=intent)
    )
    assert _HANDLER.inspect_staged(repository=repo, mutation=mutation).state is (
        WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError, match="configuration_recovery_required"):
        engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_pristine_abort_preserves_prior_policy() -> None:
    service, repo, _, engine = _build_stack(mutation_ids=["mutation-1", "mutation-2"])
    first = service.update_query_policy(_cmd())
    mutation = _create_mutation(
        repo,
        revision=first.configuration_revision + 1,
        intent=_live_intent(),
        mutation_id="mutation-2",
        idempotency_key_hash=_SHA256_B,
    )
    assert _HANDLER.inspect_staged(repository=repo, mutation=mutation).state is (
        WorkspaceKnowledgeStageStateV1.ABSENT
    )
    recovery = engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    assert repo.get_knowledge_query_policy_version(
        tenant_id=_TENANT, workspace_id=_WORKSPACE, effective_revision=1,
    ) is not None
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.committed_revision == first.configuration_revision


def test_cleanup_deletes_exact_owned_row() -> None:
    _, repo, _, engine = _build_stack(mutation_ids=["mutation-policy"])
    _seed_head(repo)
    intent = _indexed_intent()
    mutation = _create_mutation(repo, revision=1, intent=intent)
    staged = _policy_row(mutation_id=mutation.mutation_id, revision=1, intent=intent)
    repo.put_knowledge_query_policy_version_if_absent(staged)
    inspection = _HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
    assert _HANDLER.cleanup_staged(repository=repo, mutation=mutation, inspection=inspection) is True
    assert repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE) == []
    recovery = engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED


# --- Historical proof ---


def test_committed_replay_resolves_historical_without_newer_revision() -> None:
    service, repo, _, _ = _build_stack()
    first = service.update_query_policy(_cmd())
    service.update_query_policy(
        _live_cmd(expected_revision=first.configuration_revision, idempotency_key_hash=_SHA256_B)
    )
    replay = service.update_query_policy(
        _cmd(expected_revision=2, idempotency_key_hash=_SHA256_A)
    )
    assert replay.policy.mode is QueryPolicyModeV1.INDEXED_ONLY
    assert replay.policy.effective_revision == 1
    assert len(repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)) == 2


def test_applied_result_rejects_wrong_mutation_ownership() -> None:
    service, repo, _, _ = _build_stack()
    result = service.update_query_policy(_cmd())
    mutation = result.mutation.model_copy(update={"mutation_id": "wrong-mutation"})
    from local_workspace_application.workspaces import knowledge_query_policy_service as svc_mod

    with pytest.raises(WorkspaceQueryPolicyError, match="query_policy_projection_incomplete"):
        svc_mod._resolve_historical_policy(
            repo,
            result=type("R", (), {
                "mutation": mutation,
                "configuration_revision": result.configuration_revision,
            })(),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            intent=_indexed_intent(),
            request_hash=mutation.normalized_request_hash,
            semantic_hash=mutation.semantic_identity_hash or "",
        )


def test_existing_result_rejects_target_revision() -> None:
    service, repo, _, _ = _build_stack()
    first = service.update_query_policy(_cmd())
    noop = service.update_query_policy(
        _cmd(expected_revision=first.configuration_revision, idempotency_key_hash=_SHA256_B)
    )
    bad = noop.mutation.model_copy(update={"target_revision": 99})
    from local_workspace_application.workspaces import knowledge_query_policy_service as svc_mod

    with pytest.raises(WorkspaceQueryPolicyError, match="query_policy_projection_incomplete"):
        svc_mod._resolve_historical_policy(
            repo,
            result=type("R", (), {"mutation": bad, "configuration_revision": noop.configuration_revision})(),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            intent=_indexed_intent(),
            request_hash=bad.normalized_request_hash,
            semantic_hash=bad.semantic_identity_hash or "",
        )


def test_wrong_result_type_projection_incomplete() -> None:
    service, repo, _, _ = _build_stack()
    result = service.update_query_policy(_cmd())
    bad = result.mutation.model_copy(update={"result_entity_type": "other"})
    from local_workspace_application.workspaces import knowledge_query_policy_service as svc_mod

    with pytest.raises(WorkspaceQueryPolicyError, match="query_policy_projection_incomplete"):
        svc_mod._resolve_historical_policy(
            repo,
            result=type("R", (), {"mutation": bad, "configuration_revision": result.configuration_revision})(),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            intent=_indexed_intent(),
            request_hash=bad.normalized_request_hash,
            semantic_hash=bad.semantic_identity_hash or "",
        )


def test_missing_historical_policy_projection_incomplete() -> None:
    service, repo, _, _ = _build_stack()
    result = service.update_query_policy(_cmd())
    repo.delete_knowledge_query_policy_version_if_match(result.policy)
    from local_workspace_application.workspaces import knowledge_query_policy_service as svc_mod

    with pytest.raises(WorkspaceQueryPolicyError, match="query_policy_projection_incomplete"):
        svc_mod._resolve_historical_policy(
            repo,
            result=type("R", (), {
                "mutation": result.mutation,
                "configuration_revision": result.configuration_revision,
            })(),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            intent=_indexed_intent(),
            request_hash=result.mutation.normalized_request_hash,
            semantic_hash=result.mutation.semantic_identity_hash or "",
        )
