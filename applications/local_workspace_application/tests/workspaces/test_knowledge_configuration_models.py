# © Artur Czarnecki. All rights reserved.

"""Unit tests for Workspace Knowledge Configuration domain models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    KnowledgeAudienceEligibilityV1,
    LiveResultRetentionV1,
    QueryPolicyModeV1,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeConfigurationHead,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationOutcomeV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicy,
)
from local_workspace_application.workspaces.models import (
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)

pytestmark = pytest.mark.unit

_NOW = datetime.now(UTC)
_SHA256 = "a" * 64
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_MUTATION = "mutation-1"


def _connection_attachment(**overrides: object) -> WorkspaceConnectionAttachment:
    payload = {
        "attachment_id": "att-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": "conn.primary",
        "safe_display_label": "Primary",
        "status": WorkspaceConnectionAttachmentStatusV1.ATTACHED,
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceConnectionAttachment(**payload)


def _indexed_source_binding(**overrides: object) -> WorkspaceIndexedSourceBinding:
    payload = {
        "indexed_source_binding_id": "idx-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "knowledge_source_binding_ref": "ksb-1",
        "source_id": "source-1",
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "semantic_identity_hash": _SHA256,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceIndexedSourceBinding(**payload)


def _live_access_binding(**overrides: object) -> WorkspaceLiveAccessBinding:
    payload = {
        "live_access_binding_id": "live-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": "conn.live",
        "allowed_capability_ids": ("cap.read",),
        "derived_provider_id": "provider-1",
        "derived_integration_kind": IntegrationCategory.WIKI_KNOWLEDGE,
        "derived_safe_display_label": "Wiki",
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "semantic_identity_hash": _SHA256,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceLiveAccessBinding(**payload)


def _query_policy(**overrides: object) -> WorkspaceQueryPolicy:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceQueryPolicy(**payload)


def _mutation_record(**overrides: object) -> WorkspaceKnowledgeMutationRecord:
    payload = {
        "mutation_id": _MUTATION,
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "operation": WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION,
        "idempotency_key_hash": _SHA256,
        "normalized_request_hash": _SHA256,
        "status": WorkspaceKnowledgeMutationStatusV1.RESERVED,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceKnowledgeMutationRecord(**payload)


def _workspace_source(**overrides: object) -> WorkspaceSource:
    payload = {
        "source_id": "source-1",
        "workspace_id": _WORKSPACE,
        "tenant_id": _TENANT,
        "created_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceSource(**payload)


# --- General model behavior ---


def test_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _connection_attachment(unknown_field="x")


def test_models_are_frozen() -> None:
    attachment = _connection_attachment()
    with pytest.raises(ValidationError):
        attachment.status = WorkspaceConnectionAttachmentStatusV1.DETACHED


def test_required_identifiers_reject_empty_values() -> None:
    with pytest.raises(ValidationError):
        _connection_attachment(attachment_id="")


@pytest.mark.parametrize(
    "hash_value",
    [
        "not-hex",
        "A" * 64,
        "a" * 63,
        "a" * 65,
    ],
)
def test_sha256_fields_reject_invalid_values(hash_value: str) -> None:
    with pytest.raises(ValidationError):
        _indexed_source_binding(semantic_identity_hash=hash_value)


# --- Connection attachment ---


def test_valid_connection_attachment_accepted() -> None:
    attachment = _connection_attachment()
    assert attachment.connection_ref == "conn.primary"


def test_invalid_connection_ref_rejected() -> None:
    with pytest.raises(ValidationError):
        _connection_attachment(connection_ref="-invalid")


def test_connection_attachment_rejects_credential_like_extras() -> None:
    with pytest.raises(ValidationError):
        _connection_attachment(credential_ref="secret")


# --- Indexed source binding ---


def test_valid_indexed_source_binding_accepted() -> None:
    binding = _indexed_source_binding()
    assert binding.sync_mode is IndexedSourceSyncModeV1.INCREMENTAL
    assert binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE


def test_indexed_source_binding_rejects_provider_identity_extras() -> None:
    with pytest.raises(ValidationError):
        _indexed_source_binding(provider_id="slack")


# --- Live access binding ---


def test_live_access_binding_normalizes_capability_ids() -> None:
    binding = _live_access_binding(
        allowed_capability_ids=(" cap.z ", "cap.a", "cap.a"),
    )
    assert binding.allowed_capability_ids == ("cap.a", "cap.z")


def test_live_access_binding_rejects_empty_capability_list() -> None:
    with pytest.raises(ValidationError):
        _live_access_binding(allowed_capability_ids=())


def test_live_access_binding_rejects_blank_capability_id() -> None:
    with pytest.raises(ValidationError):
        _live_access_binding(allowed_capability_ids=(" ",))


def test_live_access_binding_rejects_invalid_connection_ref() -> None:
    with pytest.raises(ValidationError):
        _live_access_binding(connection_ref="")


# --- Query policy ---


def test_valid_indexed_only_query_policy_accepted() -> None:
    policy = _query_policy()
    assert policy.mode is QueryPolicyModeV1.INDEXED_ONLY


def test_indexed_only_rejects_connection_refs() -> None:
    with pytest.raises(ValidationError):
        _query_policy(allowed_connection_refs=("conn.a",))


def test_indexed_only_rejects_capability_ids() -> None:
    with pytest.raises(ValidationError):
        _query_policy(allowed_capability_ids=("cap.a",))


def test_indexed_only_rejects_live_calls() -> None:
    with pytest.raises(ValidationError):
        _query_policy(max_live_calls=1)


def test_indexed_only_rejects_receipt_only_retention() -> None:
    with pytest.raises(ValidationError):
        _query_policy(live_result_retention=LiveResultRetentionV1.RECEIPT_ONLY)


def test_valid_live_only_query_policy_accepted() -> None:
    policy = _query_policy(
        mode=QueryPolicyModeV1.LIVE_ONLY,
        allowed_connection_refs=("conn.a",),
        allowed_capability_ids=("cap.read",),
        max_live_calls=1,
    )
    assert policy.allowed_connection_refs == ("conn.a",)


def test_live_only_requires_connection_refs() -> None:
    with pytest.raises(ValidationError):
        _query_policy(
            mode=QueryPolicyModeV1.LIVE_ONLY,
            allowed_capability_ids=("cap.read",),
            max_live_calls=1,
        )


def test_live_only_requires_capability_ids() -> None:
    with pytest.raises(ValidationError):
        _query_policy(
            mode=QueryPolicyModeV1.LIVE_ONLY,
            allowed_connection_refs=("conn.a",),
            max_live_calls=1,
        )


def test_live_only_requires_max_live_calls() -> None:
    with pytest.raises(ValidationError):
        _query_policy(
            mode=QueryPolicyModeV1.LIVE_ONLY,
            allowed_connection_refs=("conn.a",),
            allowed_capability_ids=("cap.read",),
            max_live_calls=0,
        )


def test_query_policy_rejects_unsupported_modes() -> None:
    with pytest.raises(ValidationError):
        WorkspaceQueryPolicy(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            mode="hybrid",
            mutation_id=_MUTATION,
            effective_revision=1,
            updated_at=_NOW,
        )
    with pytest.raises(ValidationError):
        WorkspaceQueryPolicy(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            mode="automatic",
            mutation_id=_MUTATION,
            effective_revision=1,
            updated_at=_NOW,
        )


def test_query_policy_canonicalizes_tuple_values() -> None:
    policy = _query_policy(
        mode=QueryPolicyModeV1.LIVE_ONLY,
        allowed_connection_refs=(" conn.b ", "conn.a", "conn.a"),
        allowed_capability_ids=(" cap.z ", "cap.a", "cap.a"),
        max_live_calls=2,
    )
    assert policy.allowed_connection_refs == ("conn.a", "conn.b")
    assert policy.allowed_capability_ids == ("cap.a", "cap.z")


def test_query_policy_omitted_tuple_fields_default_to_empty() -> None:
    policy = _query_policy()
    assert policy.allowed_connection_refs == ()
    assert policy.allowed_capability_ids == ()


@pytest.mark.parametrize(
    "connection_refs,capability_ids",
    [
        ((), ()),
        ([], []),
    ],
)
def test_indexed_only_accepts_explicit_empty_tuple_fields(
    connection_refs: tuple[str, ...] | list[str],
    capability_ids: tuple[str, ...] | list[str],
) -> None:
    policy = _query_policy(
        allowed_connection_refs=connection_refs,
        allowed_capability_ids=capability_ids,
    )
    assert policy.allowed_connection_refs == ()
    assert policy.allowed_capability_ids == ()


def test_query_policy_rejects_explicit_none_connection_refs() -> None:
    with pytest.raises(ValidationError):
        _query_policy(allowed_connection_refs=None)


def test_query_policy_rejects_explicit_none_capability_ids() -> None:
    with pytest.raises(ValidationError):
        _query_policy(allowed_capability_ids=None)


def test_live_access_binding_rejects_explicit_none_capability_ids() -> None:
    with pytest.raises(ValidationError):
        _live_access_binding(allowed_capability_ids=None)


# --- Configuration head ---


def _configuration_head(**overrides: object) -> WorkspaceKnowledgeConfigurationHead:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceKnowledgeConfigurationHead(**payload)


def test_valid_idle_configuration_head_accepted() -> None:
    head = _configuration_head(committed_revision=3)
    assert head.pending_revision is None
    assert head.pending_mutation_id is None


def test_valid_pending_configuration_head_accepted() -> None:
    head = _configuration_head(
        committed_revision=3,
        pending_revision=4,
        pending_mutation_id="pending-mutation",
    )
    assert head.pending_revision == 4


def test_configuration_head_rejects_orphan_pending_revision() -> None:
    with pytest.raises(ValidationError):
        _configuration_head(committed_revision=1, pending_revision=2)


def test_configuration_head_rejects_orphan_pending_mutation_id() -> None:
    with pytest.raises(ValidationError):
        _configuration_head(committed_revision=1, pending_mutation_id="pending")


def test_configuration_head_rejects_wrong_pending_revision() -> None:
    with pytest.raises(ValidationError):
        _configuration_head(
            committed_revision=1,
            pending_revision=3,
            pending_mutation_id="pending",
        )


def test_configuration_head_rejects_blank_pending_mutation_id() -> None:
    with pytest.raises(ValidationError):
        _configuration_head(
            committed_revision=1,
            pending_revision=2,
            pending_mutation_id="   ",
        )


# --- Mutation record ---


def test_mutation_record_reserved_without_target_revision_accepted() -> None:
    record = _mutation_record()
    assert record.status is WorkspaceKnowledgeMutationStatusV1.RESERVED
    assert record.target_revision is None


def test_mutation_record_reserved_with_target_revision_accepted() -> None:
    record = _mutation_record(target_revision=1)
    assert record.status is WorkspaceKnowledgeMutationStatusV1.RESERVED
    assert record.target_revision == 1


def test_mutation_record_reserved_state_accepted() -> None:
    record = _mutation_record()
    assert record.status is WorkspaceKnowledgeMutationStatusV1.RESERVED


def test_mutation_record_prepared_state_accepted() -> None:
    record = _mutation_record(
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        target_revision=1,
    )
    assert record.target_revision == 1


def test_mutation_record_committed_applied_state_accepted() -> None:
    record = _mutation_record(
        status=WorkspaceKnowledgeMutationStatusV1.COMMITTED,
        target_revision=2,
        committed_revision=2,
        outcome=WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
        result_entity_type="connection_attachment",
        result_entity_id="att-1",
        committed_at=_NOW,
    )
    assert record.outcome is WorkspaceKnowledgeMutationOutcomeV1.APPLIED


def test_mutation_record_committed_existing_result_state_accepted() -> None:
    record = _mutation_record(
        status=WorkspaceKnowledgeMutationStatusV1.COMMITTED,
        committed_revision=2,
        outcome=WorkspaceKnowledgeMutationOutcomeV1.EXISTING_RESULT,
        result_entity_type="connection_attachment",
        result_entity_id="att-1",
        committed_at=_NOW,
    )
    assert record.target_revision is None


def test_mutation_record_aborted_before_target_assignment_accepted() -> None:
    record = _mutation_record(status=WorkspaceKnowledgeMutationStatusV1.ABORTED)
    assert record.target_revision is None


def test_mutation_record_aborted_after_target_assignment_accepted() -> None:
    record = _mutation_record(
        status=WorkspaceKnowledgeMutationStatusV1.ABORTED,
        target_revision=2,
    )
    assert record.target_revision == 2


def test_mutation_record_recovery_required_partial_state_accepted() -> None:
    record = _mutation_record(
        status=WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
        target_revision=2,
        error_code="writer_slot_stale",
    )
    assert record.error_code == "writer_slot_stale"


def test_mutation_record_reserved_with_outcome_rejected() -> None:
    with pytest.raises(ValidationError):
        _mutation_record(outcome=WorkspaceKnowledgeMutationOutcomeV1.APPLIED)


def test_mutation_record_prepared_without_target_revision_rejected() -> None:
    with pytest.raises(ValidationError):
        _mutation_record(status=WorkspaceKnowledgeMutationStatusV1.PREPARED)


def test_mutation_record_prepared_with_committed_revision_rejected() -> None:
    with pytest.raises(ValidationError):
        _mutation_record(
            status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
            target_revision=1,
            committed_revision=1,
        )


def test_mutation_record_committed_without_outcome_rejected() -> None:
    with pytest.raises(ValidationError):
        _mutation_record(
            status=WorkspaceKnowledgeMutationStatusV1.COMMITTED,
            target_revision=1,
            committed_revision=1,
            result_entity_type="x",
            result_entity_id="y",
            committed_at=_NOW,
        )


def test_mutation_record_applied_with_mismatched_revisions_rejected() -> None:
    with pytest.raises(ValidationError):
        _mutation_record(
            status=WorkspaceKnowledgeMutationStatusV1.COMMITTED,
            target_revision=2,
            committed_revision=1,
            outcome=WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
            result_entity_type="x",
            result_entity_id="y",
            committed_at=_NOW,
        )


def test_mutation_record_existing_result_with_target_revision_rejected() -> None:
    with pytest.raises(ValidationError):
        _mutation_record(
            status=WorkspaceKnowledgeMutationStatusV1.COMMITTED,
            target_revision=2,
            committed_revision=2,
            outcome=WorkspaceKnowledgeMutationOutcomeV1.EXISTING_RESULT,
            result_entity_type="x",
            result_entity_id="y",
            committed_at=_NOW,
        )


def test_mutation_record_committed_without_result_reference_rejected() -> None:
    with pytest.raises(ValidationError):
        _mutation_record(
            status=WorkspaceKnowledgeMutationStatusV1.COMMITTED,
            target_revision=1,
            committed_revision=1,
            outcome=WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
            committed_at=_NOW,
        )


def test_mutation_record_only_one_result_reference_field_rejected() -> None:
    with pytest.raises(ValidationError):
        _mutation_record(
            status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
            target_revision=1,
            result_entity_type="x",
        )


def test_mutation_record_committed_without_committed_at_rejected() -> None:
    with pytest.raises(ValidationError):
        _mutation_record(
            status=WorkspaceKnowledgeMutationStatusV1.COMMITTED,
            target_revision=1,
            committed_revision=1,
            outcome=WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
            result_entity_type="x",
            result_entity_id="y",
        )


def test_mutation_record_non_committed_with_committed_at_rejected() -> None:
    with pytest.raises(ValidationError):
        _mutation_record(
            status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
            target_revision=1,
            committed_at=_NOW,
        )


_CLAIM = "claim-1"


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({}, None),
        ({"stage_claim_id": _CLAIM, "target_revision": 1}, None),
        ({"stage_claim_id": ""}, "stage_claim"),
        ({"stage_claim_id": _CLAIM}, "stage_claim_requires_target_revision"),
        (
            {"stage_claim_id": _CLAIM, "target_revision": 1, "status": WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED},
            None,
        ),
    ],
)
def test_mutation_record_stage_claim_rules(overrides: dict[str, object], error_match: str | None) -> None:
    if error_match is None:
        assert _mutation_record(**overrides).stage_claim_id == overrides.get("stage_claim_id")
        return
    with pytest.raises(ValidationError, match=error_match):
        _mutation_record(**overrides)


@pytest.mark.parametrize(
    "status",
    [
        WorkspaceKnowledgeMutationStatusV1.PREPARED,
        WorkspaceKnowledgeMutationStatusV1.COMMITTED,
        WorkspaceKnowledgeMutationStatusV1.ABORTED,
    ],
)
def test_mutation_record_stage_claim_rejected_on_terminal_status(
    status: WorkspaceKnowledgeMutationStatusV1,
) -> None:
    overrides: dict[str, object] = {
        "stage_claim_id": _CLAIM,
        "target_revision": 1,
        "status": status,
    }
    if status is WorkspaceKnowledgeMutationStatusV1.PREPARED:
        overrides.update({"result_entity_type": "x", "result_entity_id": "y"})
    elif status is WorkspaceKnowledgeMutationStatusV1.COMMITTED:
        overrides.update(
            {
                "outcome": WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
                "committed_revision": 1,
                "result_entity_type": "x",
                "result_entity_id": "y",
                "committed_at": _NOW,
            }
        )
    with pytest.raises(ValidationError, match="stage_claim_invalid_for_status"):
        _mutation_record(**overrides)


# --- Configuration projection ---


def test_valid_empty_revision_zero_projection_accepted() -> None:
    config = WorkspaceKnowledgeConfigurationV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=0,
        updated_at=_NOW,
    )
    assert config.connection_attachments == ()


def test_configuration_projection_orders_children_deterministically() -> None:
    config = WorkspaceKnowledgeConfigurationV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=2,
        connection_attachments=(
            _connection_attachment(
                attachment_id="att-b",
                connection_ref="conn.z",
            ),
            _connection_attachment(
                attachment_id="att-a",
                connection_ref="conn.a",
            ),
            _connection_attachment(
                attachment_id="att-c",
                connection_ref="conn.a",
            ),
        ),
        indexed_sources=(
            _indexed_source_binding(indexed_source_binding_id="idx-b"),
            _indexed_source_binding(indexed_source_binding_id="idx-a"),
        ),
        live_access_bindings=(
            _live_access_binding(live_access_binding_id="live-b"),
            _live_access_binding(live_access_binding_id="live-a"),
        ),
        updated_at=_NOW,
    )
    assert [item.attachment_id for item in config.connection_attachments] == [
        "att-a",
        "att-c",
        "att-b",
    ]
    assert [item.indexed_source_binding_id for item in config.indexed_sources] == [
        "idx-a",
        "idx-b",
    ]
    assert [item.live_access_binding_id for item in config.live_access_bindings] == [
        "live-a",
        "live-b",
    ]


def test_configuration_projection_rejects_cross_tenant_child() -> None:
    with pytest.raises(ValidationError):
        WorkspaceKnowledgeConfigurationV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=1,
            connection_attachments=(
                _connection_attachment(tenant_id="tenant-other"),
            ),
            updated_at=_NOW,
        )


def test_configuration_projection_rejects_cross_workspace_child() -> None:
    with pytest.raises(ValidationError):
        WorkspaceKnowledgeConfigurationV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=1,
            indexed_sources=(
                _indexed_source_binding(workspace_id="workspace-other"),
            ),
            updated_at=_NOW,
        )


def test_configuration_projection_rejects_child_newer_than_revision() -> None:
    with pytest.raises(ValidationError):
        WorkspaceKnowledgeConfigurationV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=1,
            live_access_bindings=(
                _live_access_binding(effective_revision=2),
            ),
            updated_at=_NOW,
        )


def test_configuration_projection_rejects_revision_zero_with_child() -> None:
    with pytest.raises(ValidationError):
        WorkspaceKnowledgeConfigurationV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=0,
            connection_attachments=(_connection_attachment(),),
            updated_at=_NOW,
        )


# --- WorkspaceSource ---


@pytest.mark.parametrize(
    "source_type",
    [
        WorkspaceSourceType.LOCAL_FOLDER,
        WorkspaceSourceType.MANAGED_UPLOAD,
        WorkspaceSourceType.UPLOADED_FOLDER_SNAPSHOT,
        WorkspaceSourceType.WEB_RESOURCE,
    ],
)
def test_non_connected_source_types_remain_compatible(source_type: WorkspaceSourceType) -> None:
    kwargs: dict[str, object] = {
        "source_type": source_type,
    }
    if source_type is WorkspaceSourceType.LOCAL_FOLDER:
        kwargs["path"] = "/tmp/docs"
    source = _workspace_source(**kwargs)
    assert source.knowledge_configuration_creation_mutation_id is None
    assert source.knowledge_configuration_visibility_revision is None


def test_connected_source_with_ownership_fields_accepted() -> None:
    source = _workspace_source(
        source_type=WorkspaceSourceType.CONNECTED_SOURCE,
        knowledge_configuration_creation_mutation_id="mut-1",
        knowledge_configuration_visibility_revision=1,
    )
    assert source.path == ""
    assert source.recursive is False


def test_connected_source_without_ownership_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        _workspace_source(source_type=WorkspaceSourceType.CONNECTED_SOURCE)


def test_connected_source_with_only_mutation_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _workspace_source(
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            knowledge_configuration_creation_mutation_id="mut-1",
        )


def test_connected_source_with_only_visibility_revision_rejected() -> None:
    with pytest.raises(ValidationError):
        _workspace_source(
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            knowledge_configuration_visibility_revision=1,
        )


def test_non_connected_source_with_ownership_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        _workspace_source(
            source_type=WorkspaceSourceType.MANAGED_UPLOAD,
            knowledge_configuration_creation_mutation_id="mut-1",
            knowledge_configuration_visibility_revision=1,
        )


def test_connected_source_with_nonempty_path_rejected() -> None:
    with pytest.raises(ValidationError):
        _workspace_source(
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            path="/should-be-empty",
            knowledge_configuration_creation_mutation_id="mut-1",
            knowledge_configuration_visibility_revision=1,
        )


def test_connected_source_with_recursive_true_rejected() -> None:
    with pytest.raises(ValidationError):
        _workspace_source(
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            recursive=True,
            knowledge_configuration_creation_mutation_id="mut-1",
            knowledge_configuration_visibility_revision=1,
        )


def test_workspace_source_status_enum_unchanged() -> None:
    assert {item.value for item in WorkspaceSourceStatus} == {
        "registered",
        "syncing",
        "processing",
        "ready",
        "error",
    }


def test_knowledge_audience_eligibility_compatibility_alias() -> None:
    assert IndexedSourceAudienceEligibilityV1 is KnowledgeAudienceEligibilityV1


def test_indexed_binding_defaults_to_personal_only() -> None:
    binding = _indexed_source_binding()
    assert binding.audience_eligibility is KnowledgeAudienceEligibilityV1.PERSONAL_ONLY


def test_indexed_binding_accepts_explicit_shared_allowed() -> None:
    binding = _indexed_source_binding(
        audience_eligibility=KnowledgeAudienceEligibilityV1.SHARED_ALLOWED,
    )
    assert binding.audience_eligibility is KnowledgeAudienceEligibilityV1.SHARED_ALLOWED


def test_live_binding_defaults_to_personal_only() -> None:
    binding = _live_access_binding()
    assert binding.audience_eligibility is KnowledgeAudienceEligibilityV1.PERSONAL_ONLY


def test_live_binding_accepts_explicit_shared_allowed() -> None:
    binding = _live_access_binding(
        audience_eligibility=KnowledgeAudienceEligibilityV1.SHARED_ALLOWED,
    )
    assert binding.audience_eligibility is KnowledgeAudienceEligibilityV1.SHARED_ALLOWED


def test_audience_eligibility_serialized_values_unchanged() -> None:
    assert KnowledgeAudienceEligibilityV1.PERSONAL_ONLY.value == "personal_only"
    assert KnowledgeAudienceEligibilityV1.SHARED_ALLOWED.value == "shared_allowed"


def test_live_binding_missing_eligibility_field_defaults_to_personal_only() -> None:
    payload = {
        "live_access_binding_id": "live-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": "conn.live",
        "allowed_capability_ids": ("cap.read",),
        "derived_provider_id": "provider-1",
        "derived_integration_kind": IntegrationCategory.WIKI_KNOWLEDGE,
        "derived_safe_display_label": "Wiki",
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "semantic_identity_hash": _SHA256,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    binding = WorkspaceLiveAccessBinding.model_validate(payload)
    assert binding.audience_eligibility is KnowledgeAudienceEligibilityV1.PERSONAL_ONLY
