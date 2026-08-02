# © Artur Czarnecki. All rights reserved.

"""Opaque, tamper-evident Slack conversation candidate references."""

from __future__ import annotations

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDiscoveryError,
    RemoteResourceTypeV1,
    SlackConversationKindV1,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    RemoteResourceOpaqueRefCodec,
    SlackConversationCandidatePayload,
)


def encode_slack_conversation_candidate_ref(
    *,
    codec: RemoteResourceOpaqueRefCodec,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
    conversation_id: str,
    conversation_kind: SlackConversationKindV1,
    safe_display_label: str,
) -> str:
    return codec.encode_slack_conversation_candidate(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        connection_ref=connection_ref,
        conversation_id=conversation_id,
        conversation_kind=conversation_kind,
        safe_display_label=safe_display_label,
    )


def decode_slack_conversation_candidate_ref(
    codec: RemoteResourceOpaqueRefCodec,
    opaque_candidate_ref: str,
) -> SlackConversationCandidatePayload:
    return codec.decode_slack_conversation_candidate(opaque_candidate_ref)


def validate_candidate_scope(
    payload: SlackConversationCandidatePayload,
    *,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
) -> None:
    if payload.tenant_id != tenant_id:
        raise ConnectedSourceDiscoveryError("workspace_not_found")
    if payload.workspace_id != workspace_id:
        raise ConnectedSourceDiscoveryError("workspace_not_found")
    if payload.connection_ref != connection_ref:
        raise ConnectedSourceDiscoveryError("connection_not_attached")


def map_slack_conversation_kind(value: str) -> SlackConversationKindV1:
    try:
        return SlackConversationKindV1(value)
    except ValueError:
        raise ConnectedSourceDiscoveryError("candidate_inaccessible") from None


def resource_type_for_candidate(_: SlackConversationCandidatePayload) -> RemoteResourceTypeV1:
    return RemoteResourceTypeV1.SLACK_CONVERSATION
