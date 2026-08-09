# © Artur Czarnecki. All rights reserved.

"""HMAC-signed opaque references for connected-source discovery and pagination."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDiscoveryError,
    RemoteResourceTypeV1,
    SlackConversationKindV1,
)

_ENVELOPE_SCHEMA = "lkw.remote_resource_opaque_ref.v1"
_CANDIDATE_PAYLOAD_SCHEMA = "lkw.slack_conversation_candidate.v1"
_MSGRAPH_CANDIDATE_PAYLOAD_SCHEMA = "lkw.msgraph_teams_chat_candidate.v1"
_MSGRAPH_MAIL_CANDIDATE_PAYLOAD_SCHEMA = "lkw.msgraph_mail_folder_candidate.v1"
_MSGRAPH_TEAMS_CHANNEL_CANDIDATE_PAYLOAD_SCHEMA = (
    "lkw.msgraph_teams_channel_candidate.v1"
)
_MSGRAPH_CALENDAR_CANDIDATE_PAYLOAD_SCHEMA = "lkw.msgraph_calendar_candidate.v1"
_GOOGLE_WORKSPACE_CANDIDATE_PAYLOAD_SCHEMA = "lkw.google_workspace_candidate.v1"
_PAGINATION_PAYLOAD_SCHEMA = "lkw.remote_resource_pagination.v1"
_MAX_TOKEN_LEN = 1024
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class RemoteResourceOpaqueRefCodecError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class SlackConversationCandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=_CANDIDATE_PAYLOAD_SCHEMA)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    resource_type: RemoteResourceTypeV1
    conversation_id: str = Field(..., min_length=1, max_length=256)
    conversation_kind: SlackConversationKindV1
    safe_display_label: str = Field(..., min_length=1, max_length=256)
    snapshot_version: str = Field(..., min_length=64, max_length=64)

    @field_validator("snapshot_version")
    @classmethod
    def _validate_snapshot_version(cls, value: str) -> str:
        if _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("snapshot_version_invalid")
        return value


class MsGraphTeamsChatCandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=_MSGRAPH_CANDIDATE_PAYLOAD_SCHEMA)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    resource_type: RemoteResourceTypeV1
    mailbox_user_id: str = Field(..., min_length=1, max_length=256)
    chat_remote_id: str = Field(..., min_length=1, max_length=256)
    safe_display_label: str = Field(..., min_length=1, max_length=256)


class MsGraphMailFolderCandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=_MSGRAPH_MAIL_CANDIDATE_PAYLOAD_SCHEMA)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    resource_type: RemoteResourceTypeV1
    mailbox_user_id: str = Field(..., min_length=1, max_length=256)
    folder_id: str = Field(..., min_length=1, max_length=256)
    safe_display_label: str = Field(..., min_length=1, max_length=256)


class MsGraphTeamsChannelCandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=_MSGRAPH_TEAMS_CHANNEL_CANDIDATE_PAYLOAD_SCHEMA)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    resource_type: RemoteResourceTypeV1
    team_remote_id: str = Field(..., min_length=1, max_length=256)
    channel_remote_id: str = Field(..., min_length=1, max_length=256)
    safe_display_label: str = Field(..., min_length=1, max_length=256)


class MsGraphCalendarCandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=_MSGRAPH_CALENDAR_CANDIDATE_PAYLOAD_SCHEMA)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    resource_type: RemoteResourceTypeV1
    mailbox_user_id: str = Field(..., min_length=1, max_length=256)
    calendar_remote_id: str = Field(..., min_length=1, max_length=256)
    is_default_calendar: bool
    safe_display_label: str = Field(..., min_length=1, max_length=256)


class GoogleWorkspaceCandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=_GOOGLE_WORKSPACE_CANDIDATE_PAYLOAD_SCHEMA)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    resource_type: RemoteResourceTypeV1
    remote_resource_id: str = Field(..., min_length=1, max_length=256)
    safe_display_label: str = Field(..., min_length=1, max_length=256)


class RemoteResourcePaginationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=_PAGINATION_PAYLOAD_SCHEMA)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    resource_type: RemoteResourceTypeV1
    provider_cursor: str | None = Field(default=None, max_length=512)


def _canonical_json(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def snapshot_version(
    *,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
    conversation_id: str,
    conversation_kind: SlackConversationKindV1,
    safe_display_label: str,
) -> str:
    payload = _canonical_json(
        {
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "connection_ref": connection_ref.strip(),
            "conversation_id": conversation_id.strip(),
            "conversation_kind": conversation_kind.value,
            "safe_display_label": safe_display_label.strip(),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RemoteResourceOpaqueRefCodec:
    """Provider-neutral HMAC-SHA256 codec for bounded opaque application tokens."""

    def __init__(self, *, signing_key: bytes) -> None:
        if not signing_key:
            raise RemoteResourceOpaqueRefCodecError("opaque_ref_signing_key_missing")
        self._signing_key = signing_key

    @classmethod
    def from_signing_key_material(cls, material: str) -> RemoteResourceOpaqueRefCodec:
        cleaned = material.strip()
        if not cleaned:
            raise RemoteResourceOpaqueRefCodecError("opaque_ref_signing_key_missing")
        return cls(signing_key=cleaned.encode("utf-8"))

    def _sign(self, payload_bytes: bytes) -> str:
        return hmac.new(self._signing_key, payload_bytes, hashlib.sha256).hexdigest()

    def _encode_payload(self, payload: dict[str, object]) -> str:
        payload_bytes = _canonical_json(payload).encode("utf-8")
        signature = self._sign(payload_bytes)
        envelope = {
            "schema": _ENVELOPE_SCHEMA,
            "payload": payload,
            "signature": signature,
        }
        encoded = base64.urlsafe_b64encode(
            _canonical_json(envelope).encode("utf-8")
        ).decode("ascii")
        token = encoded.rstrip("=")
        if len(token) > _MAX_TOKEN_LEN:
            raise RemoteResourceOpaqueRefCodecError("opaque_ref_token_too_large")
        return token

    def _decode_payload(self, token: str) -> dict[str, object]:
        cleaned = token.strip()
        if not cleaned:
            raise RemoteResourceOpaqueRefCodecError("opaque_ref_invalid")
        padding = "=" * (-len(cleaned) % 4)
        try:
            raw = base64.urlsafe_b64decode(cleaned + padding)
            envelope = json.loads(raw.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            raise RemoteResourceOpaqueRefCodecError("opaque_ref_invalid") from None
        if not isinstance(envelope, dict):
            raise RemoteResourceOpaqueRefCodecError("opaque_ref_invalid")
        if envelope.get("schema") != _ENVELOPE_SCHEMA:
            raise RemoteResourceOpaqueRefCodecError("opaque_ref_invalid")
        payload = envelope.get("payload")
        signature = envelope.get("signature")
        if not isinstance(payload, dict) or not isinstance(signature, str):
            raise RemoteResourceOpaqueRefCodecError("opaque_ref_invalid")
        payload_bytes = _canonical_json(payload).encode("utf-8")
        expected = self._sign(payload_bytes)
        if not hmac.compare_digest(expected, signature):
            raise RemoteResourceOpaqueRefCodecError("opaque_ref_signature_invalid")
        return payload

    def encode_slack_conversation_candidate(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        conversation_id: str,
        conversation_kind: SlackConversationKindV1,
        safe_display_label: str,
    ) -> str:
        snapshot = snapshot_version(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            conversation_id=conversation_id,
            conversation_kind=conversation_kind,
            safe_display_label=safe_display_label,
        )
        payload = SlackConversationCandidatePayload(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            resource_type=RemoteResourceTypeV1.SLACK_CONVERSATION,
            conversation_id=conversation_id,
            conversation_kind=conversation_kind,
            safe_display_label=safe_display_label,
            snapshot_version=snapshot,
        )
        return self._encode_payload(payload.model_dump(mode="json"))

    def decode_slack_conversation_candidate(self, opaque_candidate_ref: str) -> SlackConversationCandidatePayload:
        try:
            data = self._decode_payload(opaque_candidate_ref)
            payload = SlackConversationCandidatePayload.model_validate(data)
        except (ValueError, RemoteResourceOpaqueRefCodecError):
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid") from None
        if payload.schema_version != _CANDIDATE_PAYLOAD_SCHEMA:
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid")
        expected_snapshot = snapshot_version(
            tenant_id=payload.tenant_id,
            workspace_id=payload.workspace_id,
            connection_ref=payload.connection_ref,
            conversation_id=payload.conversation_id,
            conversation_kind=payload.conversation_kind,
            safe_display_label=payload.safe_display_label,
        )
        if payload.snapshot_version != expected_snapshot:
            raise ConnectedSourceDiscoveryError("candidate_ref_tampered")
        return payload

    def encode_msgraph_teams_chat_candidate(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        mailbox_user_id: str,
        chat_remote_id: str,
        safe_display_label: str,
    ) -> str:
        payload = MsGraphTeamsChatCandidatePayload(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            resource_type=RemoteResourceTypeV1.MSGRAPH_TEAMS_CHAT,
            mailbox_user_id=mailbox_user_id,
            chat_remote_id=chat_remote_id,
            safe_display_label=safe_display_label,
        )
        return self._encode_payload(payload.model_dump(mode="json"))

    def decode_msgraph_teams_chat_candidate(
        self,
        opaque_candidate_ref: str,
    ) -> MsGraphTeamsChatCandidatePayload:
        try:
            data = self._decode_payload(opaque_candidate_ref)
            payload = MsGraphTeamsChatCandidatePayload.model_validate(data)
        except (ValueError, RemoteResourceOpaqueRefCodecError):
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid") from None
        if payload.schema_version != _MSGRAPH_CANDIDATE_PAYLOAD_SCHEMA:
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid")
        return payload

    def encode_msgraph_mail_folder_candidate(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        mailbox_user_id: str,
        folder_id: str,
        safe_display_label: str,
    ) -> str:
        payload = MsGraphMailFolderCandidatePayload(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            resource_type=RemoteResourceTypeV1.MSGRAPH_MAIL_FOLDER,
            mailbox_user_id=mailbox_user_id,
            folder_id=folder_id,
            safe_display_label=safe_display_label,
        )
        return self._encode_payload(payload.model_dump(mode="json"))

    def decode_msgraph_mail_folder_candidate(
        self,
        opaque_candidate_ref: str,
    ) -> MsGraphMailFolderCandidatePayload:
        try:
            data = self._decode_payload(opaque_candidate_ref)
            payload = MsGraphMailFolderCandidatePayload.model_validate(data)
        except (ValueError, RemoteResourceOpaqueRefCodecError):
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid") from None
        if payload.schema_version != _MSGRAPH_MAIL_CANDIDATE_PAYLOAD_SCHEMA:
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid")
        return payload

    def encode_msgraph_teams_channel_candidate(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        team_remote_id: str,
        channel_remote_id: str,
        safe_display_label: str,
    ) -> str:
        payload = MsGraphTeamsChannelCandidatePayload(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            resource_type=RemoteResourceTypeV1.MSGRAPH_TEAMS_CHANNEL,
            team_remote_id=team_remote_id,
            channel_remote_id=channel_remote_id,
            safe_display_label=safe_display_label,
        )
        return self._encode_payload(payload.model_dump(mode="json"))

    def decode_msgraph_teams_channel_candidate(
        self,
        opaque_candidate_ref: str,
    ) -> MsGraphTeamsChannelCandidatePayload:
        try:
            data = self._decode_payload(opaque_candidate_ref)
            payload = MsGraphTeamsChannelCandidatePayload.model_validate(data)
        except (ValueError, RemoteResourceOpaqueRefCodecError):
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid") from None
        if payload.schema_version != _MSGRAPH_TEAMS_CHANNEL_CANDIDATE_PAYLOAD_SCHEMA:
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid")
        return payload

    def encode_msgraph_calendar_candidate(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        mailbox_user_id: str,
        calendar_remote_id: str,
        is_default_calendar: bool,
        safe_display_label: str,
    ) -> str:
        payload = MsGraphCalendarCandidatePayload(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            resource_type=RemoteResourceTypeV1.MSGRAPH_CALENDAR,
            mailbox_user_id=mailbox_user_id,
            calendar_remote_id=calendar_remote_id,
            is_default_calendar=is_default_calendar,
            safe_display_label=safe_display_label,
        )
        return self._encode_payload(payload.model_dump(mode="json"))

    def decode_msgraph_calendar_candidate(
        self,
        opaque_candidate_ref: str,
    ) -> MsGraphCalendarCandidatePayload:
        try:
            data = self._decode_payload(opaque_candidate_ref)
            payload = MsGraphCalendarCandidatePayload.model_validate(data)
        except (ValueError, RemoteResourceOpaqueRefCodecError):
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid") from None
        if payload.schema_version != _MSGRAPH_CALENDAR_CANDIDATE_PAYLOAD_SCHEMA:
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid")
        return payload

    def encode_google_workspace_candidate(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        resource_type: RemoteResourceTypeV1,
        remote_resource_id: str,
        safe_display_label: str,
    ) -> str:
        payload = GoogleWorkspaceCandidatePayload(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            resource_type=resource_type,
            remote_resource_id=remote_resource_id,
            safe_display_label=safe_display_label,
        )
        return self._encode_payload(payload.model_dump(mode="json"))

    def decode_google_workspace_candidate(
        self,
        opaque_candidate_ref: str,
    ) -> GoogleWorkspaceCandidatePayload:
        try:
            data = self._decode_payload(opaque_candidate_ref)
            payload = GoogleWorkspaceCandidatePayload.model_validate(data)
        except (ValueError, RemoteResourceOpaqueRefCodecError):
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid") from None
        if payload.schema_version != _GOOGLE_WORKSPACE_CANDIDATE_PAYLOAD_SCHEMA:
            raise ConnectedSourceDiscoveryError("candidate_ref_invalid")
        return payload

    def encode_pagination_cursor(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        resource_type: RemoteResourceTypeV1,
        provider_cursor: str | None,
    ) -> str | None:
        if provider_cursor is None:
            return None
        payload = RemoteResourcePaginationPayload(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            resource_type=resource_type,
            provider_cursor=provider_cursor,
        )
        return self._encode_payload(payload.model_dump(mode="json"))

    def decode_pagination_cursor(
        self,
        *,
        opaque_cursor: str,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        resource_type: RemoteResourceTypeV1,
    ) -> str | None:
        try:
            data = self._decode_payload(opaque_cursor)
            payload = RemoteResourcePaginationPayload.model_validate(data)
        except (ValueError, RemoteResourceOpaqueRefCodecError):
            raise ConnectedSourceDiscoveryError("discovery_cursor_invalid") from None
        if payload.schema_version != _PAGINATION_PAYLOAD_SCHEMA:
            raise ConnectedSourceDiscoveryError("discovery_cursor_invalid")
        if payload.tenant_id != tenant_id:
            raise ConnectedSourceDiscoveryError("workspace_not_found")
        if payload.workspace_id != workspace_id:
            raise ConnectedSourceDiscoveryError("workspace_not_found")
        if payload.connection_ref != connection_ref:
            raise ConnectedSourceDiscoveryError("connection_not_attached")
        if payload.resource_type is not resource_type:
            raise ConnectedSourceDiscoveryError("resource_type_unsupported")
        return payload.provider_cursor
