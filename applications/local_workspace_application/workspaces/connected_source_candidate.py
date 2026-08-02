# © Artur Czarnecki. All rights reserved.

"""Opaque, tamper-evident Slack conversation candidate references."""

from __future__ import annotations

import base64
import hashlib
import json
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDiscoveryError,
    RemoteResourceTypeV1,
    SlackConversationKindV1,
)

_CANDIDATE_SCHEMA = "lkw.slack_conversation_candidate.v1"
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class _SlackConversationCandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=_CANDIDATE_SCHEMA)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
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


def _canonical_json(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _snapshot_version(
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


def encode_slack_conversation_candidate_ref(
    *,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
    conversation_id: str,
    conversation_kind: SlackConversationKindV1,
    safe_display_label: str,
) -> str:
    snapshot = _snapshot_version(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        connection_ref=connection_ref,
        conversation_id=conversation_id,
        conversation_kind=conversation_kind,
        safe_display_label=safe_display_label,
    )
    payload = _SlackConversationCandidatePayload(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        connection_ref=connection_ref,
        conversation_id=conversation_id,
        conversation_kind=conversation_kind,
        safe_display_label=safe_display_label,
        snapshot_version=snapshot,
    )
    encoded = base64.urlsafe_b64encode(
        _canonical_json(payload.model_dump(mode="json")).encode("utf-8")
    ).decode("ascii")
    return encoded.rstrip("=")


def decode_slack_conversation_candidate_ref(
    opaque_candidate_ref: str,
) -> _SlackConversationCandidatePayload:
    cleaned = opaque_candidate_ref.strip()
    if not cleaned:
        raise ConnectedSourceDiscoveryError("candidate_ref_invalid")
    padding = "=" * (-len(cleaned) % 4)
    try:
        raw = base64.urlsafe_b64decode(cleaned + padding)
        data = json.loads(raw.decode("utf-8"))
        payload = _SlackConversationCandidatePayload.model_validate(data)
    except (ValueError, json.JSONDecodeError):
        raise ConnectedSourceDiscoveryError("candidate_ref_invalid") from None
    if payload.schema_version != _CANDIDATE_SCHEMA:
        raise ConnectedSourceDiscoveryError("candidate_ref_invalid")
    expected_snapshot = _snapshot_version(
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


def validate_candidate_scope(
    payload: _SlackConversationCandidatePayload,
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


def resource_type_for_candidate(_: _SlackConversationCandidatePayload) -> RemoteResourceTypeV1:
    return RemoteResourceTypeV1.SLACK_CONVERSATION
