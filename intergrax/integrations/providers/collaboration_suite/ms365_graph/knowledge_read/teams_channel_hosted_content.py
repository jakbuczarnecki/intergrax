# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Teams Channel knowledge-read: hosted content inventory and bounded bytes."""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote, unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    parse_msgraph_collection_page,
    validate_msgraph_continuation_url,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    validate_msgraph_teams_channel_hosted_content_id,
    validate_msgraph_teams_channel_id,
    validate_msgraph_teams_channel_message_id,
    validate_msgraph_teams_team_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
    MsGraphTeamsChannelMessage,
    MsGraphTeamsChannelMessageChanged,
    MsGraphTeamsChannelMessageKind,
    MsGraphTeamsChannelMessageState,
    read_and_validate_current_teams_channel_message_observation,
    validate_msgraph_teams_channel_message,
)

DEFAULT_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES = 10 * 1024 * 1024
ABSOLUTE_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES = 25 * 1024 * 1024

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_HOSTED_CONTENT_RESPONSE = (
    "unexpected Microsoft Graph Teams hosted content response"
)
_INVALID_HOSTED_CONTENT_REQUEST = "invalid Microsoft Graph Teams hosted content request"
_INVALID_HOSTED_CONTENT_CONTINUATION = (
    "invalid Microsoft Graph Teams hosted content continuation"
)
_INVALID_HOSTED_CONTENT_RESPONSE = "Microsoft Graph Teams hosted content response is invalid"
_CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTENT_TYPE_LEN = 1024
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PREFER_UNKNOWN_ENUM = {"Prefer": "include-unknown-enum-members"}


class MsGraphTeamsChannelHostedContentTooLarge(IntegrationConfigurationError):
    def __init__(self) -> None:
        super().__init__(
            "Microsoft Graph Teams hosted content exceeds the configured limit"
        )


class MsGraphTeamsChannelHostedContent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    team_remote_id: str
    channel_remote_id: str
    message_remote_id: str
    thread_root_remote_id: str
    message_kind: MsGraphTeamsChannelMessageKind
    message_revision: str = Field(repr=False)

    remote_id: str = Field(repr=False)

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("channel_remote_id", mode="before")
    @classmethod
    def _validate_channel_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_id(value)

    @field_validator("message_remote_id", mode="before")
    @classmethod
    def _validate_message_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_message_id(value)

    @field_validator("thread_root_remote_id", mode="before")
    @classmethod
    def _validate_thread_root_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_message_id(value)

    @field_validator("message_kind", mode="before")
    @classmethod
    def _validate_message_kind(cls, value: object) -> MsGraphTeamsChannelMessageKind:
        if not isinstance(value, MsGraphTeamsChannelMessageKind):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return value

    @field_validator("message_revision", mode="before")
    @classmethod
    def _validate_message_revision(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        trimmed = value.strip()
        if not trimmed or _ASCII_CONTROL.search(trimmed):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        if len(trimmed) > 4096:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return trimmed

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_hosted_content_id(value)

    @model_validator(mode="after")
    def _validate_kind_shape(self) -> MsGraphTeamsChannelHostedContent:
        if self.message_kind is MsGraphTeamsChannelMessageKind.ROOT:
            if self.thread_root_remote_id != self.message_remote_id:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        elif self.message_kind is MsGraphTeamsChannelMessageKind.REPLY:
            if self.thread_root_remote_id == self.message_remote_id:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return self


class MsGraphTeamsChannelHostedContentPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    team_remote_id: str
    channel_remote_id: str
    message_remote_id: str
    thread_root_remote_id: str
    message_kind: MsGraphTeamsChannelMessageKind
    message_revision: str = Field(repr=False)

    items: tuple[MsGraphTeamsChannelHostedContent, ...]

    continuation: MsGraphKnowledgeContinuation | None = Field(default=None, repr=False)

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("channel_remote_id", mode="before")
    @classmethod
    def _validate_channel_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_id(value)

    @field_validator("message_remote_id", mode="before")
    @classmethod
    def _validate_message_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_message_id(value)

    @field_validator("thread_root_remote_id", mode="before")
    @classmethod
    def _validate_thread_root_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_message_id(value)

    @field_validator("message_kind", mode="before")
    @classmethod
    def _validate_message_kind(cls, value: object) -> MsGraphTeamsChannelMessageKind:
        if not isinstance(value, MsGraphTeamsChannelMessageKind):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return value

    @field_validator("message_revision", mode="before")
    @classmethod
    def _validate_message_revision(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        trimmed = value.strip()
        if not trimmed or _ASCII_CONTROL.search(trimmed):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return trimmed

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(
        cls, value: object
    ) -> tuple[MsGraphTeamsChannelHostedContent, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphTeamsChannelHostedContent):
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphTeamsChannelHostedContentPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        for item in self.items:
            if item.team_remote_id != self.team_remote_id:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
            if item.channel_remote_id != self.channel_remote_id:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
            if item.message_remote_id != self.message_remote_id:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
            if item.thread_root_remote_id != self.thread_root_remote_id:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
            if item.message_kind != self.message_kind:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
            if item.message_revision != self.message_revision:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return self


class MsGraphTeamsChannelHostedContentBytes(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    team_remote_id: str
    channel_remote_id: str
    message_remote_id: str
    thread_root_remote_id: str
    message_kind: MsGraphTeamsChannelMessageKind
    message_revision: str = Field(repr=False)

    hosted_content_remote_id: str = Field(repr=False)

    content_type: str | None = None

    data: bytes = Field(repr=False)
    size_bytes: int
    content_hash: str

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("channel_remote_id", mode="before")
    @classmethod
    def _validate_channel_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_id(value)

    @field_validator("message_remote_id", mode="before")
    @classmethod
    def _validate_message_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_message_id(value)

    @field_validator("thread_root_remote_id", mode="before")
    @classmethod
    def _validate_thread_root_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_message_id(value)

    @field_validator("message_kind", mode="before")
    @classmethod
    def _validate_message_kind(cls, value: object) -> MsGraphTeamsChannelMessageKind:
        if not isinstance(value, MsGraphTeamsChannelMessageKind):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return value

    @field_validator("hosted_content_remote_id", mode="before")
    @classmethod
    def _validate_hosted_content_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_hosted_content_id(value)

    @field_validator("content_type", mode="before")
    @classmethod
    def _validate_content_type(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        trimmed = value.strip()
        if not trimmed:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        if _ASCII_CONTROL.search(trimmed):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        if len(trimmed) > _MAX_CONTENT_TYPE_LEN:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return trimmed

    @field_validator("data", mode="before")
    @classmethod
    def _validate_data(cls, value: object) -> bytes:
        if type(value) is not bytes:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return value

    @field_validator("content_hash", mode="before")
    @classmethod
    def _validate_content_hash(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        if not _CONTENT_HASH_PATTERN.fullmatch(value):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_bytes_shape(self) -> MsGraphTeamsChannelHostedContentBytes:
        if self.size_bytes != len(self.data):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        expected_hash = hashlib.sha256(self.data).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return self


def _safe_construct_hosted_content(**kwargs: object) -> MsGraphTeamsChannelHostedContent:
    try:
        return MsGraphTeamsChannelHostedContent(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None


def _safe_construct_hosted_content_page(**kwargs: object) -> MsGraphTeamsChannelHostedContentPage:
    try:
        return MsGraphTeamsChannelHostedContentPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None


def _safe_construct_hosted_content_bytes(**kwargs: object) -> MsGraphTeamsChannelHostedContentBytes:
    try:
        return MsGraphTeamsChannelHostedContentBytes(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None


def _require_active_message(message: object) -> MsGraphTeamsChannelMessage:
    validated = validate_msgraph_teams_channel_message(message)
    if validated.state is not MsGraphTeamsChannelMessageState.ACTIVE:
        raise MsGraphTeamsChannelMessageChanged() from None
    return validated


def _message_context_kwargs(message: MsGraphTeamsChannelMessage) -> dict[str, object]:
    return {
        "team_remote_id": message.team_remote_id,
        "channel_remote_id": message.channel_remote_id,
        "message_remote_id": message.remote_id,
        "thread_root_remote_id": message.thread_root_remote_id,
        "message_kind": message.message_kind,
        "message_revision": message.revision,
    }


def parse_msgraph_teams_channel_hosted_content(
    payload: object,
    *,
    message: MsGraphTeamsChannelMessage,
) -> MsGraphTeamsChannelHostedContent:
    validated_message = _require_active_message(message)
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if "id" not in payload:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if "contentBytes" in payload and payload.get("contentBytes") is not None:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if "contentType" in payload and payload.get("contentType") is not None:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    try:
        remote_id = validate_msgraph_teams_channel_hosted_content_id(payload.get("id"))
    except ValueError:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    return _safe_construct_hosted_content(
        **_message_context_kwargs(validated_message),
        remote_id=remote_id,
    )


def validate_msgraph_teams_channel_hosted_content(
    value: object,
) -> MsGraphTeamsChannelHostedContent:
    if not isinstance(value, MsGraphTeamsChannelHostedContent):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    try:
        return MsGraphTeamsChannelHostedContent.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None


def validate_msgraph_teams_channel_hosted_content_page(
    value: object,
    *,
    message: MsGraphTeamsChannelMessage,
    graph_base_url: str,
) -> MsGraphTeamsChannelHostedContentPage:
    if not isinstance(value, MsGraphTeamsChannelHostedContentPage):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    validated_message = _require_active_message(message)

    try:
        raw_team = value.team_remote_id
        raw_channel = value.channel_remote_id
        raw_message_id = value.message_remote_id
        raw_thread_root = value.thread_root_remote_id
        raw_kind = value.message_kind
        raw_revision = value.message_revision
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    if raw_team != validated_message.team_remote_id:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if raw_channel != validated_message.channel_remote_id:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if raw_message_id != validated_message.remote_id:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if raw_thread_root != validated_message.thread_root_remote_id:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if raw_kind != validated_message.message_kind:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if raw_revision != validated_message.revision:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    validated_items: list[MsGraphTeamsChannelHostedContent] = []
    for item in raw_items:
        if not isinstance(item, MsGraphTeamsChannelHostedContent):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
        validated_item = validate_msgraph_teams_channel_hosted_content(item)
        if (
            validated_item.team_remote_id != validated_message.team_remote_id
            or validated_item.channel_remote_id != validated_message.channel_remote_id
            or validated_item.message_remote_id != validated_message.remote_id
            or validated_item.thread_root_remote_id != validated_message.thread_root_remote_id
            or validated_item.message_kind != validated_message.message_kind
            or validated_item.message_revision != validated_message.revision
        ):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
        validated_items.append(validated_item)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    validated_continuation: MsGraphKnowledgeContinuation | None = None
    if raw_continuation is not None:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
        try:
            validated_continuation = validate_msgraph_teams_channel_hosted_contents_continuation(
                raw_continuation,
                team_id=validated_message.team_remote_id,
                channel_id=validated_message.channel_remote_id,
                thread_root_id=validated_message.thread_root_remote_id,
                message_id=validated_message.remote_id,
                message_kind=validated_message.message_kind,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    try:
        return MsGraphTeamsChannelHostedContentPage(
            team_remote_id=validated_message.team_remote_id,
            channel_remote_id=validated_message.channel_remote_id,
            message_remote_id=validated_message.remote_id,
            thread_root_remote_id=validated_message.thread_root_remote_id,
            message_kind=validated_message.message_kind,
            message_revision=validated_message.revision,
            items=tuple(validated_items),
            continuation=validated_continuation,
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None


def validate_msgraph_teams_channel_hosted_content_bytes(
    value: object,
    *,
    message: MsGraphTeamsChannelMessage,
    hosted_content: MsGraphTeamsChannelHostedContent,
    max_bytes: int,
) -> MsGraphTeamsChannelHostedContentBytes:
    if (
        type(max_bytes) is not int
        or max_bytes < 1
        or max_bytes > ABSOLUTE_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES
    ):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    validated_message = _require_active_message(message)
    validated_hosted = validate_msgraph_teams_channel_hosted_content(hosted_content)

    if not isinstance(value, MsGraphTeamsChannelHostedContentBytes):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    if (
        validated_hosted.team_remote_id != validated_message.team_remote_id
        or validated_hosted.channel_remote_id != validated_message.channel_remote_id
        or validated_hosted.message_remote_id != validated_message.remote_id
        or validated_hosted.thread_root_remote_id != validated_message.thread_root_remote_id
        or validated_hosted.message_kind != validated_message.message_kind
        or validated_hosted.message_revision != validated_message.revision
    ):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    try:
        reconstructed = MsGraphTeamsChannelHostedContentBytes.model_validate(
            value.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    if (
        reconstructed.team_remote_id != validated_message.team_remote_id
        or reconstructed.channel_remote_id != validated_message.channel_remote_id
        or reconstructed.message_remote_id != validated_message.remote_id
        or reconstructed.thread_root_remote_id != validated_message.thread_root_remote_id
        or reconstructed.message_kind != validated_message.message_kind
        or reconstructed.message_revision != validated_message.revision
        or reconstructed.hosted_content_remote_id != validated_hosted.remote_id
    ):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    if reconstructed.size_bytes > max_bytes:
        raise MsGraphTeamsChannelHostedContentTooLarge() from None

    return reconstructed


def _graph_base_path(graph_base_url: str) -> str:
    parsed_base = urlparse(graph_base_url)
    return parsed_base.path.rstrip("/") or "/"


def _decode_odata_literal(literal: str) -> str:
    return literal.replace("''", "'")


def _decode_path_segment(segment: str, *, odata_literal: bool) -> str:
    decoded = unquote(segment)
    if odata_literal:
        return _decode_odata_literal(decoded)
    return decoded


def _identity_in_path(resource: str, *, odata_literal: bool) -> str:
    if odata_literal:
        return rf"{resource}\('((?:[^']|'')*)'\)"
    return rf"{resource}/([^/]+)"


def _extract_hosted_contents_path(
    path: str,
    *,
    graph_base_path: str,
) -> tuple[str, str, str, str | None] | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    root_patterns: list[tuple[str, bool, bool, bool]] = []
    for team_odata, channel_odata, root_odata in itertools.product(
        (False, True), repeat=3
    ):
        pattern = (
            rf"^{re.escape(base)}/"
            f"{_identity_in_path('teams', odata_literal=team_odata)}/"
            f"{_identity_in_path('channels', odata_literal=channel_odata)}/"
            f"{_identity_in_path('messages', odata_literal=root_odata)}/hostedContents$"
        )
        root_patterns.append((pattern, team_odata, channel_odata, root_odata))

    for pattern, team_odata, channel_odata, root_odata in root_patterns:
        match = re.fullmatch(pattern, normalized, re.IGNORECASE)
        if match is None:
            continue
        team_segment = match.group(1)
        channel_segment = match.group(2)
        root_segment = match.group(3)
        if not team_segment or not channel_segment or not root_segment:
            return None
        return (
            _decode_path_segment(team_segment, odata_literal=team_odata),
            _decode_path_segment(channel_segment, odata_literal=channel_odata),
            _decode_path_segment(root_segment, odata_literal=root_odata),
            None,
        )

    reply_patterns: list[tuple[str, bool, bool, bool, bool]] = []
    for team_odata, channel_odata, root_odata, reply_odata in itertools.product(
        (False, True), repeat=4
    ):
        pattern = (
            rf"^{re.escape(base)}/"
            f"{_identity_in_path('teams', odata_literal=team_odata)}/"
            f"{_identity_in_path('channels', odata_literal=channel_odata)}/"
            f"{_identity_in_path('messages', odata_literal=root_odata)}/"
            f"{_identity_in_path('replies', odata_literal=reply_odata)}/hostedContents$"
        )
        reply_patterns.append(
            (pattern, team_odata, channel_odata, root_odata, reply_odata)
        )

    for pattern, team_odata, channel_odata, root_odata, reply_odata in reply_patterns:
        match = re.fullmatch(pattern, normalized, re.IGNORECASE)
        if match is None:
            continue
        team_segment = match.group(1)
        channel_segment = match.group(2)
        root_segment = match.group(3)
        reply_segment = match.group(4)
        if not team_segment or not channel_segment or not root_segment or not reply_segment:
            return None
        return (
            _decode_path_segment(team_segment, odata_literal=team_odata),
            _decode_path_segment(channel_segment, odata_literal=channel_odata),
            _decode_path_segment(root_segment, odata_literal=root_odata),
            _decode_path_segment(reply_segment, odata_literal=reply_odata),
        )

    return None


def validate_msgraph_teams_channel_hosted_contents_continuation(
    continuation: object,
    *,
    team_id: str,
    channel_id: str,
    thread_root_id: str,
    message_id: str,
    message_kind: MsGraphTeamsChannelMessageKind,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    if revalidated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted = _extract_hosted_contents_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    extracted_team, extracted_channel, extracted_root, extracted_reply = extracted
    try:
        validated_team = validate_msgraph_teams_team_id(team_id)
        validated_channel = validate_msgraph_teams_channel_id(channel_id)
        validated_root = validate_msgraph_teams_channel_message_id(thread_root_id)
        validated_message = validate_msgraph_teams_channel_message_id(message_id)
        validated_extracted_team = validate_msgraph_teams_team_id(extracted_team)
        validated_extracted_channel = validate_msgraph_teams_channel_id(extracted_channel)
        validated_extracted_root = validate_msgraph_teams_channel_message_id(extracted_root)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    if (
        validated_extracted_team != validated_team
        or validated_extracted_channel != validated_channel
        or validated_extracted_root != validated_root
    ):
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    if message_kind is MsGraphTeamsChannelMessageKind.ROOT:
        if extracted_reply is not None:
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None
        if validated_extracted_root != validated_message:
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None
    elif message_kind is MsGraphTeamsChannelMessageKind.REPLY:
        if extracted_reply is None:
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None
        try:
            validated_extracted_reply = validate_msgraph_teams_channel_message_id(
                extracted_reply
            )
        except ValueError:
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None
        if validated_extracted_reply != validated_message:
            raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None
    else:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    return revalidated


def _hosted_contents_inventory_path(message: MsGraphTeamsChannelMessage) -> str:
    quoted_team = quote(message.team_remote_id, safe="")
    quoted_channel = quote(message.channel_remote_id, safe="")
    quoted_root = quote(message.thread_root_remote_id, safe="")
    if message.message_kind is MsGraphTeamsChannelMessageKind.ROOT:
        return (
            f"/teams/{quoted_team}/channels/{quoted_channel}/messages/"
            f"{quoted_root}/hostedContents"
        )
    quoted_reply = quote(message.remote_id, safe="")
    return (
        f"/teams/{quoted_team}/channels/{quoted_channel}/messages/{quoted_root}"
        f"/replies/{quoted_reply}/hostedContents"
    )


def _hosted_content_value_path(
    message: MsGraphTeamsChannelMessage,
    hosted_content_remote_id: str,
) -> str:
    quoted_team = quote(message.team_remote_id, safe="")
    quoted_channel = quote(message.channel_remote_id, safe="")
    quoted_root = quote(message.thread_root_remote_id, safe="")
    quoted_hosted = quote(hosted_content_remote_id, safe="")
    if message.message_kind is MsGraphTeamsChannelMessageKind.ROOT:
        return (
            f"/teams/{quoted_team}/channels/{quoted_channel}/messages/{quoted_root}"
            f"/hostedContents/{quoted_hosted}/$value"
        )
    quoted_reply = quote(message.remote_id, safe="")
    return (
        f"/teams/{quoted_team}/channels/{quoted_channel}/messages/{quoted_root}"
        f"/replies/{quoted_reply}/hostedContents/{quoted_hosted}/$value"
    )


def _response_status_code(response: object) -> int:
    try:
        status_code = response.status_code
    except AttributeError:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if type(status_code) is not int:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if status_code < 100 or status_code > 599:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    return status_code


def _response_headers(response: object) -> Mapping[str, str]:
    try:
        headers = response.headers
    except AttributeError:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if not isinstance(headers, Mapping):
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    return headers


def _parse_content_length(headers: Mapping[str, str]) -> int | None:
    raw_value: str | None = None
    try:
        header_items = headers.items()
    except Exception:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    try:
        for key, value in header_items:
            if not isinstance(key, str):
                raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
            if key.lower() == "content-length":
                if raw_value is not None:
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                if not isinstance(value, str):
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                raw_value = value
    except IntegrationDependencyError:
        raise
    except Exception:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if raw_value is None:
        return None
    trimmed = raw_value.strip()
    if not trimmed or not trimmed.isdigit():
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    parsed = int(trimmed)
    if parsed < 0:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    return parsed


def _parse_content_type_header(headers: Mapping[str, str]) -> str | None:
    raw_value: str | None = None
    try:
        header_items = headers.items()
    except Exception:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    try:
        for key, value in header_items:
            if not isinstance(key, str):
                raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
            if key.lower() == "content-type":
                if raw_value is not None:
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                if not isinstance(value, str):
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                raw_value = value
    except IntegrationDependencyError:
        raise
    except Exception:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if raw_value is None:
        return None
    trimmed = raw_value.strip()
    if not trimmed:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if _ASCII_CONTROL.search(trimmed):
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if len(trimmed) > _MAX_CONTENT_TYPE_LEN:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    return trimmed


def _raise_for_hosted_content_download_response(response: object) -> None:
    status_code = _response_status_code(response)
    if status_code == 200:
        return
    if status_code == 206:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if 300 <= status_code <= 399:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if status_code in {400, 401, 403}:
        raise IntegrationConfigurationError("Microsoft Graph knowledge configuration failure") from None
    if status_code == 404:
        raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure") from None
    if status_code in {408, 410, 429} or status_code >= 500:
        raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure") from None
    raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure") from None


def _execute_transport(transport_fn: Any) -> object:
    try:
        return transport_fn()
    except (
        IntegrationConfigurationError,
        IntegrationDependencyError,
        MsGraphTeamsChannelMessageChanged,
        MsGraphTeamsChannelHostedContentTooLarge,
    ):
        raise
    except Exception:
        raise IntegrationDependencyError(
            "Microsoft Graph knowledge dependency is unavailable"
        ) from None


def _validate_max_bytes(max_bytes: object) -> int:
    if type(max_bytes) is not int:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST)
    if max_bytes < 1 or max_bytes > ABSOLUTE_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST)
    return max_bytes


def _hosted_content_matches_message(
    hosted: MsGraphTeamsChannelHostedContent,
    message: MsGraphTeamsChannelMessage,
) -> bool:
    return (
        hosted.team_remote_id == message.team_remote_id
        and hosted.channel_remote_id == message.channel_remote_id
        and hosted.message_remote_id == message.remote_id
        and hosted.thread_root_remote_id == message.thread_root_remote_id
        and hosted.message_kind == message.message_kind
        and hosted.message_revision == message.revision
    )


@runtime_checkable
class MsGraphTeamsChannelHostedContentReadClient(Protocol):
    def read_teams_channel_hosted_contents_page(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphTeamsChannelHostedContentPage:
        ...

    def read_teams_channel_hosted_content_bytes(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        hosted_content: MsGraphTeamsChannelHostedContent,
        max_bytes: int,
    ) -> MsGraphTeamsChannelHostedContentBytes:
        ...


class MsGraphTeamsChannelHostedContentReader:
    """Teams channel hosted content inventory and bounded byte download."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
        graph_http_client: Any,
    ) -> None:
        self._config = config
        self._transport = transport
        self._graph_http_client = graph_http_client

    def read_teams_channel_hosted_contents_page(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphTeamsChannelHostedContentPage:
        validated_message = _require_active_message(message)
        read_and_validate_current_teams_channel_message_observation(
            message=validated_message,
            transport=self._transport,
        )

        if continuation is None:
            path = _hosted_contents_inventory_path(validated_message)
            payload = self._transport.get_initial_json(
                path=path,
                headers=_PREFER_UNKNOWN_ENUM,
            )
        else:
            validated_continuation = validate_msgraph_teams_channel_hosted_contents_continuation(
                continuation,
                team_id=validated_message.team_remote_id,
                channel_id=validated_message.channel_remote_id,
                thread_root_id=validated_message.thread_root_remote_id,
                message_id=validated_message.remote_id,
                message_kind=validated_message.message_kind,
                graph_base_url=self._config.graph_base_url,
            )
            payload = self._transport.get_continuation_json(
                continuation=validated_continuation,
                headers=_PREFER_UNKNOWN_ENUM,
            )

        collection_page = parse_msgraph_collection_page(
            payload,
            graph_base_url=self._config.graph_base_url,
            delta_mode=False,
        )
        parsed_items = tuple(
            parse_msgraph_teams_channel_hosted_content(raw_item, message=validated_message)
            for raw_item in collection_page.items
        )
        page = validate_msgraph_teams_channel_hosted_content_page(
            _safe_construct_hosted_content_page(
                **_message_context_kwargs(validated_message),
                items=parsed_items,
                continuation=collection_page.continuation,
            ),
            message=validated_message,
            graph_base_url=self._config.graph_base_url,
        )

        read_and_validate_current_teams_channel_message_observation(
            message=validated_message,
            transport=self._transport,
        )
        return page

    def read_teams_channel_hosted_content_bytes(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        hosted_content: MsGraphTeamsChannelHostedContent,
        max_bytes: int,
    ) -> MsGraphTeamsChannelHostedContentBytes:
        validated_message = _require_active_message(message)
        validated_hosted = validate_msgraph_teams_channel_hosted_content(hosted_content)
        validated_max_bytes = _validate_max_bytes(max_bytes)

        if not _hosted_content_matches_message(validated_hosted, validated_message):
            raise MsGraphTeamsChannelMessageChanged() from None

        read_and_validate_current_teams_channel_message_observation(
            message=validated_message,
            transport=self._transport,
        )

        path = _hosted_content_value_path(validated_message, validated_hosted.remote_id)

        def _do_stream() -> object:
            return self._graph_http_client.stream(
                "GET",
                path,
                headers={"Accept": "application/octet-stream"},
                follow_redirects=False,
            )

        try:
            stream_context = _execute_transport(_do_stream)
        except Exception:
            raise IntegrationDependencyError(
                "Microsoft Graph knowledge dependency is unavailable"
            ) from None

        try:
            with stream_context as response:
                _raise_for_hosted_content_download_response(response)
                headers = _response_headers(response)
                content_length = _parse_content_length(headers)
                content_type = _parse_content_type_header(headers)
                if content_length is not None and content_length > validated_max_bytes:
                    raise MsGraphTeamsChannelHostedContentTooLarge() from None

                try:
                    iter_bytes = response.iter_bytes
                except AttributeError:
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                if not callable(iter_bytes):
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None

                buffer = bytearray()
                for chunk in iter_bytes():
                    if type(chunk) is not bytes:
                        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                    buffer.extend(chunk)
                    if len(buffer) > validated_max_bytes:
                        raise MsGraphTeamsChannelHostedContentTooLarge() from None

                data = bytes(buffer)
                if content_length is not None and len(data) != content_length:
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
        except (
            IntegrationConfigurationError,
            IntegrationDependencyError,
            MsGraphTeamsChannelMessageChanged,
            MsGraphTeamsChannelHostedContentTooLarge,
        ):
            raise
        except Exception:
            raise IntegrationDependencyError(
                "Microsoft Graph knowledge dependency is unavailable"
            ) from None

        read_and_validate_current_teams_channel_message_observation(
            message=validated_message,
            transport=self._transport,
        )

        content_hash = hashlib.sha256(data).hexdigest()
        return validate_msgraph_teams_channel_hosted_content_bytes(
            _safe_construct_hosted_content_bytes(
                **_message_context_kwargs(validated_message),
                hosted_content_remote_id=validated_hosted.remote_id,
                content_type=content_type,
                data=data,
                size_bytes=len(data),
                content_hash=content_hash,
            ),
            message=validated_message,
            hosted_content=validated_hosted,
            max_bytes=validated_max_bytes,
        )
