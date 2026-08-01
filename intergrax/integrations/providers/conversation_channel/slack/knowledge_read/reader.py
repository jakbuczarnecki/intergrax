# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack Web API knowledge-read facet backed by the shared AsyncWebClient."""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
    MAX_HISTORY_REPLY_PAGE_LIMIT,
    MAX_INVENTORY_PAGE_LIMIT,
    _INVENTORY_CHANNELS_TYPES,
    _INVENTORY_CURSOR_SCHEMA_VERSION,
    _INVENTORY_IM_TYPES,
    _INVENTORY_PHASE_CHANNELS,
    _INVENTORY_PHASE_IM_MPIM,
    _MALFORMED_RESPONSE,
    validate_message_max_chars,
    validate_page_limit,
    validate_provider_cursor,
    validate_safe_text,
    validate_slack_conversation_id,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.errors import (
    SlackConversationContentTooLarge,
    SlackConversationMessageChanged,
    SlackConversationMessageNotFound,
    SlackConversationReadConfigurationError,
    SlackConversationReadError,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.models import (
    SlackConversationExactMessageResult,
    SlackConversationFileReference,
    SlackConversationInventoryPage,
    SlackConversationKind,
    SlackConversationMessage,
    SlackConversationMessagePage,
    SlackConversationPointWindow,
    SlackConversationSourceWindow,
    SlackConversationSummary,
    validate_slack_conversation_message,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.timestamp import (
    slack_timestamp_in_window,
    validate_slack_timestamp,
)
from intergrax.integrations.providers.conversation_channel.slack.mapping import parse_slack_ts
from intergrax.utils import attribute_access

_LOG = logging.getLogger(__name__)
_AUTH_ERRORS = frozenset(
    {"invalid_auth", "token_revoked", "not_authed", "account_inactive", "token_expired"}
)
_SCOPE_ERRORS = frozenset({"missing_scope"})
_NOT_FOUND_ERRORS = frozenset({"channel_not_found", "thread_not_found", "message_not_found"})
_PERMISSION_ERRORS = frozenset(
    {
        "no_permission",
        "not_in_channel",
        "access_denied",
        "restricted_action",
        "team_access_not_granted",
    }
)
_RETRYABLE_ERRORS = frozenset(
    {"ratelimited", "request_timeout", "service_unavailable", "internal_error", "fatal_error"}
)


def _response_mapping(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    data = attribute_access.optional(response, "data", None)
    if isinstance(data, Mapping):
        return data
    response_get = attribute_access.optional(response, "get", None)
    if callable(response_get):
        try:
            return {
                "ok": response_get("ok"),
                "channels": response_get("channels"),
                "messages": response_get("messages"),
                "file": response_get("file"),
                "error": response_get("error"),
                "response_metadata": response_get("response_metadata"),
                "user_id": response_get("user_id"),
            }
        except Exception:
            raise SlackConversationReadError(slack_error="malformed_response") from None
    raise SlackConversationReadError(slack_error="malformed_response")


def _extract_slack_error_code(exc: BaseException) -> str:
    response = attribute_access.optional(exc, "response", None)
    if response is None:
        return "unknown_error"
    if isinstance(response, Mapping):
        code = response.get("error")
        if isinstance(code, str) and code.strip():
            return code.strip()
    data = attribute_access.optional(response, "data", None)
    if isinstance(data, Mapping):
        code = data.get("error")
        if isinstance(code, str) and code.strip():
            return code.strip()
    return "unknown_error"


def _extract_retry_after(exc: BaseException) -> float | None:
    response = attribute_access.optional(exc, "response", None)
    if response is None:
        return None
    headers = None
    if isinstance(response, Mapping):
        headers = response.get("headers")
    if headers is None:
        headers = attribute_access.optional(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        parsed = float(str(raw).strip())
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def _normalize_slack_api_error(exc: BaseException) -> BaseException:
    code = _extract_slack_error_code(exc)
    retry_after = _extract_retry_after(exc)
    if code in _AUTH_ERRORS:
        return SlackConversationReadError(slack_error=code, retry_after_seconds=retry_after)
    if code in _SCOPE_ERRORS or code in _PERMISSION_ERRORS:
        return SlackConversationReadError(slack_error=code, retry_after_seconds=retry_after)
    if code in _NOT_FOUND_ERRORS:
        return SlackConversationMessageNotFound()
    if code == "ratelimited":
        return SlackConversationReadError(slack_error=code, retry_after_seconds=retry_after)
    if code in _RETRYABLE_ERRORS:
        return SlackConversationReadError(slack_error=code, retry_after_seconds=retry_after)
    return SlackConversationReadError(slack_error=code, retry_after_seconds=retry_after)


def _malformed_provider_response() -> SlackConversationReadError:
    return SlackConversationReadError(slack_error="malformed_response")


def _non_blank_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    if value == "" or value != value.strip():
        return None
    return value


def _parse_created_at(ts: str) -> datetime:
    parsed = parse_slack_ts(ts)
    if parsed is None:
        raise ValueError(_MALFORMED_RESPONSE)
    return parsed


def _conversation_kind_from_channel(channel: Mapping[str, Any]) -> SlackConversationKind:
    if channel.get("is_im") is True:
        return SlackConversationKind.IM
    if channel.get("is_mpim") is True:
        return SlackConversationKind.MPIM
    if channel.get("is_private") is True:
        return SlackConversationKind.PRIVATE_CHANNEL
    return SlackConversationKind.PUBLIC_CHANNEL


def _safe_conversation_name(channel: Mapping[str, Any], *, kind: SlackConversationKind) -> str:
    if kind is SlackConversationKind.IM:
        user_id = _non_blank_str(channel.get("user"))
        if user_id is not None:
            return f"Direct message ({user_id[:8]}…)"
        return "Direct message"
    if kind is SlackConversationKind.MPIM:
        name = _non_blank_str(channel.get("name"))
        return name or "Group direct message"
    name = _non_blank_str(channel.get("name"))
    return name or "Conversation"


def _parse_file_reference(raw: Mapping[str, Any]) -> SlackConversationFileReference:
    file_id = _non_blank_str(raw.get("id"))
    if file_id is None:
        raise ValueError(_MALFORMED_RESPONSE)
    safe_name = (
        _non_blank_str(raw.get("name"))
        or _non_blank_str(raw.get("title"))
        or f"file-{file_id[:8]}"
    )
    created_at = None
    created_raw = raw.get("created")
    if isinstance(created_raw, int) and created_raw >= 0:
        created_at = datetime.fromtimestamp(created_raw, tz=timezone.utc)
    size = raw.get("size")
    resolved_size = size if type(size) is int and size >= 0 else None
    return SlackConversationFileReference(
        file_id=file_id,
        safe_file_name=safe_name,
        title=_non_blank_str(raw.get("title")),
        mimetype=_non_blank_str(raw.get("mimetype")),
        filetype=_non_blank_str(raw.get("filetype")),
        size=resolved_size,
        mode=_non_blank_str(raw.get("mode")),
        created_at=created_at,
        is_external=raw.get("is_external") is True,
    )


def _normalize_root_thread_ts(*, message_ts: str, raw_thread_ts: str | None) -> str | None:
    if raw_thread_ts is None:
        return None
    if raw_thread_ts == message_ts:
        return None
    return raw_thread_ts


def _kind_requires_user_token(kind: SlackConversationKind) -> bool:
    return kind in {
        SlackConversationKind.PUBLIC_CHANNEL,
        SlackConversationKind.PRIVATE_CHANNEL,
    }


def _encode_inventory_cursor_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_inventory_cursor_payload(value: str) -> dict[str, object]:
    padding = "=" * (-len(value) % 4)
    raw = base64.urlsafe_b64decode(value + padding)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(_MALFORMED_RESPONSE)
    return data


def _decode_composite_inventory_cursor(value: str) -> tuple[str, str | None]:
    data = _decode_inventory_cursor_payload(value)
    schema = data.get("schema_version")
    if schema != _INVENTORY_CURSOR_SCHEMA_VERSION:
        raise ValueError(_MALFORMED_RESPONSE)
    phase = data.get("inventory_phase")
    if phase not in {_INVENTORY_PHASE_CHANNELS, _INVENTORY_PHASE_IM_MPIM}:
        raise ValueError(_MALFORMED_RESPONSE)
    provider_cursor = data.get("provider_cursor")
    if provider_cursor is None:
        resolved_cursor = None
    elif isinstance(provider_cursor, str):
        resolved_cursor = validate_provider_cursor(provider_cursor)
    else:
        raise ValueError(_MALFORMED_RESPONSE)
    canonical = _encode_inventory_cursor_payload(
        {
            "schema_version": _INVENTORY_CURSOR_SCHEMA_VERSION,
            "inventory_phase": phase,
            "provider_cursor": resolved_cursor,
        }
    )
    if canonical != value:
        raise ValueError(_MALFORMED_RESPONSE)
    return str(phase), resolved_cursor


def _encode_composite_inventory_cursor(
    *,
    phase: Literal["channels", "im_mpim"],
    provider_cursor: str | None,
) -> str:
    return _encode_inventory_cursor_payload(
        {
            "schema_version": _INVENTORY_CURSOR_SCHEMA_VERSION,
            "inventory_phase": phase,
            "provider_cursor": provider_cursor,
        }
    )


def _parse_message(
    *,
    conversation_id: str,
    raw: Mapping[str, Any],
    max_chars: int,
) -> SlackConversationMessage:
    message_ts = _non_blank_str(raw.get("ts"))
    if message_ts is None:
        raise ValueError(_MALFORMED_RESPONSE)
    message_ts = validate_slack_timestamp(message_ts)
    raw_thread_ts = _non_blank_str(raw.get("thread_ts"))
    root_thread_ts = None
    if raw_thread_ts is not None:
        root_thread_ts = _normalize_root_thread_ts(
            message_ts=message_ts,
            raw_thread_ts=validate_slack_timestamp(raw_thread_ts),
        )
    text = raw.get("text")
    if not isinstance(text, str):
        text = ""
    if len(text) > max_chars:
        raise SlackConversationContentTooLarge()
    edited_at = None
    edited = raw.get("edited")
    if isinstance(edited, Mapping):
        edited_ts = _non_blank_str(edited.get("ts"))
        if edited_ts is not None:
            edited_at = _parse_created_at(validate_slack_timestamp(edited_ts))
    files: list[SlackConversationFileReference] = []
    raw_files = raw.get("files")
    if isinstance(raw_files, list):
        for item in raw_files:
            if not isinstance(item, Mapping):
                raise ValueError(_MALFORMED_RESPONSE)
            files.append(_parse_file_reference(item))
    elif raw_files is not None:
        raise ValueError(_MALFORMED_RESPONSE)
    reply_count = raw.get("reply_count")
    resolved_reply_count = reply_count if type(reply_count) is int and reply_count >= 0 else None
    provider_metadata: dict[str, str] = {}
    if raw.get("is_starred") is True:
        provider_metadata["is_starred"] = "true"
    return SlackConversationMessage(
        conversation_id=conversation_id,
        message_ts=message_ts,
        root_thread_ts=root_thread_ts,
        actor_provider_id=_non_blank_str(raw.get("user")),
        text=text,
        subtype=_non_blank_str(raw.get("subtype")),
        created_at=_parse_created_at(message_ts),
        edited_at=edited_at,
        reply_count=resolved_reply_count,
        files=tuple(files),
        provider_metadata=provider_metadata,
    )


@runtime_checkable
class SlackConversationKnowledgeReadClient(Protocol):
    async def list_accessible_conversations_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> SlackConversationInventoryPage:
        ...

    async def read_conversation_history_page(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        window: SlackConversationSourceWindow,
        cursor: str | None,
        limit: int,
        max_chars_per_message: int,
    ) -> SlackConversationMessagePage:
        ...

    async def read_thread_replies_page(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        root_message_ts: str,
        window: SlackConversationSourceWindow,
        cursor: str | None,
        limit: int,
        max_chars_per_message: int,
    ) -> SlackConversationMessagePage:
        ...

    async def read_exact_message(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        message_ts: str,
        root_thread_ts: str | None,
        window: SlackConversationSourceWindow,
        expected_revision: str | None,
        max_chars_per_message: int,
    ) -> SlackConversationExactMessageResult:
        ...

    async def read_file_info(
        self,
        *,
        file_id: str,
        conversation_kind: SlackConversationKind | None = None,
    ) -> SlackConversationFileReference:
        ...


class SlackConversationKnowledgeReader:
    """Knowledge-read facet using the shared Slack AsyncWebClient."""

    def __init__(
        self,
        web_client: Any,
        *,
        knowledge_user_token: str | None = None,
    ) -> None:
        self._web_client = web_client
        self._knowledge_user_token = knowledge_user_token.strip() if knowledge_user_token else None
        if self._knowledge_user_token == "":
            self._knowledge_user_token = None
        self._bot_team_id: str | None = None
        self._knowledge_team_id: str | None = None

    def _dual_inventory_streams_enabled(self) -> bool:
        return self._knowledge_user_token is not None

    def _require_user_token_for_kind(self, conversation_kind: SlackConversationKind) -> None:
        if _kind_requires_user_token(conversation_kind) and self._knowledge_user_token is None:
            raise SlackConversationReadConfigurationError(
                "Slack conversation knowledge read requires knowledge_user_token "
                "for public or private channel operations",
            )

    def _token_override_for_kind(self, conversation_kind: SlackConversationKind) -> str | None:
        if _kind_requires_user_token(conversation_kind):
            if self._knowledge_user_token is None:
                raise SlackConversationReadConfigurationError(
                    "Slack conversation knowledge read requires knowledge_user_token "
                    "for public or private channel operations",
                )
            return self._knowledge_user_token
        return None

    async def _resolve_workspace_team_ids(self) -> None:
        if self._knowledge_user_token is None:
            return
        if self._bot_team_id is not None and self._knowledge_team_id is not None:
            return
        try:
            bot_response = await self._web_client.auth_test()
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        bot_data = _response_mapping(bot_response)
        if bot_data.get("ok") is False:
            raise SlackConversationReadError(
                slack_error=str(bot_data.get("error") or "unknown_error"),
            )
        bot_team_id = _non_blank_str(bot_data.get("team_id"))
        if bot_team_id is None:
            raise _malformed_provider_response()
        try:
            user_response = await self._web_client.auth_test(token=self._knowledge_user_token)
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        user_data = _response_mapping(user_response)
        if user_data.get("ok") is False:
            raise SlackConversationReadError(
                slack_error=str(user_data.get("error") or "unknown_error"),
            )
        knowledge_team_id = _non_blank_str(user_data.get("team_id"))
        if knowledge_team_id is None:
            raise _malformed_provider_response()
        if bot_team_id != knowledge_team_id:
            raise SlackConversationReadConfigurationError(
                "Slack bot token and knowledge user token belong to different workspaces",
            )
        self._bot_team_id = bot_team_id
        self._knowledge_team_id = knowledge_team_id

    async def _users_conversations_page(
        self,
        *,
        types: str,
        cursor: str | None,
        limit: int,
        token_override: str | None,
    ) -> Mapping[str, Any]:
        validated_limit = validate_page_limit(limit, maximum=MAX_INVENTORY_PAGE_LIMIT)
        validated_cursor = validate_provider_cursor(cursor) if cursor is not None else None
        params: dict[str, Any] = {
            "types": types,
            "exclude_archived": False,
            "limit": validated_limit,
        }
        if validated_cursor is not None:
            params["cursor"] = validated_cursor
        if token_override is not None:
            params["token"] = token_override
        try:
            response = await self._web_client.users_conversations(**params)
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        return _response_mapping(response)

    async def list_accessible_conversations_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> SlackConversationInventoryPage:
        validated_limit = validate_page_limit(limit, maximum=MAX_INVENTORY_PAGE_LIMIT)
        if self._dual_inventory_streams_enabled():
            await self._resolve_workspace_team_ids()
            phase: str
            provider_cursor: str | None
            if cursor is None:
                phase = _INVENTORY_PHASE_CHANNELS
                provider_cursor = None
            else:
                phase, provider_cursor = _decode_composite_inventory_cursor(cursor)
            if phase == _INVENTORY_PHASE_CHANNELS:
                data = await self._users_conversations_page(
                    types=_INVENTORY_CHANNELS_TYPES,
                    cursor=provider_cursor,
                    limit=validated_limit,
                    token_override=self._knowledge_user_token,
                )
                page = self._parse_inventory_response(data)
                if page.next_cursor is not None:
                    next_cursor = _encode_composite_inventory_cursor(
                        phase=_INVENTORY_PHASE_CHANNELS,
                        provider_cursor=page.next_cursor,
                    )
                    return SlackConversationInventoryPage(
                        items=page.items,
                        next_cursor=next_cursor,
                    )
                im_cursor = _encode_composite_inventory_cursor(
                    phase=_INVENTORY_PHASE_IM_MPIM,
                    provider_cursor=None,
                )
                if page.items:
                    return SlackConversationInventoryPage(
                        items=page.items,
                        next_cursor=im_cursor,
                    )
                im_data = await self._users_conversations_page(
                    types=_INVENTORY_IM_TYPES,
                    cursor=None,
                    limit=validated_limit,
                    token_override=None,
                )
                im_page = self._parse_inventory_response(im_data)
                if im_page.next_cursor is not None:
                    return SlackConversationInventoryPage(
                        items=im_page.items,
                        next_cursor=_encode_composite_inventory_cursor(
                            phase=_INVENTORY_PHASE_IM_MPIM,
                            provider_cursor=im_page.next_cursor,
                        ),
                    )
                return SlackConversationInventoryPage(items=im_page.items, next_cursor=None)
            data = await self._users_conversations_page(
                types=_INVENTORY_IM_TYPES,
                cursor=provider_cursor,
                limit=validated_limit,
                token_override=None,
            )
            page = self._parse_inventory_response(data)
            if page.next_cursor is not None:
                return SlackConversationInventoryPage(
                    items=page.items,
                    next_cursor=_encode_composite_inventory_cursor(
                        phase=_INVENTORY_PHASE_IM_MPIM,
                        provider_cursor=page.next_cursor,
                    ),
                )
            return page
        validated_cursor = validate_provider_cursor(cursor) if cursor is not None else None
        data = await self._users_conversations_page(
            types=_INVENTORY_IM_TYPES,
            cursor=validated_cursor,
            limit=validated_limit,
            token_override=None,
        )
        return self._parse_inventory_response(data)

    def _parse_inventory_response(self, data: Mapping[str, Any]) -> SlackConversationInventoryPage:
        if data.get("ok") is False:
            raise SlackConversationReadError(slack_error=str(data.get("error") or "unknown_error"))
        channels = data.get("channels")
        if not isinstance(channels, list):
            raise _malformed_provider_response()
        items: list[SlackConversationSummary] = []
        for raw_channel in channels:
            if not isinstance(raw_channel, Mapping):
                raise _malformed_provider_response()
            conversation_id = _non_blank_str(raw_channel.get("id"))
            if conversation_id is None:
                raise _malformed_provider_response()
            kind = _conversation_kind_from_channel(raw_channel)
            created_at = None
            created_raw = raw_channel.get("created")
            if isinstance(created_raw, int) and created_raw >= 0:
                created_at = datetime.fromtimestamp(created_raw, tz=timezone.utc)
            topic_obj = raw_channel.get("topic")
            purpose_obj = raw_channel.get("purpose")
            safe_topic = (
                _non_blank_str(topic_obj.get("value")) if isinstance(topic_obj, Mapping) else None
            )
            safe_purpose = (
                _non_blank_str(purpose_obj.get("value"))
                if isinstance(purpose_obj, Mapping)
                else None
            )
            items.append(
                SlackConversationSummary(
                    conversation_id=validate_slack_conversation_id(conversation_id),
                    kind=kind,
                    safe_name=_safe_conversation_name(raw_channel, kind=kind),
                    is_archived=raw_channel.get("is_archived") is True,
                    is_private=raw_channel.get("is_private") is True
                    or kind
                    in {
                        SlackConversationKind.PRIVATE_CHANNEL,
                        SlackConversationKind.IM,
                        SlackConversationKind.MPIM,
                    },
                    created_at=created_at,
                    safe_topic=safe_topic,
                    safe_purpose=safe_purpose,
                )
            )
        next_cursor = None
        metadata = data.get("response_metadata")
        if isinstance(metadata, Mapping):
            raw_cursor = _non_blank_str(metadata.get("next_cursor"))
            if raw_cursor:
                next_cursor = validate_provider_cursor(raw_cursor)
        return SlackConversationInventoryPage(items=tuple(items), next_cursor=next_cursor)

    async def read_conversation_history_page(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        window: SlackConversationSourceWindow,
        cursor: str | None,
        limit: int,
        max_chars_per_message: int,
    ) -> SlackConversationMessagePage:
        self._require_user_token_for_kind(conversation_kind)
        if self._knowledge_user_token is not None:
            await self._resolve_workspace_team_ids()
        validated_conversation_id = validate_slack_conversation_id(conversation_id)
        validated_window = SlackConversationSourceWindow.model_validate(window.model_dump())
        validated_limit = validate_page_limit(limit, maximum=MAX_HISTORY_REPLY_PAGE_LIMIT)
        validated_max_chars = validate_message_max_chars(max_chars_per_message)
        validated_cursor = validate_provider_cursor(cursor) if cursor is not None else None
        token_override = self._token_override_for_kind(conversation_kind)
        params: dict[str, Any] = {
            "channel": validated_conversation_id,
            "oldest": validated_window.oldest,
            "latest": validated_window.latest,
            "inclusive": True,
            "limit": validated_limit,
        }
        if validated_cursor is not None:
            params["cursor"] = validated_cursor
        if token_override is not None:
            params["token"] = token_override
        try:
            response = await self._web_client.conversations_history(**params)
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        return self._parse_message_page(
            data=_response_mapping(response),
            conversation_id=validated_conversation_id,
            oldest=validated_window.oldest,
            latest=validated_window.latest,
            max_chars=validated_max_chars,
        )

    async def read_thread_replies_page(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        root_message_ts: str,
        window: SlackConversationSourceWindow,
        cursor: str | None,
        limit: int,
        max_chars_per_message: int,
    ) -> SlackConversationMessagePage:
        self._require_user_token_for_kind(conversation_kind)
        if self._knowledge_user_token is not None:
            await self._resolve_workspace_team_ids()
        validated_conversation_id = validate_slack_conversation_id(conversation_id)
        validated_root = validate_slack_timestamp(root_message_ts)
        validated_window = SlackConversationSourceWindow.model_validate(window.model_dump())
        validated_limit = validate_page_limit(limit, maximum=MAX_HISTORY_REPLY_PAGE_LIMIT)
        validated_max_chars = validate_message_max_chars(max_chars_per_message)
        validated_cursor = validate_provider_cursor(cursor) if cursor is not None else None
        token_override = self._token_override_for_kind(conversation_kind)
        params: dict[str, Any] = {
            "channel": validated_conversation_id,
            "ts": validated_root,
            "oldest": validated_window.oldest,
            "latest": validated_window.latest,
            "inclusive": True,
            "limit": validated_limit,
        }
        if validated_cursor is not None:
            params["cursor"] = validated_cursor
        if token_override is not None:
            params["token"] = token_override
        try:
            response = await self._web_client.conversations_replies(**params)
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        page = self._parse_message_page(
            data=_response_mapping(response),
            conversation_id=validated_conversation_id,
            oldest=validated_window.oldest,
            latest=validated_window.latest,
            max_chars=validated_max_chars,
        )
        return self._filter_thread_replies_page(
            page=page,
            requested_root=validated_root,
            require_root=validated_cursor is None,
        )

    async def read_exact_message(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        message_ts: str,
        root_thread_ts: str | None,
        window: SlackConversationSourceWindow,
        expected_revision: str | None,
        max_chars_per_message: int,
    ) -> SlackConversationExactMessageResult:
        self._require_user_token_for_kind(conversation_kind)
        if self._knowledge_user_token is not None:
            await self._resolve_workspace_team_ids()
        validated_conversation_id = validate_slack_conversation_id(conversation_id)
        validated_message_ts = validate_slack_timestamp(message_ts)
        validated_root = (
            validate_slack_timestamp(root_thread_ts) if root_thread_ts is not None else None
        )
        SlackConversationSourceWindow.model_validate(window.model_dump())
        validated_max_chars = validate_message_max_chars(max_chars_per_message)
        point_window = SlackConversationPointWindow(message_ts=validated_message_ts)
        if validated_root is not None and validated_root != validated_message_ts:
            cursor: str | None = None
            seen_cursors: set[str] = set()
            while True:
                page = await self._read_thread_replies_point_page(
                    conversation_id=validated_conversation_id,
                    conversation_kind=conversation_kind,
                    root_message_ts=validated_root,
                    point_window=point_window,
                    cursor=cursor,
                    max_chars_per_message=validated_max_chars,
                )
                self._validate_exact_point_replies_page(
                    page=page,
                    requested_root=validated_root,
                    target_message_ts=validated_message_ts,
                    cursor=cursor,
                    seen_cursors=seen_cursors,
                )
                for item in page.items:
                    if item.message_ts != validated_message_ts:
                        continue
                    if item.root_thread_ts != validated_root:
                        raise _malformed_provider_response()
                    return self._finalize_exact_read(
                        item,
                        expected_revision=expected_revision,
                    )
                if page.next_cursor is None:
                    break
                if page.next_cursor == cursor:
                    raise _malformed_provider_response()
                cursor = page.next_cursor
            return SlackConversationExactMessageResult(found=False, message=None)
        page = await self._read_history_point_page(
            conversation_id=validated_conversation_id,
            conversation_kind=conversation_kind,
            point_window=point_window,
            max_chars_per_message=validated_max_chars,
        )
        for item in page.items:
            if item.message_ts != validated_message_ts:
                continue
            if item.root_thread_ts is not None:
                raise _malformed_provider_response()
            return self._finalize_exact_read(
                item,
                expected_revision=expected_revision,
            )
        return SlackConversationExactMessageResult(found=False, message=None)

    async def _read_history_point_page(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        point_window: SlackConversationPointWindow,
        max_chars_per_message: int,
    ) -> SlackConversationMessagePage:
        validated_max_chars = validate_message_max_chars(max_chars_per_message)
        token_override = self._token_override_for_kind(conversation_kind)
        params: dict[str, Any] = {
            "channel": conversation_id,
            "oldest": point_window.oldest,
            "latest": point_window.latest,
            "inclusive": True,
            "limit": 1,
        }
        if token_override is not None:
            params["token"] = token_override
        try:
            response = await self._web_client.conversations_history(**params)
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        return self._parse_message_page(
            data=_response_mapping(response),
            conversation_id=conversation_id,
            oldest=point_window.oldest,
            latest=point_window.latest,
            max_chars=validated_max_chars,
            enforce_window=False,
        )

    async def _read_thread_replies_point_page(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        root_message_ts: str,
        point_window: SlackConversationPointWindow,
        cursor: str | None,
        max_chars_per_message: int,
    ) -> SlackConversationMessagePage:
        validated_root = validate_slack_timestamp(root_message_ts)
        validated_max_chars = validate_message_max_chars(max_chars_per_message)
        validated_cursor = validate_provider_cursor(cursor) if cursor is not None else None
        token_override = self._token_override_for_kind(conversation_kind)
        params: dict[str, Any] = {
            "channel": conversation_id,
            "ts": validated_root,
            "oldest": point_window.oldest,
            "latest": point_window.latest,
            "inclusive": True,
            "limit": MAX_HISTORY_REPLY_PAGE_LIMIT,
        }
        if validated_cursor is not None:
            params["cursor"] = validated_cursor
        if token_override is not None:
            params["token"] = token_override
        try:
            response = await self._web_client.conversations_replies(**params)
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        page = self._parse_message_page(
            data=_response_mapping(response),
            conversation_id=conversation_id,
            oldest=point_window.oldest,
            latest=point_window.latest,
            max_chars=validated_max_chars,
            enforce_window=False,
        )
        return page

    def _filter_exact_point_replies_page(
        self,
        *,
        page: SlackConversationMessagePage,
        requested_root: str,
    ) -> SlackConversationMessagePage:
        """Strip optional root from exact point-query pages; neighbors remain invalid."""
        replies: list[SlackConversationMessage] = []
        seen_timestamps: set[str] = set()
        for item in page.items:
            if item.message_ts in seen_timestamps:
                raise _malformed_provider_response()
            seen_timestamps.add(item.message_ts)
            if item.message_ts == requested_root:
                if item.root_thread_ts is not None:
                    raise _malformed_provider_response()
                continue
            if item.root_thread_ts != requested_root:
                raise _malformed_provider_response()
            replies.append(item)
        return SlackConversationMessagePage(
            conversation_id=page.conversation_id,
            oldest=page.oldest,
            latest=page.latest,
            items=tuple(replies),
            next_cursor=page.next_cursor,
        )

    def _validate_exact_point_replies_page(
        self,
        *,
        page: SlackConversationMessagePage,
        requested_root: str,
        target_message_ts: str,
        cursor: str | None,
        seen_cursors: set[str],
    ) -> None:
        if cursor is not None:
            if cursor in seen_cursors:
                raise _malformed_provider_response()
            seen_cursors.add(cursor)
        seen_timestamps: set[str] = set()
        for item in page.items:
            if item.message_ts in seen_timestamps:
                raise _malformed_provider_response()
            seen_timestamps.add(item.message_ts)
            if item.message_ts == requested_root:
                if item.root_thread_ts is not None:
                    raise _malformed_provider_response()
                continue
            if item.root_thread_ts != requested_root:
                raise _malformed_provider_response()
            if item.message_ts != target_message_ts:
                raise _malformed_provider_response()

    async def read_file_info(
        self,
        *,
        file_id: str,
        conversation_kind: SlackConversationKind | None = None,
    ) -> SlackConversationFileReference:
        if conversation_kind is not None:
            self._require_user_token_for_kind(conversation_kind)
            if self._knowledge_user_token is not None:
                await self._resolve_workspace_team_ids()
        validated_file_id = validate_safe_text(file_id, max_length=256)
        token_override = (
            self._token_override_for_kind(conversation_kind) if conversation_kind is not None else None
        )
        params: dict[str, Any] = {"file": validated_file_id}
        if token_override is not None:
            params["token"] = token_override
        try:
            response = await self._web_client.files_info(**params)
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        data = _response_mapping(response)
        if data.get("ok") is False:
            raise SlackConversationReadError(slack_error=str(data.get("error") or "unknown_error"))
        file_obj = data.get("file")
        if not isinstance(file_obj, Mapping):
            raise _malformed_provider_response()
        parsed = _parse_file_reference(file_obj)
        if parsed.file_id != validated_file_id:
            raise _malformed_provider_response()
        return parsed

    def _filter_thread_replies_page(
        self,
        *,
        page: SlackConversationMessagePage,
        requested_root: str,
        require_root: bool,
    ) -> SlackConversationMessagePage:
        root_seen = False
        replies: list[SlackConversationMessage] = []
        seen_timestamps: set[str] = set()
        for item in page.items:
            if item.message_ts in seen_timestamps:
                raise _malformed_provider_response()
            seen_timestamps.add(item.message_ts)
            if item.message_ts == requested_root:
                if item.root_thread_ts is not None:
                    raise _malformed_provider_response()
                root_seen = True
                continue
            if item.root_thread_ts != requested_root:
                raise _malformed_provider_response()
            replies.append(item)
        if require_root and page.items and not root_seen:
            raise _malformed_provider_response()
        return SlackConversationMessagePage(
            conversation_id=page.conversation_id,
            oldest=page.oldest,
            latest=page.latest,
            items=tuple(replies),
            next_cursor=page.next_cursor,
        )

    def _parse_message_page(
        self,
        *,
        data: Mapping[str, Any],
        conversation_id: str,
        oldest: str,
        latest: str,
        max_chars: int,
        enforce_window: bool = True,
    ) -> SlackConversationMessagePage:
        if data.get("ok") is False:
            raise SlackConversationReadError(slack_error=str(data.get("error") or "unknown_error"))
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise _malformed_provider_response()
        items: list[SlackConversationMessage] = []
        seen_timestamps: set[str] = set()
        for raw_message in messages:
            if not isinstance(raw_message, Mapping):
                raise _malformed_provider_response()
            try:
                parsed = _parse_message(
                    conversation_id=conversation_id,
                    raw=raw_message,
                    max_chars=max_chars,
                )
            except ValueError as exc:
                raise _malformed_provider_response() from exc
            except SlackConversationContentTooLarge:
                raise
            if parsed.conversation_id != conversation_id:
                raise _malformed_provider_response()
            if parsed.message_ts in seen_timestamps:
                raise _malformed_provider_response()
            seen_timestamps.add(parsed.message_ts)
            if enforce_window and not slack_timestamp_in_window(
                value=parsed.message_ts,
                oldest=oldest,
                latest=latest,
            ):
                raise _malformed_provider_response()
            items.append(parsed)
        next_cursor = None
        metadata = data.get("response_metadata")
        if isinstance(metadata, Mapping):
            raw_cursor = _non_blank_str(metadata.get("next_cursor"))
            if raw_cursor:
                next_cursor = validate_provider_cursor(raw_cursor)
        return SlackConversationMessagePage(
            conversation_id=conversation_id,
            oldest=oldest,
            latest=latest,
            items=tuple(items),
            next_cursor=next_cursor,
        )

    def _finalize_exact_read(
        self,
        message: SlackConversationMessage,
        *,
        expected_revision: str | None,
    ) -> SlackConversationExactMessageResult:
        validated = validate_slack_conversation_message(message)
        if expected_revision is not None:
            actual_revision = compute_slack_conversation_message_revision(validated)
            if actual_revision != expected_revision:
                raise SlackConversationMessageChanged()
        return SlackConversationExactMessageResult(found=True, message=validated)


def compute_slack_conversation_message_revision(message: SlackConversationMessage) -> str:
    """Deterministic adapter-owned revision from canonical safe normalized fields."""
    import hashlib
    import json

    payload = {
        "conversation_id": message.conversation_id,
        "message_ts": message.message_ts,
        "root_thread_ts": message.root_thread_ts,
        "actor_provider_id": message.actor_provider_id,
        "text": message.text,
        "subtype": message.subtype,
        "edited_at": message.edited_at.isoformat() if message.edited_at is not None else None,
        "reply_count": message.reply_count,
        "files": [
            {
                "file_id": file_ref.file_id,
                "safe_file_name": file_ref.safe_file_name,
                "title": file_ref.title,
                "mimetype": file_ref.mimetype,
                "filetype": file_ref.filetype,
                "size": file_ref.size,
                "mode": file_ref.mode,
                "created_at": (
                    file_ref.created_at.isoformat() if file_ref.created_at is not None else None
                ),
                "is_external": file_ref.is_external,
            }
            for file_ref in message.files
        ],
        "provider_metadata": dict(sorted(message.provider_metadata.items())),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "SlackConversationKnowledgeReadClient",
    "SlackConversationKnowledgeReader",
    "compute_slack_conversation_message_revision",
]
