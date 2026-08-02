# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack Web API knowledge-read facet backed by the shared AsyncWebClient."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, runtime_checkable

from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
    MAX_HISTORY_REPLY_PAGE_LIMIT,
    MAX_INVENTORY_PAGE_LIMIT,
    _INVENTORY_TYPES,
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
    {
        "invalid_auth",
        "token_revoked",
        "not_authed",
        "account_inactive",
        "token_expired",
        "not_allowed_token_type",
    }
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
        "accesslimited",
        "enterprise_is_restricted",
        "ekm_access_denied",
        "org_login_required",
        "two_factor_setup_required",
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


def _require_ok_response(data: Mapping[str, Any]) -> None:
    if data.get("ok") is True:
        return
    if data.get("ok") is False:
        raise SlackConversationReadError(slack_error=str(data.get("error") or "unknown_error"))
    raise _malformed_provider_response()


def _non_blank_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    if value == "" or value != value.strip():
        return None
    return value


def _optional_bool(value: object, *, field: str, raw: Mapping[str, Any]) -> bool | None:
    if field not in raw:
        return None
    if type(value) is not bool:
        raise ValueError(_MALFORMED_RESPONSE)
    return value


def _optional_non_negative_int(value: object, *, field: str, raw: Mapping[str, Any]) -> int | None:
    if field not in raw:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(_MALFORMED_RESPONSE)
    return value


def _optional_str_field(value: object, *, field: str, raw: Mapping[str, Any]) -> str | None:
    if field not in raw:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_RESPONSE)
    return _non_blank_str(value)


def _optional_string_field(value: object, *, field: str, raw: Mapping[str, Any]) -> str | None:
    if field not in raw:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_RESPONSE)
    return value


def _optional_topic_purpose_value(
    obj: object,
    *,
    field: str,
    raw: Mapping[str, Any],
) -> str | None:
    if field not in raw:
        return None
    if not isinstance(obj, Mapping):
        raise ValueError(_MALFORMED_RESPONSE)
    if "value" not in obj:
        return None
    value = obj["value"]
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_RESPONSE)
    return _non_blank_str(value)


def _parse_next_cursor(data: Mapping[str, Any]) -> str | None:
    if "response_metadata" not in data:
        return None
    metadata = data["response_metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError(_MALFORMED_RESPONSE)
    if "next_cursor" not in metadata:
        return None
    raw_cursor = metadata["next_cursor"]
    if raw_cursor == "":
        return None
    if not isinstance(raw_cursor, str):
        raise ValueError(_MALFORMED_RESPONSE)
    return validate_provider_cursor(raw_cursor)


def _parse_present_thread_ts(raw: Mapping[str, Any], *, message_ts: str) -> str | None:
    if "thread_ts" not in raw:
        return None
    thread_ts_value = raw["thread_ts"]
    if not isinstance(thread_ts_value, str):
        raise ValueError(_MALFORMED_RESPONSE)
    validated_thread_ts = validate_slack_timestamp(thread_ts_value)
    return _normalize_root_thread_ts(
        message_ts=message_ts,
        raw_thread_ts=validated_thread_ts,
    )


def _parse_edited_at(raw: Mapping[str, Any]) -> datetime | None:
    if "edited" not in raw:
        return None
    edited = raw["edited"]
    if not isinstance(edited, Mapping):
        raise ValueError(_MALFORMED_RESPONSE)
    if "ts" not in edited:
        raise ValueError(_MALFORMED_RESPONSE)
    edited_ts_value = edited["ts"]
    if not isinstance(edited_ts_value, str):
        raise ValueError(_MALFORMED_RESPONSE)
    return _parse_created_at(validate_slack_timestamp(edited_ts_value))


def _parse_created_at(ts: str) -> datetime:
    parsed = parse_slack_ts(ts)
    if parsed is None:
        raise ValueError(_MALFORMED_RESPONSE)
    return parsed


def _conversation_kind_from_channel(channel: Mapping[str, Any]) -> SlackConversationKind:
    is_channel = _optional_bool(channel.get("is_channel"), field="is_channel", raw=channel)
    is_im = _optional_bool(channel.get("is_im"), field="is_im", raw=channel)
    is_mpim = _optional_bool(channel.get("is_mpim"), field="is_mpim", raw=channel)
    is_private = _optional_bool(channel.get("is_private"), field="is_private", raw=channel)
    active_kinds = sum(flag is True for flag in (is_channel, is_im, is_mpim))
    if active_kinds != 1:
        raise ValueError(_MALFORMED_RESPONSE)
    if is_im is True:
        if is_mpim is True:
            raise ValueError(_MALFORMED_RESPONSE)
        return SlackConversationKind.IM
    if is_mpim is True:
        if is_im is True:
            raise ValueError(_MALFORMED_RESPONSE)
        return SlackConversationKind.MPIM
    if is_channel is True:
        if is_private is True:
            return SlackConversationKind.PRIVATE_CHANNEL
        if is_private is False:
            return SlackConversationKind.PUBLIC_CHANNEL
        raise ValueError(_MALFORMED_RESPONSE)
    raise ValueError(_MALFORMED_RESPONSE)


def _safe_conversation_name(channel: Mapping[str, Any], *, kind: SlackConversationKind) -> str:
    if kind is SlackConversationKind.IM:
        user_id = _non_blank_str(_optional_string_field(channel.get("user"), field="user", raw=channel))
        if user_id is not None:
            return f"Direct message ({user_id[:8]}…)"
        return "Direct message"
    if kind is SlackConversationKind.MPIM:
        name = _non_blank_str(_optional_string_field(channel.get("name"), field="name", raw=channel))
        return name or "Group direct message"
    name = _non_blank_str(_optional_string_field(channel.get("name"), field="name", raw=channel))
    return name or "Conversation"


def _parse_file_reference(raw: Mapping[str, Any]) -> SlackConversationFileReference:
    file_id = _non_blank_str(raw.get("id"))
    if file_id is None:
        raise ValueError(_MALFORMED_RESPONSE)
    name = _optional_str_field(raw.get("name"), field="name", raw=raw)
    title = _optional_str_field(raw.get("title"), field="title", raw=raw)
    mimetype = _optional_str_field(raw.get("mimetype"), field="mimetype", raw=raw)
    filetype = _optional_str_field(raw.get("filetype"), field="filetype", raw=raw)
    mode = _optional_str_field(raw.get("mode"), field="mode", raw=raw)
    safe_name = name or title or f"file-{file_id[:8]}"
    created_at = None
    created_raw = raw.get("created")
    if "created" in raw:
        created_value = _optional_non_negative_int(created_raw, field="created", raw=raw)
        if created_value is not None:
            created_at = datetime.fromtimestamp(created_value, tz=timezone.utc)
    resolved_size = None
    if "size" in raw:
        resolved_size = _optional_non_negative_int(raw.get("size"), field="size", raw=raw)
    is_external = False
    if "is_external" in raw:
        external_value = _optional_bool(raw.get("is_external"), field="is_external", raw=raw)
        is_external = external_value is True
    return SlackConversationFileReference(
        file_id=file_id,
        safe_file_name=safe_name,
        title=title,
        mimetype=mimetype,
        filetype=filetype,
        size=resolved_size,
        mode=mode,
        created_at=created_at,
        is_external=is_external,
    )


def _normalize_root_thread_ts(*, message_ts: str, raw_thread_ts: str | None) -> str | None:
    if raw_thread_ts is None:
        return None
    if raw_thread_ts == message_ts:
        return None
    return raw_thread_ts


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
    root_thread_ts = _parse_present_thread_ts(raw, message_ts=message_ts)
    if "text" in raw:
        text_value = raw.get("text")
        if not isinstance(text_value, str):
            raise ValueError(_MALFORMED_RESPONSE)
        text = text_value
    else:
        text = ""
    if len(text) > max_chars:
        raise SlackConversationContentTooLarge()
    edited_at = _parse_edited_at(raw)
    files: list[SlackConversationFileReference] = []
    raw_files = raw.get("files")
    if isinstance(raw_files, list):
        for item in raw_files:
            if not isinstance(item, Mapping):
                raise ValueError(_MALFORMED_RESPONSE)
            files.append(_parse_file_reference(item))
    elif raw_files is not None:
        raise ValueError(_MALFORMED_RESPONSE)
    resolved_reply_count = None
    if "reply_count" in raw:
        resolved_reply_count = _optional_non_negative_int(
            raw.get("reply_count"),
            field="reply_count",
            raw=raw,
        )
    provider_metadata: dict[str, str] = {}
    if "is_starred" in raw:
        starred = _optional_bool(raw.get("is_starred"), field="is_starred", raw=raw)
        if starred is True:
            provider_metadata["is_starred"] = "true"
    subtype = _optional_str_field(raw.get("subtype"), field="subtype", raw=raw)
    actor_provider_id = _optional_str_field(raw.get("user"), field="user", raw=raw)
    return SlackConversationMessage(
        conversation_id=conversation_id,
        message_ts=message_ts,
        root_thread_ts=root_thread_ts,
        actor_provider_id=actor_provider_id,
        text=text,
        subtype=subtype,
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

    def __init__(self, web_client: Any) -> None:
        self._web_client = web_client

    async def list_accessible_conversations_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> SlackConversationInventoryPage:
        validated_limit = validate_page_limit(limit, maximum=MAX_INVENTORY_PAGE_LIMIT)
        validated_cursor = validate_provider_cursor(cursor) if cursor is not None else None
        params: dict[str, Any] = {
            "types": _INVENTORY_TYPES,
            "exclude_archived": False,
            "limit": validated_limit,
        }
        if validated_cursor is not None:
            params["cursor"] = validated_cursor
        try:
            response = await self._web_client.users_conversations(**params)
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        return self._parse_inventory_response(_response_mapping(response))

    def _parse_inventory_response(self, data: Mapping[str, Any]) -> SlackConversationInventoryPage:
        _require_ok_response(data)
        channels = data.get("channels")
        if not isinstance(channels, list):
            raise _malformed_provider_response()
        items: list[SlackConversationSummary] = []
        seen_ids: set[str] = set()
        for raw_channel in channels:
            if not isinstance(raw_channel, Mapping):
                raise _malformed_provider_response()
            try:
                conversation_id = _non_blank_str(raw_channel.get("id"))
                if conversation_id is None:
                    raise ValueError(_MALFORMED_RESPONSE)
                if conversation_id in seen_ids:
                    raise ValueError(_MALFORMED_RESPONSE)
                seen_ids.add(conversation_id)
                kind = _conversation_kind_from_channel(raw_channel)
                created_at = None
                if "created" in raw_channel:
                    created_value = _optional_non_negative_int(
                        raw_channel.get("created"),
                        field="created",
                        raw=raw_channel,
                    )
                    if created_value is not None:
                        created_at = datetime.fromtimestamp(created_value, tz=timezone.utc)
                is_archived = _optional_bool(
                    raw_channel.get("is_archived"),
                    field="is_archived",
                    raw=raw_channel,
                )
                is_private_flag = _optional_bool(
                    raw_channel.get("is_private"),
                    field="is_private",
                    raw=raw_channel,
                )
                safe_topic = _optional_topic_purpose_value(
                    raw_channel.get("topic"),
                    field="topic",
                    raw=raw_channel,
                )
                safe_purpose = _optional_topic_purpose_value(
                    raw_channel.get("purpose"),
                    field="purpose",
                    raw=raw_channel,
                )
                _optional_string_field(raw_channel.get("name"), field="name", raw=raw_channel)
                if kind is SlackConversationKind.IM:
                    _optional_string_field(raw_channel.get("user"), field="user", raw=raw_channel)
                items.append(
                    SlackConversationSummary(
                        conversation_id=validate_slack_conversation_id(conversation_id),
                        kind=kind,
                        safe_name=_safe_conversation_name(raw_channel, kind=kind),
                        is_archived=is_archived is True,
                        is_private=is_private_flag is True
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
            except ValueError as exc:
                raise _malformed_provider_response() from exc
        try:
            next_cursor = _parse_next_cursor(data)
        except ValueError as exc:
            raise _malformed_provider_response() from exc
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
        validated_conversation_id = validate_slack_conversation_id(conversation_id)
        validated_window = SlackConversationSourceWindow.model_validate(window.model_dump())
        validated_limit = validate_page_limit(limit, maximum=MAX_HISTORY_REPLY_PAGE_LIMIT)
        validated_max_chars = validate_message_max_chars(max_chars_per_message)
        validated_cursor = validate_provider_cursor(cursor) if cursor is not None else None
        params: dict[str, Any] = {
            "channel": validated_conversation_id,
            "oldest": validated_window.oldest,
            "latest": validated_window.latest,
            "inclusive": True,
            "limit": validated_limit,
        }
        if validated_cursor is not None:
            params["cursor"] = validated_cursor
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
        validated_conversation_id = validate_slack_conversation_id(conversation_id)
        validated_root = validate_slack_timestamp(root_message_ts)
        validated_window = SlackConversationSourceWindow.model_validate(window.model_dump())
        validated_limit = validate_page_limit(limit, maximum=MAX_HISTORY_REPLY_PAGE_LIMIT)
        validated_max_chars = validate_message_max_chars(max_chars_per_message)
        validated_cursor = validate_provider_cursor(cursor) if cursor is not None else None
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
            point_window=point_window,
            max_chars_per_message=validated_max_chars,
        )
        self._validate_exact_point_history_page(
            page=page,
            target_message_ts=validated_message_ts,
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
        point_window: SlackConversationPointWindow,
        max_chars_per_message: int,
    ) -> SlackConversationMessagePage:
        validated_max_chars = validate_message_max_chars(max_chars_per_message)
        params: dict[str, Any] = {
            "channel": conversation_id,
            "oldest": point_window.oldest,
            "latest": point_window.latest,
            "inclusive": True,
            "limit": 1,
        }
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
        root_message_ts: str,
        point_window: SlackConversationPointWindow,
        cursor: str | None,
        max_chars_per_message: int,
    ) -> SlackConversationMessagePage:
        validated_root = validate_slack_timestamp(root_message_ts)
        validated_max_chars = validate_message_max_chars(max_chars_per_message)
        validated_cursor = validate_provider_cursor(cursor) if cursor is not None else None
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
        try:
            response = await self._web_client.conversations_replies(**params)
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

    def _validate_exact_point_history_page(
        self,
        *,
        page: SlackConversationMessagePage,
        target_message_ts: str,
    ) -> None:
        seen_timestamps: set[str] = set()
        for item in page.items:
            if item.message_ts in seen_timestamps:
                raise _malformed_provider_response()
            seen_timestamps.add(item.message_ts)
            if item.message_ts != target_message_ts:
                raise _malformed_provider_response()
            if item.root_thread_ts is not None:
                raise _malformed_provider_response()

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
        validated_file_id = validate_safe_text(file_id, max_length=256)
        params: dict[str, Any] = {"file": validated_file_id}
        try:
            response = await self._web_client.files_info(**params)
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        data = _response_mapping(response)
        _require_ok_response(data)
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
        _require_ok_response(data)
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
        try:
            next_cursor = _parse_next_cursor(data)
        except ValueError as exc:
            raise _malformed_provider_response() from exc
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
