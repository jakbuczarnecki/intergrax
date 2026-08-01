# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack Web API knowledge-read facet backed by the shared AsyncWebClient."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, runtime_checkable

from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
    MAX_HISTORY_REPLY_PAGE_LIMIT,
    MAX_INVENTORY_PAGE_LIMIT,
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
    SlackConversationSourceWindow,
    SlackConversationSummary,
    validate_slack_conversation_message,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.timestamp import (
    validate_slack_timestamp,
)
from intergrax.integrations.providers.conversation_channel.slack.mapping import parse_slack_ts
from intergrax.utils import attribute_access

_LOG = logging.getLogger(__name__)
_INVENTORY_TYPES = "public_channel,private_channel,im,mpim"
_AUTH_ERRORS = frozenset({"invalid_auth", "token_revoked", "not_authed", "account_inactive"})
_SCOPE_ERRORS = frozenset({"missing_scope"})
_NOT_FOUND_ERRORS = frozenset({"channel_not_found", "thread_not_found", "message_not_found"})
_PERMISSION_ERRORS = frozenset(
    {"no_permission", "not_in_channel", "access_denied", "restricted_action"}
)
_RETRYABLE_ERRORS = frozenset(
    {"ratelimited", "request_timeout", "service_unavailable", "internal_error", "fatal_error"}
)
_METADATA_ALLOWLIST = frozenset({"is_starred"})


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
            }
        except Exception:
            raise SlackConversationReadError(slack_error="malformed_response") from None
    raise SlackConversationReadError(slack_error="malformed_response")


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
    code = ""
    response = attribute_access.optional(exc, "response", None)
    if isinstance(response, Mapping):
        code = str(response.get("error") or "")
    if not code:
        code = "unknown_error"
    retry_after = _extract_retry_after(exc)
    if code in _AUTH_ERRORS:
        return SlackConversationReadConfigurationError(
            "Slack conversation knowledge authentication failed",
        )
    if code in _SCOPE_ERRORS:
        return SlackConversationReadConfigurationError(
            "Slack conversation knowledge scope is missing",
        )
    if code in _NOT_FOUND_ERRORS:
        return SlackConversationMessageNotFound()
    if code in _PERMISSION_ERRORS:
        return SlackConversationReadError(slack_error=code, retry_after_seconds=retry_after)
    if code == "ratelimited":
        return SlackConversationReadError(slack_error=code, retry_after_seconds=retry_after)
    if code in _RETRYABLE_ERRORS:
        return SlackConversationReadError(slack_error=code, retry_after_seconds=retry_after)
    return SlackConversationReadError(slack_error=code, retry_after_seconds=retry_after)


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


def _parse_file_reference(raw: Mapping[str, Any]) -> SlackConversationFileReference | None:
    file_id = _non_blank_str(raw.get("id"))
    if file_id is None:
        return None
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
    root_thread_ts = _non_blank_str(raw.get("thread_ts"))
    if root_thread_ts is not None:
        root_thread_ts = validate_slack_timestamp(root_thread_ts)
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
            if isinstance(item, Mapping):
                parsed_file = _parse_file_reference(item)
                if parsed_file is not None:
                    files.append(parsed_file)
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
        message_ts: str,
        root_thread_ts: str | None,
        window: SlackConversationSourceWindow,
        expected_revision: str | None,
        max_chars_per_message: int,
    ) -> SlackConversationExactMessageResult:
        ...

    async def read_file_info(self, *, file_id: str) -> SlackConversationFileReference:
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
            response = await self._web_client.conversations_list(**params)
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        data = _response_mapping(response)
        if data.get("ok") is False:
            raise SlackConversationReadError(slack_error=str(data.get("error") or "unknown_error"))
        channels = data.get("channels")
        if not isinstance(channels, list):
            raise SlackConversationReadError(slack_error="malformed_response")
        items: list[SlackConversationSummary] = []
        for raw_channel in channels:
            if not isinstance(raw_channel, Mapping):
                continue
            conversation_id = _non_blank_str(raw_channel.get("id"))
            if conversation_id is None:
                continue
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
            window=validated_window,
            max_chars=validated_max_chars,
        )

    async def read_thread_replies_page(
        self,
        *,
        conversation_id: str,
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
            window=validated_window,
            max_chars=validated_max_chars,
        )
        deduped = tuple(
            item for item in page.items if item.message_ts != validated_root
        )
        return SlackConversationMessagePage(
            conversation_id=page.conversation_id,
            oldest=page.oldest,
            latest=page.latest,
            items=deduped,
            next_cursor=page.next_cursor,
        )

    async def read_exact_message(
        self,
        *,
        conversation_id: str,
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
        validated_window = SlackConversationSourceWindow.model_validate(window.model_dump())
        validated_max_chars = validate_message_max_chars(max_chars_per_message)
        if validated_root is not None and validated_root != validated_message_ts:
            page = await self.read_thread_replies_page(
                conversation_id=validated_conversation_id,
                root_message_ts=validated_root,
                window=validated_window,
                cursor=None,
                limit=MAX_HISTORY_REPLY_PAGE_LIMIT,
                max_chars_per_message=validated_max_chars,
            )
            for item in page.items:
                if item.message_ts == validated_message_ts:
                    return self._finalize_exact_read(
                        item,
                        expected_revision=expected_revision,
                    )
            return SlackConversationExactMessageResult(found=False, message=None)
        page = await self.read_conversation_history_page(
            conversation_id=validated_conversation_id,
            window=SlackConversationSourceWindow(
                oldest=validated_message_ts,
                latest=validated_message_ts,
            ),
            cursor=None,
            limit=1,
            max_chars_per_message=validated_max_chars,
        )
        for item in page.items:
            if item.message_ts == validated_message_ts:
                return self._finalize_exact_read(
                    item,
                    expected_revision=expected_revision,
                )
        return SlackConversationExactMessageResult(found=False, message=None)

    async def read_file_info(self, *, file_id: str) -> SlackConversationFileReference:
        validated_file_id = validate_safe_text(file_id, max_length=256)
        try:
            response = await self._web_client.files_info(file=validated_file_id)
        except Exception as exc:
            raise _normalize_slack_api_error(exc) from None
        data = _response_mapping(response)
        if data.get("ok") is False:
            raise SlackConversationReadError(slack_error=str(data.get("error") or "unknown_error"))
        file_obj = data.get("file")
        if not isinstance(file_obj, Mapping):
            raise SlackConversationReadError(slack_error="malformed_response")
        parsed = _parse_file_reference(file_obj)
        if parsed is None or parsed.file_id != validated_file_id:
            raise SlackConversationReadError(slack_error="malformed_response")
        return parsed

    def _parse_message_page(
        self,
        *,
        data: Mapping[str, Any],
        conversation_id: str,
        window: SlackConversationSourceWindow,
        max_chars: int,
    ) -> SlackConversationMessagePage:
        if data.get("ok") is False:
            raise SlackConversationReadError(slack_error=str(data.get("error") or "unknown_error"))
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise SlackConversationReadError(slack_error="malformed_response")
        items: list[SlackConversationMessage] = []
        for raw_message in messages:
            if not isinstance(raw_message, Mapping):
                continue
            items.append(
                _parse_message(
                    conversation_id=conversation_id,
                    raw=raw_message,
                    max_chars=max_chars,
                )
            )
        next_cursor = None
        metadata = data.get("response_metadata")
        if isinstance(metadata, Mapping):
            raw_cursor = _non_blank_str(metadata.get("next_cursor"))
            if raw_cursor:
                next_cursor = validate_provider_cursor(raw_cursor)
        return SlackConversationMessagePage(
            conversation_id=conversation_id,
            oldest=window.oldest,
            latest=window.latest,
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
