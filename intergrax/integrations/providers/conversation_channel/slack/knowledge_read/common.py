# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared constants and helpers for Slack conversation knowledge reads."""

from __future__ import annotations

import re

SLACK_CONVERSATION_SOURCE_KIND = "slack_conversation"

MAX_HISTORY_REPLY_PAGE_LIMIT = 15
MAX_INVENTORY_PAGE_LIMIT = 200
DEFAULT_MESSAGE_MAX_CHARS = 2_000_000
ABSOLUTE_MESSAGE_MAX_CHARS = 8_000_000

_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MAX_CONVERSATION_ID_LEN = 256
_MAX_SAFE_TEXT_LEN = 32_768
_MAX_FILE_ID_LEN = 256
_MAX_FILE_NAME_LEN = 4096
_MAX_MIMETYPE_LEN = 256
_MAX_FILETYPE_LEN = 128
_MAX_SUBTYPE_LEN = 128
_MAX_ACTOR_ID_LEN = 256
_MAX_PROVIDER_CURSOR_LEN = 4096
_MALFORMED_RESPONSE = "unexpected Slack conversation knowledge response"
_INVALID_REQUEST = "invalid Slack conversation knowledge request"

_METADATA_ALLOWLIST = frozenset(
    {
        "subtype",
        "has_files",
        "reply_count",
        "created_at",
        "edited_at",
        "thread_root_ts",
        "attachment_inventory_in_content",
    }
)


def validate_slack_conversation_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_RESPONSE)
    if value == "":
        raise ValueError(_MALFORMED_RESPONSE)
    if value != value.strip():
        raise ValueError(_MALFORMED_RESPONSE)
    if _ASCII_CONTROL.search(value):
        raise ValueError(_MALFORMED_RESPONSE)
    if len(value) > _MAX_CONVERSATION_ID_LEN:
        raise ValueError(_MALFORMED_RESPONSE)
    return value


def validate_safe_text(value: object, *, max_length: int = _MAX_SAFE_TEXT_LEN) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_RESPONSE)
    if value != value.strip():
        raise ValueError(_MALFORMED_RESPONSE)
    if _ASCII_CONTROL.search(value):
        raise ValueError(_MALFORMED_RESPONSE)
    if len(value) > max_length:
        raise ValueError(_MALFORMED_RESPONSE)
    return value


def validate_optional_safe_text(
    value: object,
    *,
    max_length: int = _MAX_SAFE_TEXT_LEN,
) -> str | None:
    if value is None:
        return None
    text = validate_safe_text(value, max_length=max_length)
    return text or None


def validate_provider_cursor(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_RESPONSE)
    if value == "":
        raise ValueError(_MALFORMED_RESPONSE)
    if value != value.strip():
        raise ValueError(_MALFORMED_RESPONSE)
    if _ASCII_CONTROL.search(value):
        raise ValueError(_MALFORMED_RESPONSE)
    if len(value) > _MAX_PROVIDER_CURSOR_LEN:
        raise ValueError(_MALFORMED_RESPONSE)
    return value


def validate_message_max_chars(value: object) -> int:
    if type(value) is not int:
        raise ValueError(_INVALID_REQUEST)
    if value < 1 or value > ABSOLUTE_MESSAGE_MAX_CHARS:
        raise ValueError(_INVALID_REQUEST)
    return value


def validate_page_limit(value: object, *, maximum: int) -> int:
    if type(value) is not int:
        raise ValueError(_INVALID_REQUEST)
    if value < 1 or value > maximum:
        raise ValueError(_INVALID_REQUEST)
    return value


__all__ = [
    "ABSOLUTE_MESSAGE_MAX_CHARS",
    "DEFAULT_MESSAGE_MAX_CHARS",
    "MAX_HISTORY_REPLY_PAGE_LIMIT",
    "MAX_INVENTORY_PAGE_LIMIT",
    "SLACK_CONVERSATION_SOURCE_KIND",
    "_METADATA_ALLOWLIST",
    "validate_message_max_chars",
    "validate_optional_safe_text",
    "validate_page_limit",
    "validate_provider_cursor",
    "validate_safe_text",
    "validate_slack_conversation_id",
]
