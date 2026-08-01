# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed safe models for Slack conversation knowledge reads."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
    _ASCII_CONTROL,
    _MALFORMED_RESPONSE,
    validate_optional_safe_text,
    validate_provider_cursor,
    validate_safe_text,
    validate_slack_conversation_id,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.timestamp import (
    compare_slack_timestamps,
    validate_slack_timestamp,
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)


class SlackConversationKind(StrEnum):
    PUBLIC_CHANNEL = "public_channel"
    PRIVATE_CHANNEL = "private_channel"
    IM = "im"
    MPIM = "mpim"


class SlackConversationSummary(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    conversation_id: str = Field(repr=False)
    kind: SlackConversationKind
    safe_name: str
    is_archived: bool
    is_private: bool
    created_at: datetime | None = None
    safe_topic: str | None = None
    safe_purpose: str | None = None

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _validate_conversation_id(cls, value: object) -> str:
        return validate_slack_conversation_id(value)

    @field_validator("safe_name", mode="before")
    @classmethod
    def _validate_safe_name(cls, value: object) -> str:
        return validate_safe_text(value)

    @field_validator("safe_topic", "safe_purpose", mode="before")
    @classmethod
    def _validate_optional_text(cls, value: object) -> str | None:
        return validate_optional_safe_text(value)


class SlackConversationInventoryPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    items: tuple[SlackConversationSummary, ...]
    next_cursor: str | None = Field(default=None, repr=False)

    @field_validator("next_cursor", mode="before")
    @classmethod
    def _validate_next_cursor(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_provider_cursor(value)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[SlackConversationSummary, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError(_MALFORMED_RESPONSE)
        return tuple(SlackConversationSummary.model_validate(item) for item in value)


class SlackConversationFileReference(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    file_id: str = Field(repr=False)
    safe_file_name: str
    title: str | None = None
    mimetype: str | None = None
    filetype: str | None = None
    size: int | None = None
    mode: str | None = None
    created_at: datetime | None = None
    is_external: bool = False

    @field_validator("file_id", mode="before")
    @classmethod
    def _validate_file_id(cls, value: object) -> str:
        return validate_safe_text(value, max_length=256)

    @field_validator("safe_file_name", mode="before")
    @classmethod
    def _validate_safe_file_name(cls, value: object) -> str:
        return validate_safe_text(value, max_length=4096)

    @field_validator("title", "mimetype", "filetype", "mode", mode="before")
    @classmethod
    def _validate_optional_fields(cls, value: object) -> str | None:
        return validate_optional_safe_text(value)

    @field_validator("size", mode="before")
    @classmethod
    def _validate_size(cls, value: object) -> int | None:
        if value is None:
            return None
        if type(value) is not int or value < 0:
            raise ValueError(_MALFORMED_RESPONSE)
        return value


class SlackConversationMessage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    conversation_id: str = Field(repr=False)
    message_ts: str = Field(repr=False)
    root_thread_ts: str | None = Field(default=None, repr=False)
    actor_provider_id: str | None = Field(default=None, repr=False)
    text: str
    subtype: str | None = None
    created_at: datetime
    edited_at: datetime | None = None
    reply_count: int | None = None
    files: tuple[SlackConversationFileReference, ...] = ()
    provider_metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _validate_conversation_id(cls, value: object) -> str:
        return validate_slack_conversation_id(value)

    @field_validator("message_ts", "root_thread_ts", mode="before")
    @classmethod
    def _validate_timestamps(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_slack_timestamp(value)

    @field_validator("actor_provider_id", mode="before")
    @classmethod
    def _validate_actor(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_safe_text(value, max_length=256)

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_RESPONSE)
        if _ASCII_CONTROL.search(value):
            raise ValueError(_MALFORMED_RESPONSE)
        return value

    @field_validator("subtype", mode="before")
    @classmethod
    def _validate_subtype(cls, value: object) -> str | None:
        return validate_optional_safe_text(value, max_length=128)

    @field_validator("reply_count", mode="before")
    @classmethod
    def _validate_reply_count(cls, value: object) -> int | None:
        if value is None:
            return None
        if type(value) is not int or value < 0:
            raise ValueError(_MALFORMED_RESPONSE)
        return value

    @field_validator("files", mode="before")
    @classmethod
    def _validate_files(cls, value: object) -> tuple[SlackConversationFileReference, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError(_MALFORMED_RESPONSE)
        return tuple(SlackConversationFileReference.model_validate(item) for item in value)

    @model_validator(mode="after")
    def _validate_thread_shape(self) -> Self:
        if self.root_thread_ts is not None and self.root_thread_ts == self.message_ts:
            raise ValueError(_MALFORMED_RESPONSE)
        return self


class SlackConversationMessagePage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    conversation_id: str = Field(repr=False)
    oldest: str = Field(repr=False)
    latest: str = Field(repr=False)
    items: tuple[SlackConversationMessage, ...]
    next_cursor: str | None = Field(default=None, repr=False)

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _validate_conversation_id(cls, value: object) -> str:
        return validate_slack_conversation_id(value)

    @field_validator("oldest", "latest", mode="before")
    @classmethod
    def _validate_boundaries(cls, value: object) -> str:
        return validate_slack_timestamp(value)

    @field_validator("next_cursor", mode="before")
    @classmethod
    def _validate_next_cursor(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_provider_cursor(value)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[SlackConversationMessage, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError(_MALFORMED_RESPONSE)
        return tuple(SlackConversationMessage.model_validate(item) for item in value)

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        if compare_slack_timestamps(self.oldest, self.latest) > 0:
            raise ValueError(_MALFORMED_RESPONSE)
        return self


class SlackConversationExactMessageResult(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    found: bool
    message: SlackConversationMessage | None = None

    @model_validator(mode="after")
    def _validate_result_shape(self) -> Self:
        if self.found and self.message is None:
            raise ValueError(_MALFORMED_RESPONSE)
        if not self.found and self.message is not None:
            raise ValueError(_MALFORMED_RESPONSE)
        return self


class SlackConversationSourceWindow(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    oldest: str = Field(repr=False)
    latest: str = Field(repr=False)

    @field_validator("oldest", "latest", mode="before")
    @classmethod
    def _validate_boundaries(cls, value: object) -> str:
        return validate_slack_timestamp(value)

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        ordering = compare_slack_timestamps(self.oldest, self.latest)
        if ordering > 0:
            raise ValueError(_MALFORMED_RESPONSE)
        if ordering == 0:
            raise ValueError(_MALFORMED_RESPONSE)
        return self


class SlackConversationPointWindow(BaseModel):
    """Inclusive exact timestamp point for provider exact-message reads."""

    model_config = _STRICT_MODEL_CONFIG

    message_ts: str = Field(repr=False)

    @field_validator("message_ts", mode="before")
    @classmethod
    def _validate_message_ts(cls, value: object) -> str:
        return validate_slack_timestamp(value)

    @property
    def oldest(self) -> str:
        return self.message_ts

    @property
    def latest(self) -> str:
        return self.message_ts


def validate_slack_conversation_message(value: object) -> SlackConversationMessage:
    if isinstance(value, SlackConversationMessage):
        return value
    try:
        return SlackConversationMessage.model_validate(value)
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_RESPONSE) from None


__all__ = [
    "SlackConversationExactMessageResult",
    "SlackConversationFileReference",
    "SlackConversationInventoryPage",
    "SlackConversationKind",
    "SlackConversationMessage",
    "SlackConversationMessagePage",
    "SlackConversationPointWindow",
    "SlackConversationSourceWindow",
    "SlackConversationSummary",
    "validate_slack_conversation_message",
]
