# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Conversation channel shared models and backend protocol.

A conversation channel is an external near-real-time communication system
that delivers human-originated conversation events to an application
and allows the application to reply within the same addressable
conversation context.

Zewnętrzny system komunikacji czasu zbliżonego do rzeczywistego,
który dostarcza aplikacji zdarzenia rozmowy pochodzące od człowieka
i pozwala aplikacji odpowiadać w kontekście tej samej adresowalnej rozmowy.

Category guarantees (v1):
- event_id is provider-scoped and installation-scoped (not globally unique).
- Product-level idempotency remains application-owned.
- Vendor transport acknowledgement stays inside the provider implementation.
- No exactly-once, read-receipt, durable queue, or strict ordering promises.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from intergrax.integrations.contracts.base import HealthStatus

MAX_CONVERSATION_CHOICE_OPTIONS = 25
MAX_OUTBOUND_TEXT_LENGTH = 4000
MAX_CHOICE_LABEL_LENGTH = 150
MAX_CHOICE_VALUE_LENGTH = 150
MAX_ACTION_ID_LENGTH = 150
MAX_SINGLE_CHOICE_COMPONENTS = 1
MAX_ATTACHMENT_ID_LENGTH = 255
MAX_ATTACHMENT_FILE_NAME_LENGTH = 255
MAX_ATTACHMENT_CONTENT_TYPE_LENGTH = 255

ConversationEventHandler = Callable[["InboundConversationEvent"], Awaitable[None]]


def _has_nul_or_control(value: str) -> bool:
    return any(ord(ch) < 32 for ch in value)


def _optional_safe_text(value: str | None, *, max_length: int) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if _has_nul_or_control(normalized):
        raise ValueError("must not contain NUL or ASCII control characters")
    if len(normalized) > max_length:
        raise ValueError(f"must be at most {max_length} characters")
    return normalized


def _require_safe_text(value: str, *, max_length: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be a non-blank string")
    if _has_nul_or_control(normalized):
        raise ValueError("must not contain NUL or ASCII control characters")
    if len(normalized) > max_length:
        raise ValueError(f"must be at most {max_length} characters")
    return normalized


class ConversationAttachmentReference(BaseModel):
    """Provider-scoped opaque attachment reference (no credentials or URLs)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_id: str
    file_name: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("attachment_id")
    @classmethod
    def _require_attachment_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("attachment_id must be a non-blank string")
        if len(normalized) > MAX_ATTACHMENT_ID_LENGTH:
            raise ValueError(
                f"attachment_id must be at most {MAX_ATTACHMENT_ID_LENGTH} characters"
            )
        return normalized

    @field_validator("file_name")
    @classmethod
    def _optional_file_name(cls, value: str | None) -> str | None:
        return _optional_safe_text(value, max_length=MAX_ATTACHMENT_FILE_NAME_LENGTH)

    @field_validator("content_type")
    @classmethod
    def _optional_content_type(cls, value: str | None) -> str | None:
        return _optional_safe_text(value, max_length=MAX_ATTACHMENT_CONTENT_TYPE_LENGTH)

    @field_validator("size_bytes")
    @classmethod
    def _optional_size_bytes(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("size_bytes must be an integer")
        if value < 0:
            raise ValueError("size_bytes must be >= 0")
        return value


class ConversationAttachmentContent(BaseModel):
    """Fetched attachment bytes with safe identity metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_id: str
    file_name: str
    content_type: str
    body: bytes

    @field_validator("attachment_id")
    @classmethod
    def _require_attachment_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("attachment_id must be a non-blank string")
        if len(normalized) > MAX_ATTACHMENT_ID_LENGTH:
            raise ValueError(
                f"attachment_id must be at most {MAX_ATTACHMENT_ID_LENGTH} characters"
            )
        return normalized

    @field_validator("file_name")
    @classmethod
    def _require_file_name(cls, value: str) -> str:
        return _require_safe_text(value, max_length=MAX_ATTACHMENT_FILE_NAME_LENGTH)

    @field_validator("content_type")
    @classmethod
    def _require_content_type(cls, value: str) -> str:
        return _require_safe_text(value, max_length=MAX_ATTACHMENT_CONTENT_TYPE_LENGTH)

    @field_validator("body", mode="before")
    @classmethod
    def _require_bytes_body(cls, value: Any) -> bytes:
        if type(value) is not bytes:
            raise ValueError("body must be bytes")
        return value


class ConversationAttachmentFetchError(RuntimeError):
    """Stable, provider-detail-free attachment fetch failure."""

    def __init__(self, *, kind: str) -> None:
        normalized = (kind or "").strip()
        if not normalized:
            raise ValueError("kind must be a non-blank string")
        super().__init__(normalized)
        self.kind = normalized


@runtime_checkable
class ConversationAttachmentFetcher(Protocol):
    """Optional provider capability for downloading attachment bytes."""

    async def fetch_attachment(
        self,
        attachment: ConversationAttachmentReference,
        *,
        max_bytes: int,
    ) -> ConversationAttachmentContent:
        """Fetch attachment content bounded by ``max_bytes``."""


class ConversationEventKind(str, Enum):
    MESSAGE = "message"
    ACTION = "action"


class ConversationAddress(BaseModel):
    """Vendor-neutral address for an addressable conversation context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    installation_id: str
    conversation_id: str
    thread_id: str | None = None

    @field_validator("installation_id", "conversation_id")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-blank string")
        return normalized

    @field_validator("thread_id")
    @classmethod
    def _optional_non_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("thread_id must be non-blank when provided")
        return normalized


class ConversationActor(BaseModel):
    """Provider-scoped actor identity (not a product authorization decision)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_id: str
    display_name: str | None = None
    is_bot: bool = False

    @field_validator("actor_id")
    @classmethod
    def _require_actor_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("actor_id must be a non-blank string")
        return normalized


class ConversationActionSelection(BaseModel):
    """Single-choice interaction response from a human."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    selected_value: str

    @field_validator("action_id", "selected_value")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-blank string")
        return normalized


class ConversationChoiceOption(BaseModel):
    """One option inside a single-choice component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str
    label: str

    @field_validator("value", "label")
    @classmethod
    def _require_bounded_non_blank(cls, value: str, info: Any) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-blank string")
        limit = MAX_CHOICE_VALUE_LENGTH if info.field_name == "value" else MAX_CHOICE_LABEL_LENGTH
        if len(normalized) > limit:
            raise ValueError(f"must be at most {limit} characters")
        return normalized


class ConversationSingleChoice(BaseModel):
    """v1 interactive component: text prompt plus one exclusive choice list."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    prompt: str | None = None
    options: tuple[ConversationChoiceOption, ...]

    @field_validator("action_id")
    @classmethod
    def _require_action_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("action_id must be a non-blank string")
        if len(normalized) > MAX_ACTION_ID_LENGTH:
            raise ValueError(f"action_id must be at most {MAX_ACTION_ID_LENGTH} characters")
        return normalized

    @field_validator("options")
    @classmethod
    def _validate_options(cls, value: tuple[ConversationChoiceOption, ...]) -> tuple[ConversationChoiceOption, ...]:
        if not value:
            raise ValueError("at least one option is required")
        if len(value) > MAX_CONVERSATION_CHOICE_OPTIONS:
            raise ValueError(f"at most {MAX_CONVERSATION_CHOICE_OPTIONS} options are allowed")
        option_values = [option.value for option in value]
        if len(option_values) != len(set(option_values)):
            raise ValueError("option values must be unique within one choice component")
        return value


class InboundConversationEvent(BaseModel):
    """Application-facing inbound conversation event (vendor-neutral)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    address: ConversationAddress
    actor: ConversationActor
    kind: ConversationEventKind
    text: str | None = None
    occurred_at: datetime | None = None
    action: ConversationActionSelection | None = None
    attachments: tuple[ConversationAttachmentReference, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id")
    @classmethod
    def _require_event_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("event_id must be a non-blank string")
        return normalized

    @model_validator(mode="after")
    def _validate_kind_payload(self) -> InboundConversationEvent:
        if self.kind is ConversationEventKind.MESSAGE:
            has_text = self.text is not None and bool(self.text.strip())
            has_attachments = len(self.attachments) > 0
            if not has_text and not has_attachments:
                raise ValueError(
                    "MESSAGE events require non-blank text or at least one attachment"
                )
            if self.action is not None:
                raise ValueError("MESSAGE events must not include action")
        elif self.kind is ConversationEventKind.ACTION:
            if self.action is None:
                raise ValueError("ACTION events require action selection")
            if self.attachments:
                raise ValueError("ACTION events must not include attachments")
        return self


class OutboundConversationMessage(BaseModel):
    """Outbound reply: plain text, optionally with one single-choice component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    address: ConversationAddress
    text: str
    components: tuple[ConversationSingleChoice, ...] = ()

    @field_validator("text")
    @classmethod
    def _require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must be a non-blank string")
        if len(normalized) > MAX_OUTBOUND_TEXT_LENGTH:
            raise ValueError(f"text must be at most {MAX_OUTBOUND_TEXT_LENGTH} characters")
        return normalized

    @field_validator("components")
    @classmethod
    def _validate_components(
        cls,
        value: tuple[ConversationSingleChoice, ...],
    ) -> tuple[ConversationSingleChoice, ...]:
        if len(value) > MAX_SINGLE_CHOICE_COMPONENTS:
            raise ValueError(
                f"at most {MAX_SINGLE_CHOICE_COMPONENTS} ConversationSingleChoice component(s) allowed in v1"
            )
        return value


class ConversationDeliveryReceipt(BaseModel):
    """Vendor accepted/created the outbound message — not a human read receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str
    address: ConversationAddress
    delivered_at: datetime | None = None

    @field_validator("message_id")
    @classmethod
    def _require_message_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message_id must be a non-blank string")
        return normalized


@runtime_checkable
class ConversationChannelBackend(Protocol):
    """Operational backend for conversation channel providers.

    ``start`` begins vendor event delivery and registers exactly one handler.
    ``stop`` closes vendor resources and should be safe after partial startup.
    ``send`` delivers text (and at most one single-choice component).
    ``health`` reports provider readiness only (no product readiness).
    Vendor acknowledgement and retries remain implementation-private.
    """

    async def start(self, handler: ConversationEventHandler) -> None:
        """Start vendor event delivery with the application handler."""

    async def stop(self) -> None:
        """Stop event delivery and release vendor resources."""

    async def send(self, message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        """Send an outbound message to the address in ``message``."""

    def health(self) -> HealthStatus | bool:
        """Return current provider readiness."""


__all__ = [
    "MAX_ACTION_ID_LENGTH",
    "MAX_ATTACHMENT_CONTENT_TYPE_LENGTH",
    "MAX_ATTACHMENT_FILE_NAME_LENGTH",
    "MAX_ATTACHMENT_ID_LENGTH",
    "MAX_CHOICE_LABEL_LENGTH",
    "MAX_CHOICE_VALUE_LENGTH",
    "MAX_CONVERSATION_CHOICE_OPTIONS",
    "MAX_OUTBOUND_TEXT_LENGTH",
    "MAX_SINGLE_CHOICE_COMPONENTS",
    "ConversationActionSelection",
    "ConversationActor",
    "ConversationAddress",
    "ConversationAttachmentContent",
    "ConversationAttachmentFetchError",
    "ConversationAttachmentFetcher",
    "ConversationAttachmentReference",
    "ConversationChannelBackend",
    "ConversationChoiceOption",
    "ConversationDeliveryReceipt",
    "ConversationEventHandler",
    "ConversationEventKind",
    "ConversationSingleChoice",
    "InboundConversationEvent",
    "OutboundConversationMessage",
]
