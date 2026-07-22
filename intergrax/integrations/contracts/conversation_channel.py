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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import HealthStatus

MAX_CONVERSATION_CHOICE_OPTIONS = 25
MAX_OUTBOUND_TEXT_LENGTH = 4000
MAX_CHOICE_LABEL_LENGTH = 150
MAX_CHOICE_VALUE_LENGTH = 150
MAX_ACTION_ID_LENGTH = 150
MAX_SINGLE_CHOICE_COMPONENTS = 1

ConversationEventHandler = Callable[["InboundConversationEvent"], Awaitable[None]]


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
            if self.text is None or not self.text.strip():
                raise ValueError("MESSAGE events require non-blank text")
            if self.action is not None:
                raise ValueError("MESSAGE events must not include action")
        elif self.kind is ConversationEventKind.ACTION:
            if self.action is None:
                raise ValueError("ACTION events require action selection")
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
    "MAX_CHOICE_LABEL_LENGTH",
    "MAX_CHOICE_VALUE_LENGTH",
    "MAX_CONVERSATION_CHOICE_OPTIONS",
    "MAX_OUTBOUND_TEXT_LENGTH",
    "MAX_SINGLE_CHOICE_COMPONENTS",
    "ConversationActionSelection",
    "ConversationActor",
    "ConversationAddress",
    "ConversationChannelBackend",
    "ConversationChoiceOption",
    "ConversationDeliveryReceipt",
    "ConversationEventHandler",
    "ConversationEventKind",
    "ConversationSingleChoice",
    "InboundConversationEvent",
    "OutboundConversationMessage",
]
