# © Artur Czarnecki. All rights reserved.

"""Durable persisted envelope for conversational memory (COMPAT-X-R2-R1)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_CONVERSATIONAL_MEMORY_V1: Final = "conversational_memory.v1"


class ConversationalMemoryMessageWire(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1)
    content: str


class ConversationalMemoryPersistedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["conversational_memory.v1"] = SCHEMA_CONVERSATIONAL_MEMORY_V1
    max_messages: int = Field(ge=1)
    messages: tuple[ConversationalMemoryMessageWire, ...] = ()


class LegacyConversationalMemoryMessageWire(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1)
    content: str


class LegacyConversationalMemoryPayload(BaseModel):
    """Historical durable JSON without schema identity (evidence @ e5ae6a3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_messages: int = Field(ge=1)
    messages: tuple[LegacyConversationalMemoryMessageWire, ...] = ()


LEGACY_UNVERSIONED_PRE_V1: Final = "LEGACY_UNVERSIONED_PRE_V1"


def parse_conversational_memory_persisted(
    raw: dict[str, object],
) -> ConversationalMemoryPersistedPayload:
    schema_version = raw.get("schema_version")
    if schema_version == SCHEMA_CONVERSATIONAL_MEMORY_V1:
        return ConversationalMemoryPersistedPayload.model_validate(raw)
    if "schema_version" not in raw:
        legacy = LegacyConversationalMemoryPayload.model_validate(raw)
        return ConversationalMemoryPersistedPayload(
            schema_version=SCHEMA_CONVERSATIONAL_MEMORY_V1,
            max_messages=legacy.max_messages,
            messages=tuple(
                ConversationalMemoryMessageWire(role=item.role, content=item.content)
                for item in legacy.messages
            ),
        )
    raise ValueError(
        f"unsupported conversational memory schema version {schema_version!r}; "
        f"expected {SCHEMA_CONVERSATIONAL_MEMORY_V1!r} or {LEGACY_UNVERSIONED_PRE_V1}"
    )
