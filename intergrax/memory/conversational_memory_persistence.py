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
