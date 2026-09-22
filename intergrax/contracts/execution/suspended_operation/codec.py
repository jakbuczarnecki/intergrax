# © Artur Czarnecki. All rights reserved.

"""Typed suspended-operation payload codec contracts (UCA-6C-R6)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class SuspendedOperationKind(StrEnum):
    EXECUTION_BOUND_CATALOG_TOOL = "execution_bound_catalog_tool"


class SerializedSuspendedOperationEnvelope(BaseModel):
    """Opaque serialized payload — store does not interpret semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_kind: SuspendedOperationKind
    payload_schema_version: str = Field(min_length=1)
    canonical_json: str = Field(min_length=1)


class SuspendedOperationPayload(BaseModel):
    """Marker base for typed payload models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@runtime_checkable
class SuspendedOperationPayloadCodec(Protocol):
    """Encode/decode one (operation_kind, payload_schema_version) pair."""

    @property
    def operation_kind(self) -> SuspendedOperationKind: ...

    @property
    def payload_schema_version(self) -> str: ...

    def encode(
        self, payload: SuspendedOperationPayload
    ) -> SerializedSuspendedOperationEnvelope:
        """Serialize typed payload to canonical envelope."""
        ...

    def decode(
        self,
        envelope: SerializedSuspendedOperationEnvelope,
    ) -> SuspendedOperationPayload:
        """Decode envelope to typed payload; fail closed on mismatch."""
        ...


class SuspendedOperationCodecRegistry(ABC):
    """Resolve codecs by stable (operation_kind, payload_schema_version)."""

    @abstractmethod
    def resolve(
        self,
        operation_kind: SuspendedOperationKind,
        payload_schema_version: str,
    ) -> SuspendedOperationPayloadCodec:
        """Return codec or raise LookupError when unknown."""
        ...


__all__ = [
    "SerializedSuspendedOperationEnvelope",
    "SuspendedOperationCodecRegistry",
    "SuspendedOperationKind",
    "SuspendedOperationPayload",
    "SuspendedOperationPayloadCodec",
]
