# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed codecs for decision event payloads on durable storage (W3-C)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.decision_event_append import DecisionEventPayload
from intergrax.knowledge.contracts.validation import JsonValue


class DecisionEventPayloadCodec(Protocol):
    """Encode and decode one concrete :class:`DecisionEventPayload` type."""

    def encode(self, payload: DecisionEventPayload) -> JsonValue:
        """Serialize one typed payload for durable storage."""
        ...

    def decode(self, payload: JsonValue) -> DecisionEventPayload:
        """Restore one typed payload from durable storage."""
        ...

    def payload_type_name(self) -> str:
        """Stable discriminator stored alongside the JSON blob."""
        ...


@dataclass(frozen=True, slots=True)
class DecisionEventPayloadCodecRegistry:
    """Immutable registry of event payload codecs keyed by ``payload_type_name``."""

    _codecs: dict[str, DecisionEventPayloadCodec]

    def encode(self, payload: DecisionEventPayload) -> tuple[str, JsonValue]:
        for name, codec in self._codecs.items():
            try:
                return name, codec.encode(payload)
            except TypeError:
                continue
        raise TypeError(
            f"no decision event payload codec for {type(payload).__name__}",
        )

    def decode(self, *, payload_type: str, payload: JsonValue) -> DecisionEventPayload:
        codec = self._codecs.get(payload_type)
        if codec is None:
            raise ValueError(f"unknown decision event payload type: {payload_type!r}")
        return codec.decode(payload)


def decision_event_payload_codec_registry(
    *,
    codecs: tuple[DecisionEventPayloadCodec, ...],
) -> DecisionEventPayloadCodecRegistry:
    """Build one registry from explicit payload codecs."""
    by_name: dict[str, DecisionEventPayloadCodec] = {}
    for codec in codecs:
        name = codec.payload_type_name()
        if name in by_name:
            raise ValueError(f"duplicate decision event payload codec: {name!r}")
        by_name[name] = codec
    return DecisionEventPayloadCodecRegistry(_codecs=by_name)
