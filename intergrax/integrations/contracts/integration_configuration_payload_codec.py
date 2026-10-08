# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed durable codecs for provider-owned IntegrationConfigurationPayload (TRACE-X-P5-R2-P2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.existing_capability_configuration import (
    IntegrationConfigurationPayload,
)
from intergrax.knowledge.contracts.validation import JsonValue


@runtime_checkable
class IntegrationConfigurationPayloadCodec(Protocol):
    """Encode/decode one concrete provider configuration payload type."""

    def configuration_type(self) -> str:
        """Stable discriminator matching ``IntegrationConfigurationPayload.configuration_type``."""
        ...

    def encode(self, payload: IntegrationConfigurationPayload) -> JsonValue:
        """Serialize one typed payload for durable storage."""
        ...

    def decode(self, payload: JsonValue) -> IntegrationConfigurationPayload:
        """Restore one typed payload from durable storage."""
        ...


@dataclass(frozen=True, slots=True)
class IntegrationConfigurationPayloadCodecRegistry:
    """Immutable registry keyed by ``configuration_type``."""

    _codecs: dict[str, IntegrationConfigurationPayloadCodec]

    def encode(self, payload: IntegrationConfigurationPayload) -> tuple[str, JsonValue]:
        config_type = payload.configuration_type
        codec = self._codecs.get(config_type)
        if codec is None:
            raise ValueError(f"unknown integration configuration payload type: {config_type!r}")
        return config_type, codec.encode(payload)

    def decode(self, *, configuration_type: str, payload: JsonValue) -> IntegrationConfigurationPayload:
        codec = self._codecs.get(configuration_type)
        if codec is None:
            raise ValueError(f"unknown integration configuration payload type: {configuration_type!r}")
        return codec.decode(payload)


def integration_configuration_payload_codec_registry(
    *,
    codecs: tuple[IntegrationConfigurationPayloadCodec, ...],
) -> IntegrationConfigurationPayloadCodecRegistry:
    by_type: dict[str, IntegrationConfigurationPayloadCodec] = {}
    for codec in codecs:
        name = codec.configuration_type()
        if name in by_type:
            raise ValueError(f"duplicate integration configuration payload codec: {name!r}")
        by_type[name] = codec
    return IntegrationConfigurationPayloadCodecRegistry(_codecs=by_type)


__all__ = [
    "IntegrationConfigurationPayloadCodec",
    "IntegrationConfigurationPayloadCodecRegistry",
    "integration_configuration_payload_codec_registry",
]
