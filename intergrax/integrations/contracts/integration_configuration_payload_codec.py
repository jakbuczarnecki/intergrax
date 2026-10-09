# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed durable codecs for provider-owned IntegrationConfigurationPayload (TRACE-X-P5-R2-P2)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
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
    """Immutable registry keyed by ``configuration_type``.

    Maps configuration type discriminators to serialization codecs only.
    Does not activate providers, authorize configuration, or imply effective integration.
    """

    _codecs: Mapping[str, IntegrationConfigurationPayloadCodec]

    def encode(self, payload: IntegrationConfigurationPayload) -> tuple[str, JsonValue]:
        config_type = payload.configuration_type
        codec = self._codecs.get(config_type)
        if codec is None:
            raise ValueError(f"unknown integration configuration payload type: {config_type!r}")
        registered_type = codec.configuration_type()
        if registered_type != config_type:
            raise ValueError(
                "integration configuration payload codec identity mismatch on encode: "
                f"registered {registered_type!r} vs payload {config_type!r}",
            )
        return config_type, codec.encode(payload)

    def decode(self, *, configuration_type: str, payload: JsonValue) -> IntegrationConfigurationPayload:
        codec = self._codecs.get(configuration_type)
        if codec is None:
            raise ValueError(f"unknown integration configuration payload type: {configuration_type!r}")
        decoded = codec.decode(payload)
        if decoded.configuration_type != configuration_type:
            raise ValueError(
                "integration configuration payload codec identity mismatch on decode: "
                f"requested {configuration_type!r} vs decoded {decoded.configuration_type!r}",
            )
        return decoded


def _validate_codec_configuration_type(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("integration configuration payload codec type must be non-empty")
    if name != name.strip():
        raise ValueError(
            f"integration configuration payload codec type must not have leading/trailing whitespace: {name!r}",
        )
    return name


def integration_configuration_payload_codec_registry(
    *,
    codecs: tuple[IntegrationConfigurationPayloadCodec, ...],
) -> IntegrationConfigurationPayloadCodecRegistry:
    by_type: dict[str, IntegrationConfigurationPayloadCodec] = {}
    for codec in codecs:
        if not isinstance(codec, IntegrationConfigurationPayloadCodec):
            raise TypeError("integration configuration payload codec must implement IntegrationConfigurationPayloadCodec")
        name = _validate_codec_configuration_type(codec.configuration_type())
        if name in by_type:
            raise ValueError(f"duplicate integration configuration payload codec: {name!r}")
        by_type[name] = codec
    return IntegrationConfigurationPayloadCodecRegistry(
        _codecs=MappingProxyType(dict(by_type)),
    )


__all__ = [
    "IntegrationConfigurationPayloadCodec",
    "IntegrationConfigurationPayloadCodecRegistry",
    "integration_configuration_payload_codec_registry",
]
