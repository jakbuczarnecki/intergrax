# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit typed payload codec seam for Decision artifact content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from intergrax.contracts.decision_record import DecisionArtifactKind, validate_decision_artifact_kind
from intergrax.knowledge.contracts.validation import JsonValue, validate_json_value
from intergrax.runtime.execution.decision_persistence_codec_errors import (
    DecisionPersistenceUnknownPayloadCodecError,
)

T = TypeVar("T")


class DecisionArtifactPayloadCodec(Protocol[T]):
    """Encode/decode one explicit artifact payload type for durable persistence."""

    def encode(self, payload: T) -> JsonValue:
        """Serialize one typed artifact payload to JSON-safe wire values."""
        ...

    def decode(self, payload: JsonValue) -> T:
        """Reconstruct one typed artifact payload from JSON-safe wire values."""
        ...


@dataclass(frozen=True, slots=True)
class DecisionArtifactPayloadCodecRegistry:
    """Immutable registry of explicit artifact payload codecs keyed by kind."""

    _codecs: dict[str, DecisionArtifactPayloadCodec[object]]

    def encode_content(
        self,
        *,
        kind: DecisionArtifactKind,
        content: object,
    ) -> JsonValue:
        codec = self._codecs.get(str(kind))
        if codec is None:
            raise DecisionPersistenceUnknownPayloadCodecError(
                f"no payload codec registered for artifact kind {str(kind)!r}",
            )
        return codec.encode(content)

    def decode_content(
        self,
        *,
        kind: DecisionArtifactKind,
        wire: JsonValue,
    ) -> object:
        codec = self._codecs.get(str(kind))
        if codec is None:
            raise DecisionPersistenceUnknownPayloadCodecError(
                f"no payload codec registered for artifact kind {str(kind)!r}",
            )
        return codec.decode(wire)


def decision_artifact_payload_codec_registry(
    *,
    codecs: dict[DecisionArtifactKind, DecisionArtifactPayloadCodec[object]],
) -> DecisionArtifactPayloadCodecRegistry:
    """Build one registry from explicit artifact-kind to codec mappings."""
    normalized: dict[str, DecisionArtifactPayloadCodec[object]] = {}
    for kind, codec in codecs.items():
        validated_kind = validate_decision_artifact_kind(kind)
        normalized[str(validated_kind)] = codec
    return DecisionArtifactPayloadCodecRegistry(_codecs=normalized)


@dataclass(frozen=True, slots=True)
class JsonObjectDecisionArtifactPayloadCodec:
    """Built-in codec for JSON-object artifact payloads with explicit field schema."""

    _fields: tuple[str, ...]

    def encode(self, payload: object) -> JsonValue:
        if type(payload) is not dict:
            raise TypeError("JsonObjectDecisionArtifactPayloadCodec expects dict payload")
        encoded: dict[str, JsonValue] = {}
        for field_name in self._fields:
            if field_name not in payload:
                raise ValueError(f"missing required artifact payload field {field_name!r}")
            encoded[field_name] = validate_json_value(
                payload[field_name],
                field_name=field_name,
            )
        return encoded

    def decode(self, payload: JsonValue) -> dict[str, JsonValue]:
        if type(payload) is not dict:
            raise TypeError("artifact payload wire value must be a JSON object")
        decoded: dict[str, JsonValue] = {}
        for field_name in self._fields:
            if field_name not in payload:
                raise ValueError(f"missing required artifact payload field {field_name!r}")
            decoded[field_name] = validate_json_value(
                payload[field_name],
                field_name=field_name,
            )
        return decoded

