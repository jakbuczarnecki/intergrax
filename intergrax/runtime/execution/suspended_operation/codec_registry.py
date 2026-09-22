# © Artur Czarnecki. All rights reserved.

"""Default suspended-operation codec registry."""

from __future__ import annotations

from intergrax.contracts.execution.suspended_operation.codec import (
    SerializedSuspendedOperationEnvelope,
    SuspendedOperationCodecRegistry,
    SuspendedOperationKind,
    SuspendedOperationPayload,
    SuspendedOperationPayloadCodec,
)
from intergrax.runtime.execution.suspended_operation.catalog_tool_codec import (
    ExecutionBoundCatalogToolPayloadCodec,
)


class DefaultSuspendedOperationCodecRegistry(SuspendedOperationCodecRegistry):
    def __init__(
        self,
        codecs: tuple[SuspendedOperationPayloadCodec, ...] | None = None,
    ) -> None:
        resolved = codecs or (ExecutionBoundCatalogToolPayloadCodec(),)
        self._codecs: dict[tuple[SuspendedOperationKind, str], SuspendedOperationPayloadCodec] = {}
        for codec in resolved:
            key = (codec.operation_kind, codec.payload_schema_version)
            if key in self._codecs:
                raise ValueError(f"duplicate suspended operation codec: {key}")
            self._codecs[key] = codec

    def resolve(
        self,
        operation_kind: SuspendedOperationKind,
        payload_schema_version: str,
    ) -> SuspendedOperationPayloadCodec:
        codec = self._codecs.get((operation_kind, payload_schema_version))
        if codec is None:
            raise LookupError(
                f"unknown suspended operation codec: {operation_kind}/{payload_schema_version}",
            )
        return codec

    def encode(
        self,
        payload: SuspendedOperationPayload,
        *,
        operation_kind: SuspendedOperationKind,
        payload_schema_version: str,
    ) -> SerializedSuspendedOperationEnvelope:
        return self.resolve(operation_kind, payload_schema_version).encode(payload)


__all__ = ["DefaultSuspendedOperationCodecRegistry"]
