# © Artur Czarnecki. All rights reserved.

"""Codec for execution-bound catalog tool suspended payloads."""

from __future__ import annotations

import json

from intergrax.contracts.execution.suspended_operation.codec import (
    SerializedSuspendedOperationEnvelope,
    SuspendedOperationKind,
    SuspendedOperationPayload,
)
from intergrax.contracts.execution.suspended_operation.payload_catalog import (
    EXECUTION_BOUND_CATALOG_TOOL_PAYLOAD_V1,
    ExecutionBoundCatalogToolOperationPayload,
)
from intergrax.runtime.execution.suspended_operation.payload_digest import (
    canonical_json_from_model,
)


class ExecutionBoundCatalogToolPayloadCodec:
    @property
    def operation_kind(self) -> SuspendedOperationKind:
        return SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL

    @property
    def payload_schema_version(self) -> str:
        return EXECUTION_BOUND_CATALOG_TOOL_PAYLOAD_V1

    def encode(
        self,
        payload: SuspendedOperationPayload,
    ) -> SerializedSuspendedOperationEnvelope:
        if type(payload) is not ExecutionBoundCatalogToolOperationPayload:
            raise TypeError("payload must be ExecutionBoundCatalogToolOperationPayload")
        canonical_json = canonical_json_from_model(payload)
        return SerializedSuspendedOperationEnvelope(
            operation_kind=self.operation_kind,
            payload_schema_version=self.payload_schema_version,
            canonical_json=canonical_json,
        )

    def decode(
        self,
        envelope: SerializedSuspendedOperationEnvelope,
    ) -> ExecutionBoundCatalogToolOperationPayload:
        if envelope.operation_kind is not self.operation_kind:
            raise ValueError("operation_kind mismatch")
        if envelope.payload_schema_version != self.payload_schema_version:
            raise ValueError("payload_schema_version mismatch")
        raw = json.loads(envelope.canonical_json)
        return ExecutionBoundCatalogToolOperationPayload.model_validate(raw)


__all__ = ["ExecutionBoundCatalogToolPayloadCodec"]
