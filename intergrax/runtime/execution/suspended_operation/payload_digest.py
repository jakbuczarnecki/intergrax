# © Artur Czarnecki. All rights reserved.

"""Canonical digest for suspended operation serialized envelopes."""

from __future__ import annotations

import json

from intergrax.contracts.canonical_payload_hash import stable_payload_hash
from intergrax.contracts.execution.suspended_operation.codec import (
    SerializedSuspendedOperationEnvelope,
)


def digest_suspended_operation_envelope(
    envelope: SerializedSuspendedOperationEnvelope,
) -> str:
    payload = {
        "operation_kind": envelope.operation_kind.value,
        "payload_schema_version": envelope.payload_schema_version,
        "canonical_json": envelope.canonical_json,
    }
    return stable_payload_hash(payload)


def canonical_json_from_model(model) -> str:
    dumped = model.model_dump(mode="json")
    return json.dumps(dumped, sort_keys=True, separators=(",", ":"))


__all__ = ["canonical_json_from_model", "digest_suspended_operation_envelope"]
