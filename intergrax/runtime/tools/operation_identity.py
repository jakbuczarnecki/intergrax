# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pydantic import BaseModel

from intergrax.contracts.idempotency_store import InvocationOperationIdentity
from intergrax.runtime.attestation.canonical_json import stable_payload_hash


def compute_invocation_operation_identity(
    tool_id: str,
    input_model: BaseModel,
) -> InvocationOperationIdentity:
    """Deterministic logical operation identity for idempotency ledger binding."""
    fingerprint = stable_payload_hash(input_model.model_dump(mode="json"))
    return InvocationOperationIdentity(
        tool_id=tool_id,
        operation_fingerprint=fingerprint,
    )
