# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed wire codec for durable Decision persistence adapters."""

from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.decision_checkpoint import DecisionCheckpointState
from intergrax.contracts.decision_finalization import DecisionFinalizeGuardState
from intergrax.contracts.decision_lifecycle import DecisionLifecycleState
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.contracts.decision_resolution import AuthoritativeResolutionRecord
from intergrax.contracts.decision_revision import DecisionRevisionCheckpointState

T = TypeVar("T")

_CHECKPOINT_SCHEMA_V1 = 1
_CHECKPOINT_SCHEMA_V2 = 2


@dataclass(frozen=True, slots=True)
class _CheckpointWireEnvelope(Generic[T]):
    schema_version: int
    lifecycle: DecisionLifecycleState
    finalization: DecisionFinalizeGuardState[T]
    revision: DecisionRevisionCheckpointState | None = None


def encode_checkpoint_blob(checkpoint: DecisionCheckpointState[T]) -> str:
    """Serialize one checkpoint snapshot for durable storage."""
    envelope = _CheckpointWireEnvelope(
        schema_version=_CHECKPOINT_SCHEMA_V2,
        lifecycle=checkpoint.lifecycle,
        finalization=checkpoint.finalization,
        revision=checkpoint.revision,
    )
    return base64.b64encode(
        pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL),
    ).decode("ascii")


def decode_checkpoint_blob(blob: str) -> DecisionCheckpointState[object]:
    """Deserialize one checkpoint snapshot from durable storage."""
    envelope = pickle.loads(base64.b64decode(blob.encode("ascii")))
    if type(envelope) is not _CheckpointWireEnvelope:
        raise ValueError("checkpoint blob must decode to checkpoint wire envelope")
    if envelope.schema_version == _CHECKPOINT_SCHEMA_V1:
        return DecisionCheckpointState(
            lifecycle=envelope.lifecycle,
            finalization=envelope.finalization,
            revision=None,
        )
    if envelope.schema_version == _CHECKPOINT_SCHEMA_V2:
        return DecisionCheckpointState(
            lifecycle=envelope.lifecycle,
            finalization=envelope.finalization,
            revision=envelope.revision,
        )
    raise ValueError(f"unsupported checkpoint schema version: {envelope.schema_version}")


def encode_outcome_blob(
    outcome: AuthoritativeAcceptedDecision[T] | AuthoritativeResolutionRecord,
) -> str:
    """Serialize one authoritative outcome for durable storage."""
    return base64.b64encode(
        pickle.dumps(outcome, protocol=pickle.HIGHEST_PROTOCOL),
    ).decode("ascii")


def decode_outcome_blob(
    blob: str,
) -> AuthoritativeAcceptedDecision[object] | AuthoritativeResolutionRecord:
    """Deserialize one authoritative outcome from durable storage."""
    outcome = pickle.loads(base64.b64decode(blob.encode("ascii")))
    if type(outcome) is AuthoritativeAcceptedDecision:
        return outcome
    if type(outcome) is AuthoritativeResolutionRecord:
        return outcome
    raise ValueError("outcome blob must decode to authoritative decision record")
