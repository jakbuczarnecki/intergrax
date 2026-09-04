# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed wire codec for durable Decision persistence adapters."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.decision_checkpoint import DecisionCheckpointState
from intergrax.contracts.decision_finalization import (
    DecisionFinalizationKey,
    DecisionFinalizeGuardState,
    decision_finalization_key,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    validate_decision_id,
    validate_decision_tenant_id,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    DecisionLineageRef,
    DecisionProposalRef,
    DecisionVersionLineage,
    decision_lineage_ref,
    validate_decision_artifact_kind,
    validate_decision_branch_id,
)
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.contracts.decision_revision import DecisionRevisionCheckpointState
from intergrax.contracts.execution_identity import (
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.knowledge.contracts.validation import JsonValue
from intergrax.runtime.execution.decision_artifact_payload_codec import (
    DecisionArtifactPayloadCodecRegistry,
)
from intergrax.runtime.execution.decision_persistence_codec_errors import (
    DecisionPersistenceCodecError,
    DecisionPersistenceLegacyPickleUnsupportedError,
    DecisionPersistenceRecordTypeError,
    DecisionPersistenceUnsupportedSchemaError,
)

T = TypeVar("T")

_CHECKPOINT_SCHEMA_V1 = 1
_CHECKPOINT_SCHEMA_V2 = 2
_OUTCOME_SCHEMA_V1 = 1

_RECORD_TYPE_CHECKPOINT = "decision_checkpoint"
_RECORD_TYPE_ACCEPTED = "authoritative_accepted_decision"
_RECORD_TYPE_RESOLUTION = "authoritative_resolution"


@dataclass(frozen=True, slots=True)
class _CheckpointWireEnvelope(Generic[T]):
    schema_version: int
    lifecycle: DecisionLifecycleState
    finalization: DecisionFinalizeGuardState[T]
    revision: DecisionRevisionCheckpointState | None = None


def _canonical_json_text(payload: dict[str, JsonValue]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _require_mapping(value: object, label: str) -> dict[str, JsonValue]:
    if type(value) is not dict:
        raise DecisionPersistenceCodecError(f"{label} must be a JSON object")
    return value


def _require_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise DecisionPersistenceCodecError(f"{label} must be str")
    return value


def _require_int(value: object, label: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise DecisionPersistenceCodecError(f"{label} must be int")
    return value


def _require_list(value: object, label: str) -> list[JsonValue]:
    if type(value) is not list:
        raise DecisionPersistenceCodecError(f"{label} must be a JSON array")
    return value


def _decode_scope_wire(wire: dict[str, JsonValue]) -> DecisionScope:
    return DecisionScope(
        namespace=_require_str(wire.get("namespace"), "scope.namespace"),
        subject=_require_str(wire.get("subject"), "scope.subject"),
    )


def _encode_scope(scope: DecisionScope) -> dict[str, JsonValue]:
    return {
        "namespace": scope.namespace,
        "subject": scope.subject,
    }


def _decode_execution_wire(wire: dict[str, JsonValue]) -> DecisionExecutionLineage:
    execution_id_wire = wire.get("execution_id")
    execution_id = None
    if execution_id_wire is not None:
        execution_id = validate_execution_id(_require_str(execution_id_wire, "execution.execution_id"))
    return DecisionExecutionLineage(
        task_id=validate_task_id(_require_str(wire.get("task_id"), "execution.task_id")),
        run_id=validate_run_id(_require_str(wire.get("run_id"), "execution.run_id")),
        attempt_id=validate_attempt_id(
            _require_str(wire.get("attempt_id"), "execution.attempt_id"),
        ),
        execution_id=execution_id,
    )


def _encode_execution(execution: DecisionExecutionLineage) -> dict[str, JsonValue]:
    wire: dict[str, JsonValue] = {
        "task_id": str(execution.task_id),
        "run_id": str(execution.run_id),
        "attempt_id": str(execution.attempt_id),
    }
    if execution.execution_id is not None:
        wire["execution_id"] = str(execution.execution_id)
    return wire


def _decode_identity_wire(wire: dict[str, JsonValue]) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=validate_decision_id(_require_str(wire.get("decision_id"), "identity.decision_id")),
        version=DecisionVersion(_require_int(wire.get("decision_version"), "identity.decision_version")),
        scope=_decode_scope_wire(_require_mapping(wire.get("scope"), "identity.scope")),
        tenant_id=validate_decision_tenant_id(_require_str(wire.get("tenant_id"), "identity.tenant_id")),
        execution=_decode_execution_wire(
            _require_mapping(wire.get("execution"), "identity.execution"),
        ),
    )


def _encode_identity(identity: DecisionIdentity) -> dict[str, JsonValue]:
    return {
        "decision_id": str(identity.decision_id),
        "decision_version": identity.version.value,
        "tenant_id": identity.tenant_id,
        "scope": _encode_scope(identity.scope),
        "execution": _encode_execution(identity.execution),
    }


def _decode_finalization_key_wire(wire: dict[str, JsonValue]) -> DecisionFinalizationKey:
    return DecisionFinalizationKey(
        decision_id=validate_decision_id(_require_str(wire.get("decision_id"), "finalization_key.decision_id")),
        scope=_decode_scope_wire(_require_mapping(wire.get("scope"), "finalization_key.scope")),
        tenant_id=validate_decision_tenant_id(
            _require_str(wire.get("tenant_id"), "finalization_key.tenant_id"),
        ),
    )


def _encode_finalization_key(key: DecisionFinalizationKey) -> dict[str, JsonValue]:
    return {
        "decision_id": str(key.decision_id),
        "tenant_id": key.tenant_id,
        "scope": _encode_scope(key.scope),
    }


def _decode_lifecycle_stage(value: object) -> DecisionLifecycleStage:
    stage_value = _require_str(value, "lifecycle.stage")
    try:
        return DecisionLifecycleStage(stage_value)
    except ValueError as exc:
        raise DecisionPersistenceCodecError(
            f"unsupported lifecycle stage: {stage_value!r}",
        ) from exc


def _decode_lifecycle_wire(wire: dict[str, JsonValue]) -> DecisionLifecycleState:
    return DecisionLifecycleState(
        identity=_decode_identity_wire(_require_mapping(wire.get("identity"), "lifecycle.identity")),
        stage=_decode_lifecycle_stage(wire.get("stage")),
        transition_index=_require_int(wire.get("transition_index"), "lifecycle.transition_index"),
    )


def _encode_lifecycle(lifecycle: DecisionLifecycleState) -> dict[str, JsonValue]:
    return {
        "identity": _encode_identity(lifecycle.identity),
        "stage": lifecycle.stage.value,
        "transition_index": lifecycle.transition_index,
    }


def _decode_lineage_ref_wire(wire: dict[str, JsonValue]) -> DecisionLineageRef:
    return decision_lineage_ref(
        DecisionVersion(_require_int(wire.get("version"), "lineage_ref.version")),
        validate_decision_branch_id(_require_str(wire.get("branch_id"), "lineage_ref.branch_id")),
    )


def _encode_lineage_ref(ref: DecisionLineageRef) -> dict[str, JsonValue]:
    return {
        "version": ref.version.value,
        "branch_id": str(ref.branch_id),
    }


def _decode_lineage_wire(wire: dict[str, JsonValue]) -> DecisionVersionLineage:
    parents_wire = _require_list(wire.get("parents"), "lineage.parents")
    parents = tuple(
        _decode_lineage_ref_wire(_require_mapping(parent, "lineage.parent"))
        for parent in parents_wire
    )
    return DecisionVersionLineage(
        current=_decode_lineage_ref_wire(_require_mapping(wire.get("current"), "lineage.current")),
        parents=parents,
    )


def _encode_lineage(lineage: DecisionVersionLineage) -> dict[str, JsonValue]:
    return {
        "current": _encode_lineage_ref(lineage.current),
        "parents": [_encode_lineage_ref(parent) for parent in lineage.parents],
    }


def _decode_proposal_ref_wire(wire: dict[str, JsonValue]) -> DecisionProposalRef:
    return DecisionProposalRef(
        identity=_decode_identity_wire(_require_mapping(wire.get("identity"), "proposal_ref.identity")),
        lineage_ref=_decode_lineage_ref_wire(
            _require_mapping(wire.get("lineage_ref"), "proposal_ref.lineage_ref"),
        ),
    )


def _encode_proposal_ref(proposal_ref: DecisionProposalRef) -> dict[str, JsonValue]:
    return {
        "identity": _encode_identity(proposal_ref.identity),
        "lineage_ref": _encode_lineage_ref(proposal_ref.lineage_ref),
    }


def _decode_revision_wire(wire: dict[str, JsonValue]) -> DecisionRevisionCheckpointState:
    return DecisionRevisionCheckpointState(
        proposal_ref=_decode_proposal_ref_wire(
            _require_mapping(wire.get("proposal_ref"), "revision.proposal_ref"),
        ),
        revision_count=_require_int(wire.get("revision_count"), "revision.revision_count"),
        max_revisions=_require_int(wire.get("max_revisions"), "revision.max_revisions"),
    )


def _encode_revision(revision: DecisionRevisionCheckpointState) -> dict[str, JsonValue]:
    return {
        "proposal_ref": _encode_proposal_ref(revision.proposal_ref),
        "revision_count": revision.revision_count,
        "max_revisions": revision.max_revisions,
    }


def _decode_resolution(value: object) -> DecisionResolution:
    resolution_value = _require_str(value, "resolution")
    try:
        return DecisionResolution(resolution_value)
    except ValueError as exc:
        raise DecisionPersistenceCodecError(
            f"unsupported decision resolution: {resolution_value!r}",
        ) from exc


def _decode_accepted_wire(
    wire: dict[str, JsonValue],
    *,
    payload_codecs: DecisionArtifactPayloadCodecRegistry,
) -> AuthoritativeAcceptedDecision[object]:
    identity = _decode_identity_wire(_require_mapping(wire.get("identity"), "accepted.identity"))
    artifact_wire = _require_mapping(wire.get("artifact"), "accepted.artifact")
    kind = validate_decision_artifact_kind(_require_str(artifact_wire.get("kind"), "artifact.kind"))
    content_wire = artifact_wire.get("content")
    if content_wire is None:
        raise DecisionPersistenceCodecError("accepted.artifact.content is required")
    content = payload_codecs.decode_content(kind=kind, wire=content_wire)
    artifact = DecisionArtifact(kind=kind, content=content)
    lineage = _decode_lineage_wire(_require_mapping(wire.get("lineage"), "accepted.lineage"))
    return AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=artifact,
        lineage=lineage,
    )


def _encode_accepted(
    accepted: AuthoritativeAcceptedDecision[T],
    *,
    payload_codecs: DecisionArtifactPayloadCodecRegistry,
) -> dict[str, JsonValue]:
    kind = accepted.artifact.kind
    return {
        "identity": _encode_identity(accepted.identity),
        "artifact": {
            "kind": str(kind),
            "content": payload_codecs.encode_content(
                kind=kind,
                content=accepted.artifact.content,
            ),
        },
        "lineage": _encode_lineage(accepted.lineage),
    }


def _decode_resolution_record_wire(wire: dict[str, JsonValue]) -> AuthoritativeResolutionRecord:
    return AuthoritativeResolutionRecord(
        identity=_decode_identity_wire(_require_mapping(wire.get("identity"), "resolution.identity")),
        resolution=_decode_resolution(wire.get("resolution")),
    )


def _encode_resolution_record(record: AuthoritativeResolutionRecord) -> dict[str, JsonValue]:
    return {
        "identity": _encode_identity(record.identity),
        "resolution": record.resolution.value,
    }


def _decode_authoritative_outcome_wire(
    wire: dict[str, JsonValue] | None,
    *,
    payload_codecs: DecisionArtifactPayloadCodecRegistry,
) -> AuthoritativeAcceptedDecision[object] | AuthoritativeResolutionRecord | None:
    if wire is None:
        return None
    record_type = _require_str(wire.get("record_type"), "authoritative_outcome.record_type")
    schema_version = _require_int(wire.get("schema_version"), "authoritative_outcome.schema_version")
    if schema_version != _OUTCOME_SCHEMA_V1:
        raise DecisionPersistenceUnsupportedSchemaError(
            f"unsupported authoritative outcome schema version: {schema_version}",
        )
    payload = _require_mapping(wire.get("payload"), "authoritative_outcome.payload")
    if record_type == _RECORD_TYPE_ACCEPTED:
        return _decode_accepted_wire(payload, payload_codecs=payload_codecs)
    if record_type == _RECORD_TYPE_RESOLUTION:
        return _decode_resolution_record_wire(payload)
    raise DecisionPersistenceRecordTypeError(
        f"unsupported authoritative outcome record_type: {record_type!r}",
    )


def _encode_authoritative_outcome(
    outcome: AuthoritativeAcceptedDecision[T] | AuthoritativeResolutionRecord | None,
    *,
    payload_codecs: DecisionArtifactPayloadCodecRegistry,
) -> JsonValue | None:
    if outcome is None:
        return None
    if type(outcome) is AuthoritativeAcceptedDecision:
        return {
            "schema_version": _OUTCOME_SCHEMA_V1,
            "record_type": _RECORD_TYPE_ACCEPTED,
            "payload": _encode_accepted(outcome, payload_codecs=payload_codecs),
        }
    if type(outcome) is AuthoritativeResolutionRecord:
        return {
            "schema_version": _OUTCOME_SCHEMA_V1,
            "record_type": _RECORD_TYPE_RESOLUTION,
            "payload": _encode_resolution_record(outcome),
        }
    raise DecisionPersistenceCodecError(
        "authoritative outcome must be AuthoritativeAcceptedDecision "
        "or AuthoritativeResolutionRecord",
    )


def _decode_finalization_wire(
    wire: dict[str, JsonValue],
    *,
    payload_codecs: DecisionArtifactPayloadCodecRegistry,
) -> DecisionFinalizeGuardState[object]:
    key = _decode_finalization_key_wire(_require_mapping(wire.get("key"), "finalization.key"))
    outcome_wire = wire.get("authoritative_outcome")
    outcome = None
    if outcome_wire is not None:
        outcome = _decode_authoritative_outcome_wire(
            _require_mapping(outcome_wire, "finalization.authoritative_outcome"),
            payload_codecs=payload_codecs,
        )
    return DecisionFinalizeGuardState(key=key, authoritative_outcome=outcome)


def _encode_finalization(
    finalization: DecisionFinalizeGuardState[T],
    *,
    payload_codecs: DecisionArtifactPayloadCodecRegistry,
) -> dict[str, JsonValue]:
    encoded_outcome = _encode_authoritative_outcome(
        finalization.authoritative_outcome,
        payload_codecs=payload_codecs,
    )
    wire: dict[str, JsonValue] = {
        "key": _encode_finalization_key(finalization.key),
    }
    if encoded_outcome is not None:
        wire["authoritative_outcome"] = encoded_outcome
    return wire


def _reject_legacy_executable_blob(blob: str) -> None:
    stripped = blob.strip()
    if stripped.startswith("{"):
        return
    try:
        raw = base64.b64decode(blob.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeError) as exc:
        raise DecisionPersistenceCodecError("invalid durable decision blob") from exc
    if raw[:1] == b"\x80" or raw[:2] == b"(" or raw[:4] == b"\x93\x00":
        raise DecisionPersistenceLegacyPickleUnsupportedError(
            "legacy executable durable decision blobs are unsupported at runtime",
        )
    raise DecisionPersistenceCodecError("invalid durable decision blob")


def _parse_envelope(blob: str) -> dict[str, JsonValue]:
    _reject_legacy_executable_blob(blob)
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise DecisionPersistenceCodecError("durable decision blob is not valid JSON") from exc
    return _require_mapping(parsed, "durable envelope")


def encode_checkpoint_blob(
    checkpoint: DecisionCheckpointState[T],
    *,
    payload_codecs: DecisionArtifactPayloadCodecRegistry,
) -> str:
    """Serialize one checkpoint snapshot for durable storage."""
    lifecycle = checkpoint.lifecycle
    finalization = checkpoint.finalization
    revision = checkpoint.revision
    payload: dict[str, JsonValue] = {
        "lifecycle": _encode_lifecycle(lifecycle),
        "finalization": _encode_finalization(
            finalization,
            payload_codecs=payload_codecs,
        ),
    }
    schema_version = _CHECKPOINT_SCHEMA_V2
    if revision is not None:
        payload["revision"] = _encode_revision(revision)
    envelope: dict[str, JsonValue] = {
        "schema_version": schema_version,
        "record_type": _RECORD_TYPE_CHECKPOINT,
        "payload": payload,
    }
    return _canonical_json_text(envelope)


def decode_checkpoint_blob(
    blob: str,
    *,
    payload_codecs: DecisionArtifactPayloadCodecRegistry,
) -> DecisionCheckpointState[object]:
    """Deserialize one checkpoint snapshot from durable storage."""
    envelope = _parse_envelope(blob)
    record_type = _require_str(envelope.get("record_type"), "record_type")
    if record_type != _RECORD_TYPE_CHECKPOINT:
        raise DecisionPersistenceRecordTypeError(
            f"unsupported checkpoint record_type: {record_type!r}",
        )
    schema_version = _require_int(envelope.get("schema_version"), "schema_version")
    if schema_version not in (_CHECKPOINT_SCHEMA_V1, _CHECKPOINT_SCHEMA_V2):
        raise DecisionPersistenceUnsupportedSchemaError(
            f"unsupported checkpoint schema version: {schema_version}",
        )
    payload = _require_mapping(envelope.get("payload"), "payload")
    lifecycle = _decode_lifecycle_wire(_require_mapping(payload.get("lifecycle"), "payload.lifecycle"))
    finalization = _decode_finalization_wire(
        _require_mapping(payload.get("finalization"), "payload.finalization"),
        payload_codecs=payload_codecs,
    )
    revision_wire = payload.get("revision")
    revision = None
    if revision_wire is not None:
        revision = _decode_revision_wire(_require_mapping(revision_wire, "payload.revision"))
    if schema_version == _CHECKPOINT_SCHEMA_V1:
        return DecisionCheckpointState(
            lifecycle=lifecycle,
            finalization=finalization,
            revision=None,
        )
    return DecisionCheckpointState(
        lifecycle=lifecycle,
        finalization=finalization,
        revision=revision,
    )


def encode_outcome_blob(
    outcome: AuthoritativeAcceptedDecision[T] | AuthoritativeResolutionRecord,
    *,
    payload_codecs: DecisionArtifactPayloadCodecRegistry,
) -> str:
    """Serialize one authoritative outcome for durable storage."""
    if type(outcome) is AuthoritativeAcceptedDecision:
        envelope: dict[str, JsonValue] = {
            "schema_version": _OUTCOME_SCHEMA_V1,
            "record_type": _RECORD_TYPE_ACCEPTED,
            "payload": _encode_accepted(outcome, payload_codecs=payload_codecs),
        }
    elif type(outcome) is AuthoritativeResolutionRecord:
        envelope = {
            "schema_version": _OUTCOME_SCHEMA_V1,
            "record_type": _RECORD_TYPE_RESOLUTION,
            "payload": _encode_resolution_record(outcome),
        }
    else:
        raise DecisionPersistenceCodecError(
            "outcome must be AuthoritativeAcceptedDecision or AuthoritativeResolutionRecord",
        )
    return _canonical_json_text(envelope)


def decode_outcome_blob(
    blob: str,
    *,
    payload_codecs: DecisionArtifactPayloadCodecRegistry,
) -> AuthoritativeAcceptedDecision[object] | AuthoritativeResolutionRecord:
    """Deserialize one authoritative outcome from durable storage."""
    envelope = _parse_envelope(blob)
    record_type = _require_str(envelope.get("record_type"), "record_type")
    schema_version = _require_int(envelope.get("schema_version"), "schema_version")
    if schema_version != _OUTCOME_SCHEMA_V1:
        raise DecisionPersistenceUnsupportedSchemaError(
            f"unsupported outcome schema version: {schema_version}",
        )
    payload = _require_mapping(envelope.get("payload"), "payload")
    if record_type == _RECORD_TYPE_ACCEPTED:
        return _decode_accepted_wire(payload, payload_codecs=payload_codecs)
    if record_type == _RECORD_TYPE_RESOLUTION:
        return _decode_resolution_record_wire(payload)
    raise DecisionPersistenceRecordTypeError(
        f"unsupported outcome record_type: {record_type!r}",
    )


def encode_identity_wire(identity: DecisionIdentity) -> dict[str, JsonValue]:
    """Encode one decision identity for codec tests and diagnostics."""
    return _encode_identity(identity)


def decode_identity_wire(wire: dict[str, JsonValue]) -> DecisionIdentity:
    """Decode one decision identity from explicit wire fields."""
    return _decode_identity_wire(wire)


def encode_finalization_key_wire(key: DecisionFinalizationKey) -> dict[str, JsonValue]:
    """Encode one finalization key for codec tests and diagnostics."""
    return _encode_finalization_key(key)


def decode_finalization_key_wire(wire: dict[str, JsonValue]) -> DecisionFinalizationKey:
    """Decode one finalization key from explicit wire fields."""
    return _decode_finalization_key_wire(wire)


def decision_finalization_key_from_identity_wire(
    wire: dict[str, JsonValue],
) -> DecisionFinalizationKey:
    """Derive canonical finalization key from one encoded identity."""
    return decision_finalization_key(_decode_identity_wire(wire))


__all__ = [
    "decode_checkpoint_blob",
    "decode_finalization_key_wire",
    "decode_identity_wire",
    "decode_outcome_blob",
    "decision_finalization_key_from_identity_wire",
    "encode_checkpoint_blob",
    "encode_finalization_key_wire",
    "encode_identity_wire",
    "encode_outcome_blob",
]
