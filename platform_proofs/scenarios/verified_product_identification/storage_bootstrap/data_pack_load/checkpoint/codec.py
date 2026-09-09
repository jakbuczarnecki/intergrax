"""Explicit typed codec for durable storage bootstrap checkpoint state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.json_decode import (
    JsonValue,
    require_int,
    require_known_keys,
    require_mapping,
    require_optional_int,
    require_optional_str,
    require_str,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.contracts import (
    VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
    BootstrapCheckpointState,
    BootstrapRunIdentity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.errors import (
    CheckpointCorrupt,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchPhase,
    RelationalTargetId,
    VectorTargetId,
    VerificationMode,
)

_CHECKPOINT_ENVELOPE_KEYS = frozenset({"schema_version", "state_checksum_sha256", "state"})
_STATE_KEYS = frozenset(
    {
        "run_identity",
        "batch_size",
        "total_records",
        "batch_count",
        "last_committed_batch_number",
        "last_committed_global_row_index",
        "last_committed_identity",
        "committed_record_count",
        "batch_phases",
        "state_revision",
        "updated_at_utc",
        "current_batch_phase",
    }
)
_RUN_IDENTITY_KEYS = frozenset(
    {
        "data_pack_content_identity",
        "data_pack_version",
        "record_count",
        "batch_size",
        "relational_target",
        "vector_target",
        "verification_mode",
        "ordering_policy",
        "state_schema_version",
        "source_dataset_sha256",
        "embedding_model_identity",
    }
)
_FORBIDDEN_SERIALIZED_KEYS = frozenset(
    {
        "password",
        "dsn",
        "api_key",
        "credentials",
        "record_json",
        "semantic_text",
        "dense_embedding",
        "embedding",
    }
)


@dataclass(frozen=True, slots=True)
class EncodedCheckpoint:
    text: str
    bytes_payload: bytes


def run_identity_digest(run_identity: BootstrapRunIdentity) -> str:
    """Content-bound checkpoint namespace — operational fields validated after load."""
    canonical = json.dumps(
        {
            "data_pack_content_identity": run_identity.data_pack_content_identity,
            "state_schema_version": run_identity.state_schema_version,
            "ordering_policy": run_identity.ordering_policy,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_identity_to_json_dict(run_identity: BootstrapRunIdentity) -> dict[str, JsonValue]:
    return {
        "data_pack_content_identity": run_identity.data_pack_content_identity,
        "data_pack_version": run_identity.data_pack_version,
        "record_count": run_identity.record_count,
        "batch_size": run_identity.batch_size,
        "relational_target": str(run_identity.relational_target),
        "vector_target": str(run_identity.vector_target),
        "verification_mode": run_identity.verification_mode.value,
        "ordering_policy": run_identity.ordering_policy,
        "state_schema_version": run_identity.state_schema_version,
        "source_dataset_sha256": run_identity.source_dataset_sha256,
        "embedding_model_identity": run_identity.embedding_model_identity,
    }


def checkpoint_state_to_json_dict(state: BootstrapCheckpointState) -> dict[str, JsonValue]:
    return {
        "run_identity": run_identity_to_json_dict(state.run_identity),
        "batch_size": state.batch_size,
        "total_records": state.total_records,
        "batch_count": state.batch_count,
        "last_committed_batch_number": state.last_committed_batch_number,
        "last_committed_global_row_index": state.last_committed_global_row_index,
        "last_committed_identity": state.last_committed_identity,
        "committed_record_count": state.committed_record_count,
        "batch_phases": [phase.value for phase in state.batch_phases],
        "state_revision": state.state_revision,
        "updated_at_utc": state.updated_at_utc,
        "current_batch_phase": (
            state.current_batch_phase.value if state.current_batch_phase is not None else None
        ),
    }


def encode_checkpoint(state: BootstrapCheckpointState) -> EncodedCheckpoint:
    state_payload = checkpoint_state_to_json_dict(state)
    _assert_no_forbidden_keys(state_payload)
    canonical_state = json.dumps(
        state_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    checksum = hashlib.sha256(canonical_state.encode("utf-8")).hexdigest()
    envelope: dict[str, JsonValue] = {
        "schema_version": VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
        "state_checksum_sha256": checksum,
        "state": state_payload,
    }
    text = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    return EncodedCheckpoint(text=text, bytes_payload=text.encode("utf-8"))


def decode_checkpoint(data: str | bytes) -> BootstrapCheckpointState:
    try:
        raw_text = data.decode("utf-8") if isinstance(data, bytes) else data
        envelope = json.loads(raw_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorrupt("checkpoint envelope is not valid JSON") from exc
    if not isinstance(envelope, dict):
        raise CheckpointCorrupt("checkpoint envelope must be a JSON object")
    try:
        require_known_keys(envelope, allowed=_CHECKPOINT_ENVELOPE_KEYS, context="checkpoint envelope")
        schema_version = require_str(envelope, "schema_version")
        if schema_version != VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCorrupt(f"unsupported checkpoint schema version: {schema_version}")
        checksum = require_str(envelope, "state_checksum_sha256")
        state_payload = require_mapping(envelope["state"], field_name="state")
        require_known_keys(state_payload, allowed=_STATE_KEYS, context="checkpoint state")
        canonical_state = json.dumps(
            state_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        expected_checksum = hashlib.sha256(canonical_state.encode("utf-8")).hexdigest()
        if checksum != expected_checksum:
            raise CheckpointCorrupt("checkpoint state checksum mismatch")
        return _checkpoint_state_from_json_dict(state_payload)
    except CheckpointCorrupt:
        raise
    except ValueError as exc:
        raise CheckpointCorrupt(str(exc)) from exc


def _checkpoint_state_from_json_dict(payload: dict[str, JsonValue]) -> BootstrapCheckpointState:
    run_identity_payload = require_mapping(payload["run_identity"], field_name="run_identity")
    require_known_keys(run_identity_payload, allowed=_RUN_IDENTITY_KEYS, context="run identity")
    verification_raw = require_str(run_identity_payload, "verification_mode")
    try:
        verification_mode = VerificationMode(verification_raw)
    except ValueError as exc:
        raise CheckpointCorrupt(f"invalid verification_mode: {verification_raw}") from exc
    run_identity = BootstrapRunIdentity(
        data_pack_content_identity=require_str(run_identity_payload, "data_pack_content_identity"),
        data_pack_version=require_str(run_identity_payload, "data_pack_version"),
        record_count=require_int(run_identity_payload, "record_count", minimum=0),
        batch_size=require_int(run_identity_payload, "batch_size", minimum=1),
        relational_target=RelationalTargetId(require_str(run_identity_payload, "relational_target")),
        vector_target=VectorTargetId(require_str(run_identity_payload, "vector_target")),
        verification_mode=verification_mode,
        ordering_policy=require_str(run_identity_payload, "ordering_policy"),
        state_schema_version=require_str(run_identity_payload, "state_schema_version"),
        source_dataset_sha256=require_str(run_identity_payload, "source_dataset_sha256"),
        embedding_model_identity=require_str(run_identity_payload, "embedding_model_identity"),
    )
    batch_phases_raw = payload.get("batch_phases")
    if not isinstance(batch_phases_raw, list):
        raise CheckpointCorrupt("batch_phases must be a JSON array")
    batch_phases: list[BootstrapBatchPhase] = []
    for index, raw_phase in enumerate(batch_phases_raw):
        if not isinstance(raw_phase, str):
            raise CheckpointCorrupt(f"batch_phases[{index}] must be a string")
        try:
            batch_phases.append(BootstrapBatchPhase(raw_phase))
        except ValueError as exc:
            raise CheckpointCorrupt(f"invalid batch phase: {raw_phase}") from exc
    current_phase_raw = payload.get("current_batch_phase")
    current_batch_phase: BootstrapBatchPhase | None = None
    if current_phase_raw is not None:
        if not isinstance(current_phase_raw, str):
            raise CheckpointCorrupt("current_batch_phase must be a string or null")
        try:
            current_batch_phase = BootstrapBatchPhase(current_phase_raw)
        except ValueError as exc:
            raise CheckpointCorrupt(f"invalid current_batch_phase: {current_phase_raw}") from exc
    return BootstrapCheckpointState(
        schema_version=VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
        run_identity=run_identity,
        batch_size=require_int(payload, "batch_size", minimum=1),
        total_records=require_int(payload, "total_records", minimum=0),
        batch_count=require_int(payload, "batch_count", minimum=0),
        last_committed_batch_number=require_optional_int(
            payload,
            "last_committed_batch_number",
            minimum=0,
        ),
        last_committed_global_row_index=require_optional_int(
            payload,
            "last_committed_global_row_index",
            minimum=0,
        ),
        last_committed_identity=require_optional_str(payload, "last_committed_identity"),
        committed_record_count=require_int(payload, "committed_record_count", minimum=0),
        batch_phases=tuple(batch_phases),
        state_revision=require_int(payload, "state_revision", minimum=1),
        updated_at_utc=require_str(payload, "updated_at_utc"),
        current_batch_phase=current_batch_phase,
    )


def _assert_no_forbidden_keys(payload: dict[str, JsonValue]) -> None:
    for key in payload:
        lowered = key.lower()
        if lowered in _FORBIDDEN_SERIALIZED_KEYS:
            raise ValueError(f"forbidden checkpoint field: {key}")
        value = payload[key]
        if isinstance(value, dict):
            _assert_no_forbidden_keys(value)
