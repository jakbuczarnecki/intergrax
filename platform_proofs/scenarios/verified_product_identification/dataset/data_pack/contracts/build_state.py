"""Typed resumable builder state — separate from distributable Data Pack v1 manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildStateError,
    VpiDataPackFormatError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.json_decode import (
    JsonValue,
    require_int,
    require_known_keys,
    require_mapping,
    require_optional_str,
    require_sha256_hex,
    require_str,
)

VPI_DATA_PACK_BUILD_STATE_VERSION = "vpi.data_pack.build_state/1"

_BUILD_STATE_KEYS = frozenset(
    {
        "state_version",
        "build_id",
        "content_identity",
        "expected_record_count",
        "shard_size",
        "shard_count",
        "catalog_id",
        "started_at_utc",
        "updated_at_utc",
        "completed_shards",
        "shards",
    }
)
_SHARD_STATE_KEYS = frozenset(
    {
        "ordinal",
        "start_row_index",
        "end_row_index_exclusive",
        "expected_record_count",
        "status",
        "relational_relative_path",
        "embedding_relative_path",
        "attempt",
        "relational_sha256",
        "embedding_sha256",
        "relational_source_ref_set_sha256",
        "embedding_source_ref_set_sha256",
        "last_error_code",
        "last_error_message",
    }
)


class DataPackShardStatus(StrEnum):
    PENDING = "PENDING"
    DERIVING = "DERIVING"
    EMBEDDING = "EMBEDDING"
    WRITING = "WRITING"
    VALIDATING = "VALIDATING"
    READY = "READY"


@dataclass(frozen=True, slots=True)
class DataPackShardBuildState:
    ordinal: int
    start_row_index: int
    end_row_index_exclusive: int
    expected_record_count: int
    status: DataPackShardStatus
    relational_relative_path: str | None
    embedding_relative_path: str | None
    attempt: int
    relational_sha256: str | None = None
    embedding_sha256: str | None = None
    relational_source_ref_set_sha256: str | None = None
    embedding_source_ref_set_sha256: str | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 1:
            raise ValueError("ordinal must be >= 1")
        if self.start_row_index < 0:
            raise ValueError("start_row_index must be >= 0")
        if self.end_row_index_exclusive <= self.start_row_index:
            raise ValueError("end_row_index_exclusive must be > start_row_index")
        if self.expected_record_count <= 0:
            raise ValueError("expected_record_count must be > 0")
        if self.attempt < 0:
            raise ValueError("attempt must be >= 0")
        if self.status is DataPackShardStatus.READY:
            if self.relational_relative_path is None:
                raise ValueError("relational_relative_path is required when shard status is READY")
            if self.embedding_relative_path is None:
                raise ValueError("embedding_relative_path is required when shard status is READY")
            if self.relational_sha256 is None:
                raise ValueError("relational_sha256 is required when shard status is READY")
            if self.embedding_sha256 is None:
                raise ValueError("embedding_sha256 is required when shard status is READY")
            if self.relational_source_ref_set_sha256 is None:
                raise ValueError(
                    "relational_source_ref_set_sha256 is required when shard status is READY"
                )
            if self.embedding_source_ref_set_sha256 is None:
                raise ValueError(
                    "embedding_source_ref_set_sha256 is required when shard status is READY"
                )


@dataclass(frozen=True, slots=True)
class DataPackBuildState:
    state_version: str
    build_id: str
    content_identity: str
    expected_record_count: int
    shard_size: int
    shard_count: int
    catalog_id: str
    started_at_utc: str
    updated_at_utc: str
    completed_shards: int
    shards: tuple[DataPackShardBuildState, ...]

    def __post_init__(self) -> None:
        if self.state_version != VPI_DATA_PACK_BUILD_STATE_VERSION:
            raise ValueError(f"unsupported build state version: {self.state_version}")
        if not self.build_id.strip():
            raise ValueError("build_id must be non-empty")
        if not self.content_identity.strip():
            raise ValueError("content_identity must be non-empty")
        if self.expected_record_count <= 0:
            raise ValueError("expected_record_count must be > 0")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be > 0")
        if self.shard_count <= 0:
            raise ValueError("shard_count must be > 0")
        if len(self.shards) != self.shard_count:
            raise ValueError("shards length must equal shard_count")
        if self.completed_shards < 0 or self.completed_shards > self.shard_count:
            raise ValueError("completed_shards out of range")
        ready_count = sum(1 for shard in self.shards if shard.status is DataPackShardStatus.READY)
        if ready_count != self.completed_shards:
            raise ValueError("completed_shards must equal READY shard count")
        _validate_shard_plan_integrity(
            expected_record_count=self.expected_record_count,
            shards=self.shards,
        )


def _validate_shard_plan_integrity(
    *,
    expected_record_count: int,
    shards: tuple[DataPackShardBuildState, ...],
) -> None:
    shard_count = len(shards)
    ordinals = tuple(shard.ordinal for shard in shards)
    seen_ordinals: set[int] = set()
    for ordinal in ordinals:
        if ordinal in seen_ordinals:
            raise ValueError(f"duplicate shard ordinal: {ordinal}")
        seen_ordinals.add(ordinal)

    expected_ordinals = tuple(range(1, shard_count + 1))
    if ordinals != expected_ordinals:
        raise ValueError(
            f"shard ordinals must be contiguous and ordered 1..{shard_count}; got {ordinals}"
        )

    first_shard = shards[0]
    if first_shard.start_row_index != 0:
        raise ValueError(
            f"first shard must start at row index 0; got {first_shard.start_row_index}"
        )

    total_records = 0
    for index, shard in enumerate(shards):
        range_length = shard.end_row_index_exclusive - shard.start_row_index
        if range_length != shard.expected_record_count:
            raise ValueError(
                f"shard {shard.ordinal} range length {range_length} "
                f"does not match expected_record_count {shard.expected_record_count}"
            )
        total_records += shard.expected_record_count

        if index + 1 < shard_count:
            next_shard = shards[index + 1]
            if shard.end_row_index_exclusive != next_shard.start_row_index:
                if shard.end_row_index_exclusive > next_shard.start_row_index:
                    raise ValueError(
                        f"shard range overlap between ordinal {shard.ordinal} "
                        f"and {next_shard.ordinal}"
                    )
                raise ValueError(
                    f"shard range gap between ordinal {shard.ordinal} and "
                    f"{next_shard.ordinal}: {shard.end_row_index_exclusive} != "
                    f"{next_shard.start_row_index}"
                )

    last_shard = shards[-1]
    if last_shard.end_row_index_exclusive != expected_record_count:
        raise ValueError(
            f"last shard end {last_shard.end_row_index_exclusive} does not match "
            f"build expected_record_count {expected_record_count}"
        )
    if total_records != expected_record_count:
        raise ValueError(
            f"sum of shard expected_record_count {total_records} does not match "
            f"build expected_record_count {expected_record_count}"
        )


def _shard_status_from_str(value: str) -> DataPackShardStatus:
    try:
        return DataPackShardStatus(value)
    except ValueError as exc:
        raise VpiDataPackBuildStateError(f"invalid shard status: {value}") from exc


def shard_build_state_from_json_dict(payload: dict[str, JsonValue]) -> DataPackShardBuildState:
    try:
        require_known_keys(payload, allowed=_SHARD_STATE_KEYS, context="shard build state")
        status = _shard_status_from_str(require_str(payload, "status"))
        relational_sha256 = payload.get("relational_sha256")
        embedding_sha256 = payload.get("embedding_sha256")
        relational_digest = payload.get("relational_source_ref_set_sha256")
        embedding_digest = payload.get("embedding_source_ref_set_sha256")
        return DataPackShardBuildState(
            ordinal=require_int(payload, "ordinal", minimum=1),
            start_row_index=require_int(payload, "start_row_index", minimum=0),
            end_row_index_exclusive=require_int(payload, "end_row_index_exclusive", minimum=1),
            expected_record_count=require_int(payload, "expected_record_count", minimum=1),
            status=status,
            relational_relative_path=require_optional_str(payload, "relational_relative_path"),
            embedding_relative_path=require_optional_str(payload, "embedding_relative_path"),
            attempt=require_int(payload, "attempt", minimum=0),
            relational_sha256=(
                require_sha256_hex(payload, "relational_sha256")
                if relational_sha256 is not None
                else None
            ),
            embedding_sha256=(
                require_sha256_hex(payload, "embedding_sha256") if embedding_sha256 is not None else None
            ),
            relational_source_ref_set_sha256=(
                require_sha256_hex(payload, "relational_source_ref_set_sha256")
                if relational_digest is not None
                else None
            ),
            embedding_source_ref_set_sha256=(
                require_sha256_hex(payload, "embedding_source_ref_set_sha256")
                if embedding_digest is not None
                else None
            ),
            last_error_code=require_optional_str(payload, "last_error_code"),
            last_error_message=require_optional_str(payload, "last_error_message"),
        )
    except VpiDataPackFormatError as exc:
        raise VpiDataPackBuildStateError(str(exc)) from exc
    except ValueError as exc:
        raise VpiDataPackBuildStateError(str(exc)) from exc


def build_state_from_json_dict(payload: dict[str, JsonValue]) -> DataPackBuildState:
    try:
        require_known_keys(payload, allowed=_BUILD_STATE_KEYS, context="data pack build state")
        version = require_str(payload, "state_version")
        if version != VPI_DATA_PACK_BUILD_STATE_VERSION:
            raise VpiDataPackBuildStateError(f"unsupported build state version: {version}")
        shards_raw = payload.get("shards")
        if not isinstance(shards_raw, list):
            raise VpiDataPackBuildStateError("shards must be a list")
        shards = tuple(
            shard_build_state_from_json_dict(require_mapping(item, field_name=f"shards[{index}]"))
            for index, item in enumerate(shards_raw)
        )
        return DataPackBuildState(
            state_version=version,
            build_id=require_str(payload, "build_id"),
            content_identity=require_str(payload, "content_identity"),
            expected_record_count=require_int(payload, "expected_record_count", minimum=1),
            shard_size=require_int(payload, "shard_size", minimum=1),
            shard_count=require_int(payload, "shard_count", minimum=1),
            catalog_id=require_str(payload, "catalog_id"),
            started_at_utc=require_str(payload, "started_at_utc"),
            updated_at_utc=require_str(payload, "updated_at_utc"),
            completed_shards=require_int(payload, "completed_shards", minimum=0),
            shards=shards,
        )
    except VpiDataPackFormatError as exc:
        raise VpiDataPackBuildStateError(str(exc)) from exc
    except ValueError as exc:
        raise VpiDataPackBuildStateError(str(exc)) from exc


def shard_build_state_to_json_dict(shard: DataPackShardBuildState) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "ordinal": shard.ordinal,
        "start_row_index": shard.start_row_index,
        "end_row_index_exclusive": shard.end_row_index_exclusive,
        "expected_record_count": shard.expected_record_count,
        "status": shard.status.value,
        "relational_relative_path": shard.relational_relative_path,
        "embedding_relative_path": shard.embedding_relative_path,
        "attempt": shard.attempt,
        "relational_sha256": shard.relational_sha256,
        "embedding_sha256": shard.embedding_sha256,
        "relational_source_ref_set_sha256": shard.relational_source_ref_set_sha256,
        "embedding_source_ref_set_sha256": shard.embedding_source_ref_set_sha256,
        "last_error_code": shard.last_error_code,
        "last_error_message": shard.last_error_message,
    }
    return payload


def build_state_to_json_dict(state: DataPackBuildState) -> dict[str, JsonValue]:
    return {
        "state_version": state.state_version,
        "build_id": state.build_id,
        "content_identity": state.content_identity,
        "expected_record_count": state.expected_record_count,
        "shard_size": state.shard_size,
        "shard_count": state.shard_count,
        "catalog_id": state.catalog_id,
        "started_at_utc": state.started_at_utc,
        "updated_at_utc": state.updated_at_utc,
        "completed_shards": state.completed_shards,
        "shards": [shard_build_state_to_json_dict(shard) for shard in state.shards],
    }


def write_build_state_file(path: Path, state: DataPackBuildState) -> None:
    payload = build_state_to_json_dict(state)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_text(canonical + "\n", encoding="utf-8")
    temp_path.replace(path)


def read_build_state_file(path: Path) -> DataPackBuildState:
    if not path.is_file():
        raise VpiDataPackBuildStateError(f"build state file missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VpiDataPackBuildStateError(f"failed to read build state: {path}") from exc
    if not isinstance(payload, dict):
        raise VpiDataPackBuildStateError("build state must be a JSON object")
    return build_state_from_json_dict(payload)
