"""Typed shard index contract for VPI data packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackFormatError,
    VpiDataPackIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.json_decode import (
    require_int,
    require_mapping,
    require_sha256_hex,
    require_str,
)


@dataclass(frozen=True, slots=True)
class ShardDescriptor:
    ordinal: int
    relative_path: str
    record_count: int
    sha256: str
    source_ref_count: int
    source_ref_set_sha256: str
    schema_version: str

    def __post_init__(self) -> None:
        if self.ordinal < 1:
            raise ValueError("ordinal must be >= 1")
        if self.record_count <= 0:
            raise ValueError("record_count must be > 0")
        if self.source_ref_count <= 0:
            raise ValueError("source_ref_count must be > 0")
        if self.source_ref_count != self.record_count:
            raise ValueError("source_ref_count must equal record_count")
        normalized_sha256 = self.sha256.lower()
        if len(normalized_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_sha256
        ):
            raise ValueError("sha256 must be a 64-character lowercase hex digest")
        normalized_source_ref_set_sha256 = self.source_ref_set_sha256.lower()
        if len(normalized_source_ref_set_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in normalized_source_ref_set_sha256
        ):
            raise ValueError("source_ref_set_sha256 must be a 64-character lowercase hex digest")
        if not self.relative_path.strip():
            raise ValueError("relative_path must be non-empty")


def _descriptor_to_dict(descriptor: ShardDescriptor) -> dict[str, object]:
    return {
        "ordinal": descriptor.ordinal,
        "relative_path": descriptor.relative_path,
        "record_count": descriptor.record_count,
        "sha256": descriptor.sha256,
        "source_ref_count": descriptor.source_ref_count,
        "source_ref_set_sha256": descriptor.source_ref_set_sha256,
        "schema_version": descriptor.schema_version,
    }


def _descriptor_from_dict(payload: dict[str, object], *, field_name: str) -> ShardDescriptor:
    return ShardDescriptor(
        ordinal=require_int(payload, "ordinal", minimum=1),
        relative_path=require_str(payload, "relative_path"),
        record_count=require_int(payload, "record_count", minimum=1),
        sha256=require_sha256_hex(payload, "sha256"),
        source_ref_count=require_int(payload, "source_ref_count", minimum=1),
        source_ref_set_sha256=require_sha256_hex(payload, "source_ref_set_sha256"),
        schema_version=require_str(payload, "schema_version"),
    )


def _validate_unique_ordinals(
    descriptors: tuple[ShardDescriptor, ...],
    *,
    field_name: str,
) -> None:
    ordinals = [descriptor.ordinal for descriptor in descriptors]
    if len(set(ordinals)) != len(ordinals):
        raise VpiDataPackFormatError(f"duplicate ordinals in {field_name}")


@dataclass(frozen=True, slots=True)
class ShardIndex:
    shard_count: int
    relational_shards: tuple[ShardDescriptor, ...]
    embedding_shards: tuple[ShardDescriptor, ...]

    def __post_init__(self) -> None:
        if self.shard_count <= 0:
            raise ValueError("shard_count must be > 0")
        if len(self.relational_shards) != self.shard_count:
            raise ValueError("relational_shards length must equal shard_count")
        if len(self.embedding_shards) != self.shard_count:
            raise ValueError("embedding_shards length must equal shard_count")


def shard_index_to_json_dict(shard_index: ShardIndex) -> dict[str, object]:
    return {
        "shard_count": shard_index.shard_count,
        "relational_shards": [
            _descriptor_to_dict(descriptor) for descriptor in shard_index.relational_shards
        ],
        "embedding_shards": [
            _descriptor_to_dict(descriptor) for descriptor in shard_index.embedding_shards
        ],
    }


def shard_index_from_json_dict(payload: dict[str, object]) -> ShardIndex:
    shard_count = require_int(payload, "shard_count", minimum=1)
    relational_raw = payload.get("relational_shards")
    embedding_raw = payload.get("embedding_shards")
    if not isinstance(relational_raw, list):
        raise VpiDataPackFormatError("relational_shards must be a list")
    if not isinstance(embedding_raw, list):
        raise VpiDataPackFormatError("embedding_shards must be a list")
    relational_shards = tuple(
        _descriptor_from_dict(require_mapping(item, field_name="relational_shards[]"), field_name="relational_shards[]")
        for item in relational_raw
    )
    embedding_shards = tuple(
        _descriptor_from_dict(require_mapping(item, field_name="embedding_shards[]"), field_name="embedding_shards[]")
        for item in embedding_raw
    )
    _validate_unique_ordinals(relational_shards, field_name="relational_shards")
    _validate_unique_ordinals(embedding_shards, field_name="embedding_shards")
    return ShardIndex(
        shard_count=shard_count,
        relational_shards=relational_shards,
        embedding_shards=embedding_shards,
    )


def write_shard_index_file(path: Path, shard_index: ShardIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(shard_index_to_json_dict(shard_index), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def read_shard_index_file(path: Path) -> ShardIndex:
    if not path.is_file():
        raise VpiDataPackIntegrityError(f"shard index not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VpiDataPackFormatError("shard index root must be a JSON object")
    return shard_index_from_json_dict(payload)
