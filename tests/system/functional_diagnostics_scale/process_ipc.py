# © Artur Czarnecki. All rights reserved.

"""Typed IPC contracts for S1 scale worker subprocess communication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from intergrax.knowledge.contracts.validation import JsonObject, JsonValue

IPC_SCHEMA_VERSION = 1


class ScaleWorkerPhase(StrEnum):
    WRITE = "write"
    READ = "read"
    IDEMPOTENT = "idempotent"
    CONFLICT = "conflict"
    READ_WRITE = "read-write"
    RECOVERY = "recovery"


@dataclass(frozen=True, slots=True)
class ScaleWorkerCommand:
    phase: ScaleWorkerPhase
    collection_name: str
    cursor_secret_hex: str
    page_size: int
    query_page_limit: int
    worker_index: int
    worker_count: int
    seed: int
    profile_name: str

    def to_json_dict(self) -> JsonObject:
        return {
            "schema_version": IPC_SCHEMA_VERSION,
            "phase": self.phase.value,
            "collection_name": self.collection_name,
            "cursor_secret_hex": self.cursor_secret_hex,
            "page_size": self.page_size,
            "query_page_limit": self.query_page_limit,
            "worker_index": self.worker_index,
            "worker_count": self.worker_count,
            "seed": self.seed,
            "profile_name": self.profile_name,
        }


@dataclass(frozen=True, slots=True)
class ScaleWorkerResult:
    schema_version: int
    pid: int
    phase: str
    worker_index: int
    written_count: int
    read_count: int
    append_latency_ms: tuple[float, ...]
    read_latency_ms: tuple[float, ...]
    conflicts: int
    errors: int
    exit_code: int
    detail: str

    def to_json_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "pid": self.pid,
            "phase": self.phase,
            "worker_index": self.worker_index,
            "written_count": self.written_count,
            "read_count": self.read_count,
            "append_latency_ms": list(self.append_latency_ms),
            "read_latency_ms": list(self.read_latency_ms),
            "conflicts": self.conflicts,
            "errors": self.errors,
            "exit_code": self.exit_code,
            "detail": self.detail,
        }


def encode_ipc_payload(payload: JsonObject) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_worker_result(line: str) -> ScaleWorkerResult:
    raw: JsonValue = json.loads(line)
    if not isinstance(raw, dict):
        raise ValueError("scale_worker_result_invalid")
    return ScaleWorkerResult(
        schema_version=int(raw["schema_version"]),
        pid=int(raw["pid"]),
        phase=str(raw["phase"]),
        worker_index=int(raw["worker_index"]),
        written_count=int(raw["written_count"]),
        read_count=int(raw["read_count"]),
        append_latency_ms=tuple(float(item) for item in raw["append_latency_ms"]),
        read_latency_ms=tuple(float(item) for item in raw["read_latency_ms"]),
        conflicts=int(raw["conflicts"]),
        errors=int(raw["errors"]),
        exit_code=int(raw["exit_code"]),
        detail=str(raw.get("detail", "")),
    )


__all__ = [
    "IPC_SCHEMA_VERSION",
    "ScaleWorkerCommand",
    "ScaleWorkerPhase",
    "ScaleWorkerResult",
    "encode_ipc_payload",
    "parse_worker_result",
]
