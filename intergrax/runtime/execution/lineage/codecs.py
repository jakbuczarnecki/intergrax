# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strict codecs for execution lineage durable rows (DG-001 R1)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from intergrax.contracts.execution_identity import (
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAdmissionRecord,
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageAttemptDiscoveryRecord,
    ExecutionLineageAttemptScope,
    ExecutionLineageAttemptState,
    ExecutionLineageDiscoveryCoverageOrigin,
    ExecutionLineageDiscoveryRunState,
    ExecutionLineageError,
    ExecutionLineageRunScope,
    ExecutionLineageSegmentLifecycle,
    ExecutionLineageSegmentRecord,
    ExecutionLineageSealRecord,
    build_execution_lineage_attempt_scope,
    build_execution_lineage_run_scope,
)

_ATTEMPT_STATE_SCHEMA_V1 = 1
_ATTEMPT_STATE_SCHEMA_V2 = 2
_DISCOVERY_SCHEMA_V1 = 1
_SCHEMA_VERSION = 1


def _reject_unknown_keys(payload: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = set(payload.keys()) - set(allowed)
    if unknown:
        raise ExecutionLineageError(
            f"unknown execution lineage fields: {sorted(unknown)}",
        )


def encode_execution_lineage_attempt_scope(
    scope: ExecutionLineageAttemptScope,
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "tenant_id": scope.tenant_id,
        "task_id": str(scope.task_id),
        "run_id": str(scope.run_id),
        "attempt_id": str(scope.attempt_id),
    }


def decode_execution_lineage_attempt_scope(
    payload: Mapping[str, Any],
) -> ExecutionLineageAttemptScope:
    _reject_unknown_keys(
        payload,
        frozenset({"schema_version", "tenant_id", "task_id", "run_id", "attempt_id"}),
    )
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ExecutionLineageError(
            "unsupported execution lineage scope schema version"
        )
    return build_execution_lineage_attempt_scope(
        tenant_id=str(payload["tenant_id"]),
        task_id=validate_task_id(payload["task_id"]),
        run_id=validate_run_id(payload["run_id"]),
        attempt_id=validate_attempt_id(payload["attempt_id"]),
    )


def encode_execution_lineage_attempt_state(
    state: ExecutionLineageAttemptState,
) -> dict[str, Any]:
    return {
        "schema_version": _ATTEMPT_STATE_SCHEMA_V2,
        "scope": encode_execution_lineage_attempt_scope(state.scope),
        "generation": state.generation,
        "next_admission_position": state.next_admission_position,
        "active_segment_root_execution_id": (
            str(state.active_segment_root_execution_id)
            if state.active_segment_root_execution_id is not None
            else None
        ),
        "degraded": state.degraded,
        "sealed": state.sealed,
        "closure_kind": (
            state.closure_kind.value if state.closure_kind is not None else None
        ),
        "discovery_contract_version": state.discovery_contract_version,
    }


def decode_execution_lineage_attempt_state(
    payload: Mapping[str, Any],
) -> ExecutionLineageAttemptState:
    schema_version = payload.get("schema_version")
    if schema_version == _ATTEMPT_STATE_SCHEMA_V1:
        allowed = frozenset(
            {
                "schema_version",
                "scope",
                "generation",
                "next_admission_position",
                "active_segment_root_execution_id",
                "degraded",
                "sealed",
                "closure_kind",
            },
        )
        discovery_contract_version = None
    elif schema_version == _ATTEMPT_STATE_SCHEMA_V2:
        allowed = frozenset(
            {
                "schema_version",
                "scope",
                "generation",
                "next_admission_position",
                "active_segment_root_execution_id",
                "degraded",
                "sealed",
                "closure_kind",
                "discovery_contract_version",
            },
        )
        marker = payload.get("discovery_contract_version")
        if marker is not None and marker != 1:
            raise ExecutionLineageError("invalid discovery_contract_version marker")
        discovery_contract_version = marker
    else:
        raise ExecutionLineageError(
            "unsupported execution lineage attempt state schema version",
        )
    _reject_unknown_keys(payload, allowed)
    closure_raw = payload.get("closure_kind")
    closure_kind = (
        ExecutionLineageAttemptClosureKind(closure_raw)
        if closure_raw is not None
        else None
    )
    active_segment = payload.get("active_segment_root_execution_id")
    return ExecutionLineageAttemptState(
        scope=decode_execution_lineage_attempt_scope(payload["scope"]),
        generation=_require_positive_int(payload.get("generation"), label="generation"),
        next_admission_position=_require_positive_int(
            payload.get("next_admission_position"),
            label="next_admission_position",
        ),
        active_segment_root_execution_id=(
            validate_execution_id(active_segment)
            if active_segment is not None
            else None
        ),
        degraded=bool(payload.get("degraded")),
        sealed=bool(payload.get("sealed")),
        closure_kind=closure_kind,
        discovery_contract_version=discovery_contract_version,
    )


def encode_execution_lineage_segment_record(
    record: ExecutionLineageSegmentRecord,
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "scope": encode_execution_lineage_attempt_scope(record.scope),
        "root_execution_id": str(record.root_execution_id),
        "predecessor_root_execution_id": (
            str(record.predecessor_root_execution_id)
            if record.predecessor_root_execution_id is not None
            else None
        ),
        "lifecycle": record.lifecycle.value,
    }


def decode_execution_lineage_segment_record(
    payload: Mapping[str, Any],
) -> ExecutionLineageSegmentRecord:
    _reject_unknown_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "scope",
                "root_execution_id",
                "predecessor_root_execution_id",
                "lifecycle",
            },
        ),
    )
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ExecutionLineageError(
            "unsupported execution lineage segment schema version"
        )
    predecessor = payload.get("predecessor_root_execution_id")
    return ExecutionLineageSegmentRecord(
        scope=decode_execution_lineage_attempt_scope(payload["scope"]),
        root_execution_id=validate_execution_id(payload["root_execution_id"]),
        predecessor_root_execution_id=(
            validate_execution_id(predecessor) if predecessor is not None else None
        ),
        lifecycle=ExecutionLineageSegmentLifecycle(payload["lifecycle"]),
    )


def encode_execution_lineage_admission_record(
    record: ExecutionLineageAdmissionRecord,
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "scope": encode_execution_lineage_attempt_scope(record.scope),
        "segment_root_execution_id": str(record.segment_root_execution_id),
        "execution_id": str(record.execution_id),
        "parent_execution_id": (
            str(record.parent_execution_id)
            if record.parent_execution_id is not None
            else None
        ),
        "admission_position": record.admission_position,
        "graph_node_id": record.graph_node_id,
    }


def decode_execution_lineage_admission_record(
    payload: Mapping[str, Any],
) -> ExecutionLineageAdmissionRecord:
    _reject_unknown_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "scope",
                "segment_root_execution_id",
                "execution_id",
                "parent_execution_id",
                "admission_position",
                "graph_node_id",
            },
        ),
    )
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ExecutionLineageError(
            "unsupported execution lineage admission schema version"
        )
    parent = payload.get("parent_execution_id")
    return ExecutionLineageAdmissionRecord(
        scope=decode_execution_lineage_attempt_scope(payload["scope"]),
        segment_root_execution_id=validate_execution_id(
            payload["segment_root_execution_id"]
        ),
        execution_id=validate_execution_id(payload["execution_id"]),
        parent_execution_id=validate_execution_id(parent)
        if parent is not None
        else None,
        admission_position=_require_positive_int(
            payload.get("admission_position"),
            label="admission_position",
        ),
        graph_node_id=(
            str(payload["graph_node_id"])
            if payload.get("graph_node_id") is not None
            else None
        ),
    )


def encode_execution_lineage_seal_record(
    record: ExecutionLineageSealRecord,
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "scope": encode_execution_lineage_attempt_scope(record.scope),
        "closure_kind": record.closure_kind.value,
        "degraded": record.degraded,
    }


def decode_execution_lineage_seal_record(
    payload: Mapping[str, Any],
) -> ExecutionLineageSealRecord:
    _reject_unknown_keys(
        payload,
        frozenset({"schema_version", "scope", "closure_kind", "degraded"}),
    )
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ExecutionLineageError("unsupported execution lineage seal schema version")
    return ExecutionLineageSealRecord(
        scope=decode_execution_lineage_attempt_scope(payload["scope"]),
        closure_kind=ExecutionLineageAttemptClosureKind(payload["closure_kind"]),
        degraded=bool(payload.get("degraded")),
    )


def encode_execution_lineage_attempt_state_bytes(
    state: ExecutionLineageAttemptState,
) -> bytes:
    return json.dumps(
        encode_execution_lineage_attempt_state(state),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def decode_execution_lineage_attempt_state_bytes(
    raw: bytes,
) -> ExecutionLineageAttemptState:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionLineageError(
            "invalid execution lineage attempt state encoding"
        ) from exc
    if not isinstance(payload, dict):
        raise ExecutionLineageError("invalid execution lineage attempt state payload")
    return decode_execution_lineage_attempt_state(payload)


def _require_positive_int(raw: object, *, label: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        raise ExecutionLineageError(f"invalid execution lineage {label}")
    return raw


def _require_non_negative_int(raw: object, *, label: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        raise ExecutionLineageError(f"invalid execution lineage {label}")
    return raw


def encode_execution_lineage_run_scope(
    run_scope: ExecutionLineageRunScope,
) -> dict[str, Any]:
    return {
        "schema_version": _DISCOVERY_SCHEMA_V1,
        "tenant_id": run_scope.tenant_id,
        "task_id": str(run_scope.task_id),
        "run_id": str(run_scope.run_id),
    }


def decode_execution_lineage_run_scope(
    payload: Mapping[str, Any],
) -> ExecutionLineageRunScope:
    _reject_unknown_keys(
        payload,
        frozenset({"schema_version", "tenant_id", "task_id", "run_id"}),
    )
    if payload.get("schema_version") != _DISCOVERY_SCHEMA_V1:
        raise ExecutionLineageError(
            "unsupported execution lineage run scope schema version"
        )
    return build_execution_lineage_run_scope(
        tenant_id=str(payload["tenant_id"]),
        task_id=validate_task_id(payload["task_id"]),
        run_id=validate_run_id(payload["run_id"]),
    )


def encode_execution_lineage_attempt_discovery_record(
    record: ExecutionLineageAttemptDiscoveryRecord,
) -> dict[str, Any]:
    return {
        "schema_version": _DISCOVERY_SCHEMA_V1,
        "run_scope": encode_execution_lineage_run_scope(record.run_scope),
        "attempt_id": str(record.attempt_id),
        "discovery_position": record.discovery_position,
    }


def decode_execution_lineage_attempt_discovery_record(
    payload: Mapping[str, Any],
) -> ExecutionLineageAttemptDiscoveryRecord:
    _reject_unknown_keys(
        payload,
        frozenset({"schema_version", "run_scope", "attempt_id", "discovery_position"}),
    )
    if payload.get("schema_version") != _DISCOVERY_SCHEMA_V1:
        raise ExecutionLineageError(
            "unsupported execution lineage attempt discovery schema version",
        )
    return ExecutionLineageAttemptDiscoveryRecord(
        run_scope=decode_execution_lineage_run_scope(payload["run_scope"]),
        attempt_id=validate_attempt_id(payload["attempt_id"]),
        discovery_position=_require_positive_int(
            payload.get("discovery_position"),
            label="discovery_position",
        ),
    )


def encode_execution_lineage_discovery_run_state(
    state: ExecutionLineageDiscoveryRunState,
) -> dict[str, Any]:
    return {
        "schema_version": _DISCOVERY_SCHEMA_V1,
        "run_scope": encode_execution_lineage_run_scope(state.run_scope),
        "generation": state.generation,
        "next_discovery_position": state.next_discovery_position,
        "coverage_contract_version": state.coverage_contract_version,
        "coverage_origin": (
            state.coverage_origin.value if state.coverage_origin is not None else None
        ),
    }


def decode_execution_lineage_discovery_run_state(
    payload: Mapping[str, Any],
) -> ExecutionLineageDiscoveryRunState:
    _reject_unknown_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "run_scope",
                "generation",
                "next_discovery_position",
                "coverage_contract_version",
                "coverage_origin",
            },
        ),
    )
    if payload.get("schema_version") != _DISCOVERY_SCHEMA_V1:
        raise ExecutionLineageError(
            "unsupported execution lineage discovery run state schema version",
        )
    coverage_raw = payload.get("coverage_origin")
    coverage_origin = (
        ExecutionLineageDiscoveryCoverageOrigin(coverage_raw)
        if coverage_raw is not None
        else None
    )
    coverage_version = payload.get("coverage_contract_version")
    return ExecutionLineageDiscoveryRunState(
        run_scope=decode_execution_lineage_run_scope(payload["run_scope"]),
        generation=_require_non_negative_int(
            payload.get("generation"), label="generation"
        ),
        next_discovery_position=_require_positive_int(
            payload.get("next_discovery_position"),
            label="next_discovery_position",
        ),
        coverage_contract_version=(
            int(coverage_version) if coverage_version is not None else None
        ),
        coverage_origin=coverage_origin,
    )
