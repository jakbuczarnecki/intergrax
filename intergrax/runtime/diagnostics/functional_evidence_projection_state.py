# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Derived projection completeness metadata for functional evidence execution indexes (DIAG-FUNCTIONAL-READ-R1-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)

_PROJECTION_STATE_SCHEMA_V1 = "intergrax.functional_evidence.projection_state.v1"
_PROJECTION_STATE_ROW_PREFIX = "execidxmeta:"
_STATE_BUILDING = "building"
_STATE_COMPLETE = "complete"
_GENERATION_FIELD = "generation"
_V1_ROWS_RECONCILED_FIELD = "v1_rows_reconciled"
_STATE_FIELD = "state"


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceProjectionState:
    """Immutable derived migration/control metadata — not functional evidence truth."""

    schema_version: Literal["intergrax.functional_evidence.projection_state.v1"]
    state: Literal["building", "complete"]
    generation: int
    v1_rows_reconciled: int


def functional_evidence_projection_state_row_key(
    *,
    task_id: TaskId,
    run_id: RunId,
) -> str:
    return f"{_PROJECTION_STATE_ROW_PREFIX}{task_id}:{run_id}"


def encode_functional_evidence_projection_state(
    state: FunctionalEvidenceProjectionState,
) -> dict[str, str]:
    return {
        "schema_version": state.schema_version,
        _STATE_FIELD: state.state,
        _GENERATION_FIELD: str(state.generation),
        _V1_ROWS_RECONCILED_FIELD: str(state.v1_rows_reconciled),
    }


def decode_functional_evidence_projection_state(
    data: object,
) -> FunctionalEvidenceProjectionState:
    if not isinstance(data, dict):
        raise ValueError("invalid functional evidence projection state")
    schema_version = data.get("schema_version")
    if schema_version != _PROJECTION_STATE_SCHEMA_V1:
        raise ValueError("unsupported functional evidence projection state schema")
    raw_state = data.get(_STATE_FIELD)
    if raw_state not in {_STATE_BUILDING, _STATE_COMPLETE}:
        raise ValueError("invalid functional evidence projection state value")
    raw_generation = data.get(_GENERATION_FIELD)
    raw_v1_rows = data.get(_V1_ROWS_RECONCILED_FIELD)
    if not isinstance(raw_generation, str) or not raw_generation.isdigit():
        raise ValueError("invalid functional evidence projection generation")
    if not isinstance(raw_v1_rows, str) or not raw_v1_rows.isdigit():
        raise ValueError("invalid functional evidence projection v1_rows_reconciled")
    generation = int(raw_generation)
    if generation < 1:
        raise ValueError("invalid functional evidence projection generation")
    v1_rows_reconciled = int(raw_v1_rows)
    if v1_rows_reconciled < 0:
        raise ValueError("invalid functional evidence projection v1_rows_reconciled")
    return FunctionalEvidenceProjectionState(
        schema_version=_PROJECTION_STATE_SCHEMA_V1,
        state=raw_state,
        generation=generation,
        v1_rows_reconciled=v1_rows_reconciled,
    )


class FunctionalEvidenceProjectionStateStore:
    """Conditional persistence for execution-scoped v2 projection completeness."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        self._document_store = document_store

    def load(
        self,
        *,
        partition_key: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> FunctionalEvidenceProjectionState | None:
        row_key = functional_evidence_projection_state_row_key(
            task_id=task_id,
            run_id=run_id,
        )
        record = self._document_store.get(partition_key, row_key)
        if record is None:
            return None
        return decode_functional_evidence_projection_state(dict(record.data))

    def begin_rebuild(
        self,
        *,
        partition_key: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> int:
        row_key = functional_evidence_projection_state_row_key(
            task_id=task_id,
            run_id=run_id,
        )
        existing = self.load(
            partition_key=partition_key,
            task_id=task_id,
            run_id=run_id,
        )
        generation = 1 if existing is None else existing.generation + 1
        building = FunctionalEvidenceProjectionState(
            schema_version=_PROJECTION_STATE_SCHEMA_V1,
            state=_STATE_BUILDING,
            generation=generation,
            v1_rows_reconciled=0,
        )
        document = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data=encode_functional_evidence_projection_state(building),
        )
        if existing is None:
            self._document_store.put_if_absent(document)
            return generation
        expected = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data=encode_functional_evidence_projection_state(existing),
        )
        if not self._document_store.replace_if_match(
            expected=expected,
            replacement=document,
        ):
            reloaded = self.load(
                partition_key=partition_key,
                task_id=task_id,
                run_id=run_id,
            )
            if reloaded is not None and reloaded.state == _STATE_BUILDING:
                return reloaded.generation
            return self.begin_rebuild(
                partition_key=partition_key,
                task_id=task_id,
                run_id=run_id,
            )
        return generation

    def mark_complete(
        self,
        *,
        partition_key: str,
        task_id: TaskId,
        run_id: RunId,
        generation: int,
        v1_rows_reconciled: int,
    ) -> None:
        row_key = functional_evidence_projection_state_row_key(
            task_id=task_id,
            run_id=run_id,
        )
        existing = self.load(
            partition_key=partition_key,
            task_id=task_id,
            run_id=run_id,
        )
        if existing is None:
            raise ValueError("functional evidence projection state missing for completion")
        if existing.generation != generation or existing.state != _STATE_BUILDING:
            return
        complete = FunctionalEvidenceProjectionState(
            schema_version=_PROJECTION_STATE_SCHEMA_V1,
            state=_STATE_COMPLETE,
            generation=generation,
            v1_rows_reconciled=v1_rows_reconciled,
        )
        expected = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data=encode_functional_evidence_projection_state(existing),
        )
        replacement = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data=encode_functional_evidence_projection_state(complete),
        )
        self._document_store.replace_if_match(
            expected=expected,
            replacement=replacement,
        )

    def ensure_append_projection_complete(
        self,
        *,
        partition_key: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> None:
        existing = self.load(
            partition_key=partition_key,
            task_id=task_id,
            run_id=run_id,
        )
        if existing is not None:
            return
        row_key = functional_evidence_projection_state_row_key(
            task_id=task_id,
            run_id=run_id,
        )
        complete = FunctionalEvidenceProjectionState(
            schema_version=_PROJECTION_STATE_SCHEMA_V1,
            state=_STATE_COMPLETE,
            generation=1,
            v1_rows_reconciled=0,
        )
        self._document_store.put_if_absent(
            DocumentRecord(
                partition_key=partition_key,
                row_key=row_key,
                data=encode_functional_evidence_projection_state(complete),
            ),
        )


__all__ = [
    "FunctionalEvidenceProjectionState",
    "FunctionalEvidenceProjectionStateStore",
    "decode_functional_evidence_projection_state",
    "encode_functional_evidence_projection_state",
    "functional_evidence_projection_state_row_key",
]
