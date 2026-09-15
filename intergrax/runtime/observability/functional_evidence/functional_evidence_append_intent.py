# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Durable append-intent metadata for functional evidence projection crash safety (DIAG-FUNCTIONAL-READ-R1-R2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol, runtime_checkable

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)

_APPEND_INTENT_SCHEMA_V1 = "intergrax.functional_evidence.append_intent.v1"
_APPEND_PENDING_ROW_PREFIX = "appendpending:"
_EVIDENCE_ID_FIELD = "evidence_id"


class FunctionalEvidenceAppendFaultBoundary(Enum):
    """Deterministic crash-injection boundaries for append projection protocol."""

    AFTER_INTENT = "after_intent"
    AFTER_CANONICAL = "after_canonical"
    AFTER_V2 = "after_v2"
    AFTER_V1 = "after_v1"
    BEFORE_INTENT_CLEAR = "before_intent_clear"


@runtime_checkable
class FunctionalEvidenceAppendFaultInjector(Protocol):
    """Optional test seam for deterministic append-path interruption."""

    def should_fault_after(self, boundary: FunctionalEvidenceAppendFaultBoundary) -> bool:
        """Return True once to simulate a crash immediately after the named boundary."""


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceAppendIntent:
    """Derived consistency metadata — not functional evidence truth."""

    schema_version: Literal["intergrax.functional_evidence.append_intent.v1"]
    evidence_id: str


def functional_evidence_append_pending_row_key(
    *,
    task_id: TaskId,
    run_id: RunId,
    evidence_id: str,
) -> str:
    return f"{_APPEND_PENDING_ROW_PREFIX}{task_id}:{run_id}:{evidence_id}"


def functional_evidence_append_pending_row_key_prefix(
    *,
    task_id: TaskId,
    run_id: RunId,
) -> str:
    return f"{_APPEND_PENDING_ROW_PREFIX}{task_id}:{run_id}:"


def encode_functional_evidence_append_intent(
    intent: FunctionalEvidenceAppendIntent,
) -> dict[str, str]:
    return {
        "schema_version": intent.schema_version,
        _EVIDENCE_ID_FIELD: intent.evidence_id,
    }


def decode_functional_evidence_append_intent(data: object) -> FunctionalEvidenceAppendIntent:
    if not isinstance(data, dict):
        raise ValueError("invalid functional evidence append intent")
    schema_version = data.get("schema_version")
    if schema_version != _APPEND_INTENT_SCHEMA_V1:
        raise ValueError("unsupported functional evidence append intent schema")
    evidence_id = data.get(_EVIDENCE_ID_FIELD)
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("invalid functional evidence append intent evidence_id")
    return FunctionalEvidenceAppendIntent(
        schema_version=_APPEND_INTENT_SCHEMA_V1,
        evidence_id=evidence_id,
    )


class FunctionalEvidenceAppendIntentStore:
    """Execution-scoped durable append-intent persistence."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        self._document_store = document_store

    def create_pending(
        self,
        *,
        partition_key: str,
        task_id: TaskId,
        run_id: RunId,
        evidence_id: str,
    ) -> None:
        intent = FunctionalEvidenceAppendIntent(
            schema_version=_APPEND_INTENT_SCHEMA_V1,
            evidence_id=evidence_id,
        )
        document = DocumentRecord(
            partition_key=partition_key,
            row_key=functional_evidence_append_pending_row_key(
                task_id=task_id,
                run_id=run_id,
                evidence_id=evidence_id,
            ),
            data=encode_functional_evidence_append_intent(intent),
        )
        self._document_store.put_if_absent(document)

    def load_pending(
        self,
        *,
        partition_key: str,
        task_id: TaskId,
        run_id: RunId,
        evidence_id: str,
    ) -> FunctionalEvidenceAppendIntent | None:
        row_key = functional_evidence_append_pending_row_key(
            task_id=task_id,
            run_id=run_id,
            evidence_id=evidence_id,
        )
        record = self._document_store.get(partition_key, row_key)
        if record is None:
            return None
        return decode_functional_evidence_append_intent(dict(record.data))

    def has_pending_for_execution(
        self,
        *,
        partition_key: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> bool:
        prefix = functional_evidence_append_pending_row_key_prefix(
            task_id=task_id,
            run_id=run_id,
        )
        page = self._document_store.query(
            partition_key,
            limit=1,
            row_key_prefix=prefix,
        )
        return bool(page.documents)

    def clear_pending(
        self,
        *,
        partition_key: str,
        task_id: TaskId,
        run_id: RunId,
        evidence_id: str,
    ) -> bool:
        row_key = functional_evidence_append_pending_row_key(
            task_id=task_id,
            run_id=run_id,
            evidence_id=evidence_id,
        )
        intent = FunctionalEvidenceAppendIntent(
            schema_version=_APPEND_INTENT_SCHEMA_V1,
            evidence_id=evidence_id,
        )
        expected = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data=encode_functional_evidence_append_intent(intent),
        )
        if self._document_store.delete_if_match(expected=expected):
            return True
        return self._document_store.get(partition_key, row_key) is None


__all__ = [
    "FunctionalEvidenceAppendFaultBoundary",
    "FunctionalEvidenceAppendFaultInjector",
    "FunctionalEvidenceAppendIntent",
    "FunctionalEvidenceAppendIntentStore",
    "decode_functional_evidence_append_intent",
    "encode_functional_evidence_append_intent",
    "functional_evidence_append_pending_row_key",
    "functional_evidence_append_pending_row_key_prefix",
]
