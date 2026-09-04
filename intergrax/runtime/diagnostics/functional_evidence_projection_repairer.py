# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded repair of incomplete functional evidence append projections (DIAG-FUNCTIONAL-READ-R1-R2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)
from intergrax.runtime.diagnostics.functional_evidence import PlatformFunctionalEvidence
from intergrax.runtime.diagnostics.functional_evidence_append_intent import (
    FunctionalEvidenceAppendIntentStore,
    decode_functional_evidence_append_intent,
    functional_evidence_append_pending_row_key_prefix,
)
from intergrax.runtime.diagnostics.functional_evidence_execution_index import (
    decode_execution_index_v1,
    decode_execution_index_v2,
    encode_execution_index_v1,
    encode_execution_index_v2,
    execution_index_v1_row_key,
    execution_index_v2_row_key_from_evidence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.functional_evidence_record_codec import (
    decode_functional_evidence_record,
)

_QUERY_PAGE_LIMIT = 5000
_RECORD_ROW_PREFIX = "record:"


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceAppendRepairOutcome:
    evidence_id: str
    repaired: bool
    orphan_intent_removed: bool


class FunctionalEvidenceProjectionRepairer:
    """Repairs execution-scoped pending append intents from canonical truth."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        query_page_limit: int = _QUERY_PAGE_LIMIT,
    ) -> None:
        self._document_store = document_store
        if type(query_page_limit) is not int or isinstance(query_page_limit, bool):
            raise ValueError("functional_evidence_projection_repair_page_limit_invalid")
        if query_page_limit < 1:
            raise ValueError("functional_evidence_projection_repair_page_limit_invalid")
        self._query_page_limit = query_page_limit
        self._append_intent_store = FunctionalEvidenceAppendIntentStore(document_store)

    def repair_execution_pending_appends(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
    ) -> tuple[FunctionalEvidenceAppendRepairOutcome, ...]:
        if not self._append_intent_store.has_pending_for_execution(
            partition_key=partition_key,
            task_id=task_id,
            run_id=run_id,
        ):
            return ()
        prefix = functional_evidence_append_pending_row_key_prefix(
            task_id=task_id,
            run_id=run_id,
        )
        outcomes: list[FunctionalEvidenceAppendRepairOutcome] = []
        continuation: str | None = None
        while True:
            page = self._document_store.query(
                partition_key,
                limit=self._query_page_limit,
                row_key_prefix=prefix,
                cursor=continuation,
            )
            for pending_document in page.documents:
                outcomes.append(
                    self._repair_pending_document(
                        pending_document,
                        tenant_id=tenant_id,
                        task_id=task_id,
                        run_id=run_id,
                        partition_key=partition_key,
                    ),
                )
            if page.next_cursor is None:
                break
            continuation = page.next_cursor
        return tuple(outcomes)

    def _repair_pending_document(
        self,
        pending_document: DocumentRecord,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
    ) -> FunctionalEvidenceAppendRepairOutcome:
        try:
            intent = decode_functional_evidence_append_intent(dict(pending_document.data))
        except ValueError as exc:
            raise FunctionalEvidencePersistenceIntegrityError(
                "invalid functional evidence append intent",
            ) from exc
        if intent.evidence_id not in pending_document.row_key:
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence append intent row key inconsistent with payload",
            )
        canonical = self._document_store.get(
            partition_key,
            f"{_RECORD_ROW_PREFIX}{intent.evidence_id}",
        )
        if canonical is None:
            cleared = self._append_intent_store.clear_pending(
                partition_key=partition_key,
                task_id=task_id,
                run_id=run_id,
                evidence_id=intent.evidence_id,
            )
            if not cleared:
                raise FunctionalEvidencePersistenceIntegrityError(
                    "functional evidence append intent orphan cleanup failed",
                )
            return FunctionalEvidenceAppendRepairOutcome(
                evidence_id=intent.evidence_id,
                repaired=False,
                orphan_intent_removed=True,
            )
        evidence = self._document_to_evidence(canonical)
        self._validate_execution_scope(
            evidence,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        )
        self._repair_indexes_from_canonical(
            evidence=evidence,
            partition_key=partition_key,
        )
        cleared = self._append_intent_store.clear_pending(
            partition_key=partition_key,
            task_id=task_id,
            run_id=run_id,
            evidence_id=intent.evidence_id,
        )
        if not cleared:
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence append intent completion failed",
            )
        return FunctionalEvidenceAppendRepairOutcome(
            evidence_id=intent.evidence_id,
            repaired=True,
            orphan_intent_removed=False,
        )

    def _repair_indexes_from_canonical(
        self,
        *,
        evidence: PlatformFunctionalEvidence,
        partition_key: str,
    ) -> None:
        v2_document = DocumentRecord(
            partition_key=partition_key,
            row_key=execution_index_v2_row_key_from_evidence(evidence),
            data=encode_execution_index_v2(evidence),
        )
        if not self._document_store.put_if_absent(v2_document):
            self._verify_v2_index_document(v2_document, evidence)
        v1_document = DocumentRecord(
            partition_key=partition_key,
            row_key=execution_index_v1_row_key(
                task_id=evidence.scope.task_id,
                run_id=evidence.scope.run_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data=encode_execution_index_v1(str(evidence.evidence_id)),
        )
        if not self._document_store.put_if_absent(v1_document):
            self._verify_v1_index_document(v1_document, evidence)

    def _verify_v2_index_document(
        self,
        document: DocumentRecord,
        evidence: PlatformFunctionalEvidence,
    ) -> None:
        existing = self._document_store.get(document.partition_key, document.row_key)
        if existing is None:
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index verification failed",
            )
        indexed = decode_execution_index_v2(dict(existing.data))
        if indexed.evidence_id != str(evidence.evidence_id):
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index conflicts with expected evidence_id",
            )
        if dict(existing.data) != dict(document.data):
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index conflicts with expected projection",
            )

    def _verify_v1_index_document(
        self,
        document: DocumentRecord,
        evidence: PlatformFunctionalEvidence,
    ) -> None:
        existing = self._document_store.get(document.partition_key, document.row_key)
        if existing is None:
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index verification failed",
            )
        indexed = decode_execution_index_v1(dict(existing.data))
        if indexed.evidence_id != str(evidence.evidence_id):
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index conflicts with expected evidence_id",
            )

    def _document_to_evidence(self, document: DocumentRecord) -> PlatformFunctionalEvidence:
        try:
            return decode_functional_evidence_record(dict(document.data))
        except ValueError as exc:
            raise FunctionalEvidencePersistenceIntegrityError(
                "invalid canonical functional evidence record",
            ) from exc

    @staticmethod
    def _validate_execution_scope(
        evidence: PlatformFunctionalEvidence,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> None:
        if (
            evidence.scope.tenant_id != tenant_id
            or evidence.scope.task_id != task_id
            or evidence.scope.run_id != run_id
        ):
            raise FunctionalEvidencePersistenceIntegrityError(
                "canonical functional evidence does not match execution index scope",
            )


__all__ = [
    "FunctionalEvidenceAppendRepairOutcome",
    "FunctionalEvidenceProjectionRepairer",
]
