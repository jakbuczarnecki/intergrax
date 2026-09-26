# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ConditionalDocumentStore-backed qualified marketplace tool execution intent (S24-GAP-02-P3)."""

from __future__ import annotations

from typing import Final

from pydantic import ValidationError

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    QualifiedMarketplaceToolExecutionIntent,
    QualifiedMarketplaceToolExecutionIntentConflictError,
    QualifiedMarketplaceToolExecutionIntentIntegrityError,
    QualifiedMarketplaceToolExecutionIntentUnavailableError,
    QualifiedMarketplaceToolExecutionIntentWriteOutcome,
    QualifiedMarketplaceToolExecutionIntentWriteResult,
    SCHEMA_QUALIFIED_MARKETPLACE_TOOL_EXECUTION_INTENT_V1,
)
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)

_DOCUMENT_STORE_PARTITION: Final = (
    "intergrax.qualified_marketplace_tool_execution_intent.v1"
)
_PERSISTENCE_SCHEMA: Final = (
    "intergrax.qualified_marketplace_tool_execution_intent.persistence.v1"
)
_PAYLOAD_FIELD: Final = "intent"


def _document_row_key(execution_request_id: str) -> str:
    return execution_request_id


def _encode_intent_record(
    intent: QualifiedMarketplaceToolExecutionIntent,
) -> DocumentRecord:
    return DocumentRecord(
        partition_key=_DOCUMENT_STORE_PARTITION,
        row_key=_document_row_key(intent.execution_request_id),
        data={
            "schema_version": _PERSISTENCE_SCHEMA,
            _PAYLOAD_FIELD: intent.model_dump(mode="json"),
        },
    )


def _decode_intent_record(
    document: DocumentRecord,
    *,
    execution_request_id: str,
) -> QualifiedMarketplaceToolExecutionIntent:
    data = dict(document.data)
    schema_version = data.get("schema_version")
    if schema_version != _PERSISTENCE_SCHEMA:
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "unsupported qualified marketplace tool execution intent persistence schema",
        )
    payload = data.get(_PAYLOAD_FIELD)
    if not isinstance(payload, dict):
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "qualified marketplace tool execution intent persistence payload is invalid",
        )
    if (
        payload.get("schema_version")
        != SCHEMA_QUALIFIED_MARKETPLACE_TOOL_EXECUTION_INTENT_V1
    ):
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "qualified marketplace tool execution intent semantic schema mismatch",
        )
    try:
        intent = QualifiedMarketplaceToolExecutionIntent.model_validate(payload)
    except ValidationError as exc:
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "qualified marketplace tool execution intent payload failed validation",
        ) from exc
    if document.partition_key != _DOCUMENT_STORE_PARTITION:
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "qualified marketplace tool execution intent partition mismatch",
        )
    if document.row_key != _document_row_key(execution_request_id):
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "qualified marketplace tool execution intent row key mismatch",
        )
    if intent.execution_request_id != execution_request_id:
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "qualified marketplace tool execution intent identity mismatch",
        )
    return intent


class DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository:
    """Durable intent keyed by execution_request_id — no overwrite."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "qualified marketplace tool execution intent requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    def record(
        self,
        intent: QualifiedMarketplaceToolExecutionIntent,
    ) -> QualifiedMarketplaceToolExecutionIntentWriteResult:
        document = _encode_intent_record(intent)
        try:
            created = self._document_store.put_if_absent(document)
        except Exception as exc:
            raise QualifiedMarketplaceToolExecutionIntentUnavailableError(
                "qualified marketplace tool execution intent write failed",
            ) from exc
        if created:
            return QualifiedMarketplaceToolExecutionIntentWriteResult(
                outcome=QualifiedMarketplaceToolExecutionIntentWriteOutcome.CREATED,
            )

        existing = self.get(execution_request_id=intent.execution_request_id)
        if existing == intent:
            return QualifiedMarketplaceToolExecutionIntentWriteResult(
                outcome=(
                    QualifiedMarketplaceToolExecutionIntentWriteOutcome.ALREADY_RECORDED_IDENTICAL
                ),
            )
        raise QualifiedMarketplaceToolExecutionIntentConflictError(
            "qualified marketplace tool execution intent identity conflict",
        )

    def get(
        self,
        *,
        execution_request_id: str,
    ) -> QualifiedMarketplaceToolExecutionIntent | None:
        cleaned = require_non_empty_text(
            execution_request_id,
            label="execution_request_id",
        )
        try:
            document = self._document_store.get(
                _DOCUMENT_STORE_PARTITION,
                _document_row_key(cleaned),
            )
        except Exception as exc:
            raise QualifiedMarketplaceToolExecutionIntentUnavailableError(
                "qualified marketplace tool execution intent read failed",
            ) from exc
        if document is None:
            return None
        return _decode_intent_record(document, execution_request_id=cleaned)


__all__ = [
    "DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository",
]
