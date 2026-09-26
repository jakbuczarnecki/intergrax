# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ConditionalDocumentStore-backed handoff→tenant context association (S24-GAP-02-P2)."""

from __future__ import annotations

from typing import Final

from pydantic import ValidationError

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContext,
    MarketplaceQualifiedToolStageContextAssociationConflictError,
    MarketplaceQualifiedToolStageContextAssociationIntegrityError,
    MarketplaceQualifiedToolStageContextAssociationUnavailableError,
    MarketplaceQualifiedToolStageContextAssociationWriteOutcome,
    MarketplaceQualifiedToolStageContextAssociationWriteResult,
    SCHEMA_MARKETPLACE_QUALIFIED_TOOL_STAGE_CONTEXT_V1,
)
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)

_DOCUMENT_STORE_PARTITION: Final = (
    "intergrax.marketplace_qualified_tool_stage_context_association.v1"
)
_PERSISTENCE_SCHEMA: Final = (
    "intergrax.marketplace_qualified_tool_stage_context_association.persistence.v1"
)
_PAYLOAD_FIELD: Final = "association"


def _document_row_key(handoff_id: str) -> str:
    return handoff_id


def _encode_association_record(
    association: MarketplaceQualifiedToolStageContext,
) -> DocumentRecord:
    return DocumentRecord(
        partition_key=_DOCUMENT_STORE_PARTITION,
        row_key=_document_row_key(association.handoff_id),
        data={
            "schema_version": _PERSISTENCE_SCHEMA,
            _PAYLOAD_FIELD: association.model_dump(mode="json"),
        },
    )


def _decode_association_record(
    document: DocumentRecord,
    *,
    handoff_id: str,
) -> MarketplaceQualifiedToolStageContext:
    data = dict(document.data)
    schema_version = data.get("schema_version")
    if schema_version != _PERSISTENCE_SCHEMA:
        raise MarketplaceQualifiedToolStageContextAssociationIntegrityError(
            "unsupported marketplace qualified tool stage context persistence schema",
        )
    payload = data.get(_PAYLOAD_FIELD)
    if not isinstance(payload, dict):
        raise MarketplaceQualifiedToolStageContextAssociationIntegrityError(
            "marketplace qualified tool stage context persistence payload is invalid",
        )
    if payload.get("schema_version") != SCHEMA_MARKETPLACE_QUALIFIED_TOOL_STAGE_CONTEXT_V1:
        raise MarketplaceQualifiedToolStageContextAssociationIntegrityError(
            "marketplace qualified tool stage context semantic schema mismatch",
        )
    try:
        association = MarketplaceQualifiedToolStageContext.model_validate(payload)
    except ValidationError as exc:
        raise MarketplaceQualifiedToolStageContextAssociationIntegrityError(
            "marketplace qualified tool stage context payload failed validation",
        ) from exc
    if association.handoff_id != handoff_id:
        raise MarketplaceQualifiedToolStageContextAssociationIntegrityError(
            "marketplace qualified tool stage context document key does not match payload",
        )
    if document.partition_key != _DOCUMENT_STORE_PARTITION:
        raise MarketplaceQualifiedToolStageContextAssociationIntegrityError(
            "marketplace qualified tool stage context partition mismatch",
        )
    if document.row_key != _document_row_key(handoff_id):
        raise MarketplaceQualifiedToolStageContextAssociationIntegrityError(
            "marketplace qualified tool stage context row key does not match handoff_id",
        )
    require_non_empty_text(association.tenant_id, label="tenant_id")
    require_non_empty_text(association.acquisition_request_id, label="acquisition_request_id")
    return association


class DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository:
    """Provider-neutral durable association over ``ConditionalDocumentStore``."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "marketplace qualified tool stage context association requires "
                "ConditionalDocumentStore",
            )
        self._document_store = document_store

    def record(
        self,
        association: MarketplaceQualifiedToolStageContext,
    ) -> MarketplaceQualifiedToolStageContextAssociationWriteResult:
        document = _encode_association_record(association)
        try:
            created = self._document_store.put_if_absent(document)
        except Exception as exc:
            raise MarketplaceQualifiedToolStageContextAssociationUnavailableError(
                "marketplace qualified tool stage context association write failed",
            ) from exc
        if created:
            return MarketplaceQualifiedToolStageContextAssociationWriteResult(
                outcome=MarketplaceQualifiedToolStageContextAssociationWriteOutcome.CREATED,
            )

        existing = self._load_existing(handoff_id=association.handoff_id)
        if existing == association:
            return MarketplaceQualifiedToolStageContextAssociationWriteResult(
                outcome=(
                    MarketplaceQualifiedToolStageContextAssociationWriteOutcome.ALREADY_RECORDED_IDENTICAL
                ),
            )
        raise MarketplaceQualifiedToolStageContextAssociationConflictError(
            "marketplace qualified tool stage context association identity conflict",
        )

    def get_by_handoff_id(
        self,
        handoff_id: str,
    ) -> MarketplaceQualifiedToolStageContext | None:
        cleaned = require_non_empty_text(handoff_id, label="handoff_id")
        return self._load_existing(handoff_id=cleaned)

    def _load_existing(
        self,
        *,
        handoff_id: str,
    ) -> MarketplaceQualifiedToolStageContext | None:
        row_key = _document_row_key(handoff_id)
        try:
            document = self._document_store.get(_DOCUMENT_STORE_PARTITION, row_key)
        except Exception as exc:
            raise MarketplaceQualifiedToolStageContextAssociationUnavailableError(
                "marketplace qualified tool stage context association read failed",
            ) from exc
        if document is None:
            return None
        return _decode_association_record(document, handoff_id=handoff_id)


__all__ = [
    "DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository",
]
