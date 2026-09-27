# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ConditionalDocumentStore-backed Tool marketplace qualified staging (S24-GAP-02-P1)."""

from __future__ import annotations

from typing import Final

from pydantic import ValidationError

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
    MarketplaceQualifiedToolStageConflictError,
    MarketplaceQualifiedToolStageIntegrityError,
    MarketplaceQualifiedToolStageUnavailableError,
    MarketplaceQualifiedToolStageWriteOutcome,
    MarketplaceQualifiedToolStageWriteResult,
    SCHEMA_MARKETPLACE_QUALIFIED_TOOL_STAGE_V1,
)
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)

_DOCUMENT_STORE_PARTITION_PREFIX: Final = (
    "intergrax.marketplace_qualified_tool_stage.v1"
)
_PERSISTENCE_SCHEMA: Final = (
    "intergrax.marketplace_qualified_tool_stage.persistence.v1"
)
_PAYLOAD_FIELD: Final = "stage"


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def _document_row_key(handoff_id: str) -> str:
    return handoff_id


def _encode_stage_record(stage: MarketplaceQualifiedToolStage) -> DocumentRecord:
    return DocumentRecord(
        partition_key=_document_partition(stage.tenant_id),
        row_key=_document_row_key(stage.handoff_id),
        data={
            "schema_version": _PERSISTENCE_SCHEMA,
            _PAYLOAD_FIELD: stage.model_dump(mode="json"),
        },
    )


def _decode_stage_record(
    document: DocumentRecord,
    *,
    tenant_id: str,
    handoff_id: str,
) -> MarketplaceQualifiedToolStage:
    data = dict(document.data)
    schema_version = data.get("schema_version")
    if schema_version != _PERSISTENCE_SCHEMA:
        raise MarketplaceQualifiedToolStageIntegrityError(
            "unsupported marketplace qualified tool stage persistence schema",
        )
    payload = data.get(_PAYLOAD_FIELD)
    if not isinstance(payload, dict):
        raise MarketplaceQualifiedToolStageIntegrityError(
            "marketplace qualified tool stage persistence payload is invalid",
        )
    if payload.get("schema_version") != SCHEMA_MARKETPLACE_QUALIFIED_TOOL_STAGE_V1:
        raise MarketplaceQualifiedToolStageIntegrityError(
            "marketplace qualified tool stage semantic schema mismatch",
        )
    try:
        stage = MarketplaceQualifiedToolStage.model_validate(payload)
    except ValidationError as exc:
        raise MarketplaceQualifiedToolStageIntegrityError(
            "marketplace qualified tool stage payload failed validation",
        ) from exc
    if stage.tenant_id != tenant_id or stage.handoff_id != handoff_id:
        raise MarketplaceQualifiedToolStageIntegrityError(
            "marketplace qualified tool stage document keys do not match payload",
        )
    if document.partition_key != _document_partition(tenant_id):
        raise MarketplaceQualifiedToolStageIntegrityError(
            "marketplace qualified tool stage partition does not match tenant",
        )
    if document.row_key != _document_row_key(handoff_id):
        raise MarketplaceQualifiedToolStageIntegrityError(
            "marketplace qualified tool stage row key does not match handoff_id",
        )
    return stage


class DocumentStoreMarketplaceQualifiedToolStageRepository:
    """Provider-neutral durable staging over ``ConditionalDocumentStore``."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "marketplace qualified tool staging requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    def stage(
        self,
        record: MarketplaceQualifiedToolStage,
    ) -> MarketplaceQualifiedToolStageWriteResult:
        document = _encode_stage_record(record)
        try:
            created = self._document_store.put_if_absent(document)
        except Exception as exc:
            raise MarketplaceQualifiedToolStageUnavailableError(
                "marketplace qualified tool stage write failed",
            ) from exc
        if created:
            return MarketplaceQualifiedToolStageWriteResult(
                outcome=MarketplaceQualifiedToolStageWriteOutcome.CREATED,
            )

        existing = self._load_existing(
            tenant_id=record.tenant_id,
            handoff_id=record.handoff_id,
        )
        if existing == record:
            return MarketplaceQualifiedToolStageWriteResult(
                outcome=MarketplaceQualifiedToolStageWriteOutcome.ALREADY_STAGED_IDENTICAL,
            )
        raise MarketplaceQualifiedToolStageConflictError(
            "marketplace qualified tool stage identity conflict",
        )

    def get(
        self,
        *,
        tenant_id: str,
        handoff_id: str,
    ) -> MarketplaceQualifiedToolStage | None:
        cleaned_tenant = require_non_empty_text(tenant_id, label="tenant_id")
        cleaned_handoff = require_non_empty_text(handoff_id, label="handoff_id")
        return self._load_existing(
            tenant_id=cleaned_tenant,
            handoff_id=cleaned_handoff,
        )

    def _load_existing(
        self,
        *,
        tenant_id: str,
        handoff_id: str,
    ) -> MarketplaceQualifiedToolStage | None:
        partition_key = _document_partition(tenant_id)
        row_key = _document_row_key(handoff_id)
        try:
            document = self._document_store.get(partition_key, row_key)
        except Exception as exc:
            raise MarketplaceQualifiedToolStageUnavailableError(
                "marketplace qualified tool stage read failed",
            ) from exc
        if document is None:
            return None
        return _decode_stage_record(
            document,
            tenant_id=tenant_id,
            handoff_id=handoff_id,
        )


__all__ = [
    "DocumentStoreMarketplaceQualifiedToolStageRepository",
]
