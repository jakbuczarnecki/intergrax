# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ConditionalDocumentStore-backed marketplace tool execution intent (S24-GAP-02-P3)."""

from __future__ import annotations

from typing import Final

from pydantic import ValidationError

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStageIntegrityError,
    MarketplaceQualifiedToolStageRepository,
    MarketplaceQualifiedToolStageUnavailableError,
)
from intergrax.contracts.tools.marketplace_tool_execution_intent import (
    MarketplaceToolExecutionIntent,
    SCHEMA_MARKETPLACE_TOOL_EXECUTION_INTENT_V2,
    UcaMarketplaceToolExecutionProvenance,
)
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
_PERSISTENCE_SCHEMA_V1: Final = (
    "intergrax.qualified_marketplace_tool_execution_intent.persistence.v1"
)
_PERSISTENCE_SCHEMA_V2: Final = (
    "intergrax.marketplace_tool_execution_intent.persistence.v2"
)
_PAYLOAD_FIELD: Final = "intent"


def _document_row_key(execution_request_id: str) -> str:
    return execution_request_id


def _encode_intent_record(intent: MarketplaceToolExecutionIntent) -> DocumentRecord:
    return DocumentRecord(
        partition_key=_DOCUMENT_STORE_PARTITION,
        row_key=_document_row_key(intent.execution_request_id),
        data={
            "schema_version": _PERSISTENCE_SCHEMA_V2,
            _PAYLOAD_FIELD: intent.model_dump(mode="json"),
        },
    )


def _project_v1_uca_intent_to_v2(
    legacy: QualifiedMarketplaceToolExecutionIntent,
    *,
    stage_repository: MarketplaceQualifiedToolStageRepository,
) -> MarketplaceToolExecutionIntent:
    try:
        stage = stage_repository.get(
            tenant_id=legacy.tenant_id,
            handoff_id=legacy.handoff_id,
        )
    except MarketplaceQualifiedToolStageUnavailableError as exc:
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "historical uca intent stage lookup unavailable",
        ) from exc
    except MarketplaceQualifiedToolStageIntegrityError as exc:
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "historical uca intent stage corrupt",
        ) from exc
    if stage is None:
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "historical uca intent missing staged release for capability identity",
        )
    capability_identity = CapabilityIdentityKey.from_discovery_identity(
        stage.selected_release.discovery,
    )
    return MarketplaceToolExecutionIntent(
        execution_request_id=legacy.execution_request_id,
        binding_operation_id=legacy.binding_operation_id,
        tenant_id=legacy.tenant_id,
        task_id=legacy.task_id,
        worker_need_id=legacy.worker_need_id,
        subject_reference=legacy.qualified_subject_reference,
        capability_identity=capability_identity,
        selected_operation=legacy.selected_operation,
        provenance=UcaMarketplaceToolExecutionProvenance(
            handoff_id=legacy.handoff_id,
            resume_operation_id=legacy.resume_operation_id,
            uca_qualified_subject_reference=legacy.qualified_subject_reference,
        ),
    )


def _decode_intent_record(
    document: DocumentRecord,
    *,
    execution_request_id: str,
    stage_repository: MarketplaceQualifiedToolStageRepository | None,
) -> MarketplaceToolExecutionIntent:
    data = dict(document.data)
    persistence_schema = data.get("schema_version")
    if persistence_schema not in {
        _PERSISTENCE_SCHEMA_V1,
        _PERSISTENCE_SCHEMA_V2,
    }:
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "unsupported marketplace tool execution intent persistence schema",
        )
    payload = data.get(_PAYLOAD_FIELD)
    if not isinstance(payload, dict):
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "marketplace tool execution intent persistence payload is invalid",
        )
    semantic_schema = payload.get("schema_version")
    if document.partition_key != _DOCUMENT_STORE_PARTITION:
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "marketplace tool execution intent partition mismatch",
        )
    if document.row_key != _document_row_key(execution_request_id):
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "marketplace tool execution intent row key mismatch",
        )
    if semantic_schema == SCHEMA_MARKETPLACE_TOOL_EXECUTION_INTENT_V2:
        try:
            intent = MarketplaceToolExecutionIntent.model_validate(payload)
        except ValidationError as exc:
            raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
                "marketplace tool execution intent v2 payload failed validation",
            ) from exc
    elif semantic_schema == SCHEMA_QUALIFIED_MARKETPLACE_TOOL_EXECUTION_INTENT_V1:
        try:
            legacy = QualifiedMarketplaceToolExecutionIntent.model_validate(payload)
        except ValidationError as exc:
            raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
                "marketplace tool execution intent v1 payload failed validation",
            ) from exc
        if stage_repository is None:
            raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
                "historical uca intent requires stage repository for v2 projection",
            )
        intent = _project_v1_uca_intent_to_v2(
            legacy,
            stage_repository=stage_repository,
        )
    else:
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "marketplace tool execution intent semantic schema mismatch",
        )
    if intent.execution_request_id != execution_request_id:
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError(
            "marketplace tool execution intent identity mismatch",
        )
    return intent


class DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository:
    """Durable intent keyed by execution_request_id — no overwrite."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        stage_repository: MarketplaceQualifiedToolStageRepository | None = None,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "qualified marketplace tool execution intent requires ConditionalDocumentStore",
            )
        self._document_store = document_store
        self._stage_repository = stage_repository

    def record(
        self,
        intent: MarketplaceToolExecutionIntent,
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
    ) -> MarketplaceToolExecutionIntent | None:
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
        return _decode_intent_record(
            document,
            execution_request_id=cleaned,
            stage_repository=self._stage_repository,
        )


__all__ = [
    "DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository",
]
