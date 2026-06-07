# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentStore
from intergrax.tools.providers.records.contracts import (
    RecordsDeleteInput,
    RecordsDeleteOutput,
    RecordsDocumentOutput,
    RecordsGetInput,
    RecordsGetOutput,
    RecordsPutInput,
    RecordsPutOutput,
    RecordsQueryInput,
    RecordsQueryOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

RECORDS_GET_TOOL_ID = "records.get"
RECORDS_PUT_TOOL_ID = "records.put"
RECORDS_DELETE_TOOL_ID = "records.delete"
RECORDS_QUERY_TOOL_ID = "records.query"


def _require_document_store(ctx: ToolWiringContext) -> DocumentStore:
    store = ctx.document_store
    if store is None:
        raise RuntimeError("document_store_not_configured")
    return store


def _to_document_output(record: DocumentRecord) -> RecordsDocumentOutput:
    return RecordsDocumentOutput(
        partition_key=record.partition_key,
        row_key=record.row_key,
        data=dict(record.data),
        ttl_seconds=record.ttl_seconds,
    )


def records_get(ctx: ToolWiringContext, params: RecordsGetInput) -> RecordsGetOutput:
    record = _require_document_store(ctx).get(params.partition_key.strip(), params.row_key.strip())
    if record is None:
        return RecordsGetOutput(found=False)
    return RecordsGetOutput(found=True, document=_to_document_output(record))


def records_put(ctx: ToolWiringContext, params: RecordsPutInput) -> RecordsPutOutput:
    _require_document_store(ctx).put(
        DocumentRecord(
            partition_key=params.partition_key.strip(),
            row_key=params.row_key.strip(),
            data=dict(params.data),
            ttl_seconds=params.ttl_seconds,
        )
    )
    return RecordsPutOutput(
        stored=True,
        partition_key=params.partition_key.strip(),
        row_key=params.row_key.strip(),
    )


def records_delete(ctx: ToolWiringContext, params: RecordsDeleteInput) -> RecordsDeleteOutput:
    _require_document_store(ctx).delete(params.partition_key.strip(), params.row_key.strip())
    return RecordsDeleteOutput(
        deleted=True,
        partition_key=params.partition_key.strip(),
        row_key=params.row_key.strip(),
    )


def records_query(ctx: ToolWiringContext, params: RecordsQueryInput) -> RecordsQueryOutput:
    result = _require_document_store(ctx).query(
        params.partition_key.strip(),
        limit=params.limit,
        row_key_prefix=params.row_key_prefix,
    )
    documents = [_to_document_output(item) for item in result.documents]
    return RecordsQueryOutput(documents=documents, total=result.total or len(documents))
