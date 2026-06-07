# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.records.contracts import (
    RecordsDeleteInput,
    RecordsDeleteOutput,
    RecordsDescribeCollectionInput,
    RecordsDescribeCollectionOutput,
    RecordsCountInput,
    RecordsCountOutput,
    RecordsGetInput,
    RecordsGetOutput,
    RecordsPutInput,
    RecordsPutOutput,
    RecordsQueryInput,
    RecordsQueryOutput,
)
from intergrax.tools.providers.records.handlers import (
    RecordsDeleteHandler,
    RecordsDescribeCollectionHandler,
    RecordsCountHandler,
    RecordsGetHandler,
    RecordsPutHandler,
    RecordsQueryHandler,
)
from intergrax.tools.providers.records.service import (
    RECORDS_DELETE_TOOL_ID,
    RECORDS_DESCRIBE_COLLECTION_TOOL_ID,
    RECORDS_COUNT_TOOL_ID,
    RECORDS_GET_TOOL_ID,
    RECORDS_PUT_TOOL_ID,
    RECORDS_QUERY_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

RECORDS_BUNDLE_ID = "records"
RECORDS_TOOL_IDS: tuple[str, ...] = (
    RECORDS_GET_TOOL_ID,
    RECORDS_PUT_TOOL_ID,
    RECORDS_DELETE_TOOL_ID,
    RECORDS_QUERY_TOOL_ID,
    RECORDS_DESCRIBE_COLLECTION_TOOL_ID,
    RECORDS_COUNT_TOOL_ID,
)


def register_records_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=RECORDS_GET_TOOL_ID,
            name=RECORDS_GET_TOOL_ID,
            description="Fetch a JSON document from the configured document store by partition and row key.",
            description_short="Get document record.",
            input_schema=RecordsGetInput,
            output_schema=RecordsGetOutput,
            error_mapping={},
            side_effects=False,
            category="records",
            risk_level=ToolRiskLevel.LOW,
            tags=("records", "document_store", "json"),
        ),
        RecordsGetHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=RECORDS_PUT_TOOL_ID,
            name=RECORDS_PUT_TOOL_ID,
            description="Insert or upsert a JSON document in the configured document store.",
            description_short="Put document record.",
            input_schema=RecordsPutInput,
            output_schema=RecordsPutOutput,
            error_mapping={},
            side_effects=True,
            category="records",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("records", "document_store", "json"),
        ),
        RecordsPutHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=RECORDS_DELETE_TOOL_ID,
            name=RECORDS_DELETE_TOOL_ID,
            description="Delete a JSON document from the configured document store.",
            description_short="Delete document record.",
            input_schema=RecordsDeleteInput,
            output_schema=RecordsDeleteOutput,
            error_mapping={},
            side_effects=True,
            category="records",
            risk_level=ToolRiskLevel.HIGH,
            tags=("records", "document_store", "json"),
        ),
        RecordsDeleteHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=RECORDS_QUERY_TOOL_ID,
            name=RECORDS_QUERY_TOOL_ID,
            description="List JSON documents within a partition, optionally filtered by row-key prefix.",
            description_short="Query document records.",
            input_schema=RecordsQueryInput,
            output_schema=RecordsQueryOutput,
            error_mapping={},
            side_effects=False,
            category="records",
            risk_level=ToolRiskLevel.LOW,
            tags=("records", "document_store", "json"),
        ),
        RecordsQueryHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=RECORDS_DESCRIBE_COLLECTION_TOOL_ID,
            name=RECORDS_DESCRIBE_COLLECTION_TOOL_ID,
            description="Describe a document-store partition: sample row keys and JSON field names.",
            description_short="Describe records partition.",
            input_schema=RecordsDescribeCollectionInput,
            output_schema=RecordsDescribeCollectionOutput,
            error_mapping={},
            side_effects=False,
            category="records",
            risk_level=ToolRiskLevel.LOW,
            tags=("records", "document_store", "schema", "read_only"),
        ),
        RecordsDescribeCollectionHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=RECORDS_COUNT_TOOL_ID,
            name=RECORDS_COUNT_TOOL_ID,
            description="Return document count for a document-store partition without fetching rows.",
            description_short="Count records in partition.",
            input_schema=RecordsCountInput,
            output_schema=RecordsCountOutput,
            error_mapping={},
            side_effects=False,
            category="records",
            risk_level=ToolRiskLevel.LOW,
            tags=("records", "document_store", "metadata"),
        ),
        RecordsCountHandler(ctx),
    )
