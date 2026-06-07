# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.records.contracts import (
    RecordsDeleteInput,
    RecordsDeleteOutput,
    RecordsGetInput,
    RecordsGetOutput,
    RecordsPutInput,
    RecordsPutOutput,
    RecordsQueryInput,
    RecordsQueryOutput,
)
from intergrax.tools.providers.records.service import records_delete, records_get, records_put, records_query


class RecordsGetHandler(ServiceToolHandler[RecordsGetInput, RecordsGetOutput]):
    _service = records_get


class RecordsPutHandler(ServiceToolHandler[RecordsPutInput, RecordsPutOutput]):
    _service = records_put


class RecordsDeleteHandler(ServiceToolHandler[RecordsDeleteInput, RecordsDeleteOutput]):
    _service = records_delete


class RecordsQueryHandler(ServiceToolHandler[RecordsQueryInput, RecordsQueryOutput]):
    _service = records_query
