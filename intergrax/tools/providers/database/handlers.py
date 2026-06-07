# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.database.contracts import (
    DatabaseDescribeSchemaInput,
    DatabaseDescribeSchemaOutput,
    DatabaseExecuteInput,
    DatabaseExecuteOutput,
    DatabaseQueryInput,
    DatabaseQueryOutput,
)
from intergrax.tools.providers.database.service import database_describe_schema, database_execute, database_query


class DatabaseQueryHandler(ServiceToolHandler[DatabaseQueryInput, DatabaseQueryOutput]):
    _service = database_query


class DatabaseExecuteHandler(ServiceToolHandler[DatabaseExecuteInput, DatabaseExecuteOutput]):
    _service = database_execute


class DatabaseDescribeSchemaHandler(ServiceToolHandler[DatabaseDescribeSchemaInput, DatabaseDescribeSchemaOutput]):
    _service = database_describe_schema
