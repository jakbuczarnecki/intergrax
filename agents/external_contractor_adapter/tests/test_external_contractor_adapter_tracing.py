# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from external_contractor_adapter.tracing.example_diag import CustomCheckDiagV1
from external_contractor_adapter.tracing.registry import register_tracing_schemas
from intergrax.runtime.observability.extension_sdk import (
    get_registered_diagnostic_payload,
    list_registered_diagnostic_schema_ids,
)

pytestmark = pytest.mark.gate


def test_agent_tracing_schema_registers() -> None:
    register_tracing_schemas()
    schema_id = CustomCheckDiagV1.schema_id()
    assert schema_id in list_registered_diagnostic_schema_ids()
    assert get_registered_diagnostic_payload(schema_id) is CustomCheckDiagV1
