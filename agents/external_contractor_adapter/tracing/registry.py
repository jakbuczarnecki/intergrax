# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.runtime.observability.extension_sdk import PayloadSchemaRegistry
from external_contractor_adapter.tracing.example_diag import CustomCheckDiagV1


def register_tracing_schemas() -> None:
    """Register agent diagnostic schemas with the Harness observability spine."""
    PayloadSchemaRegistry.register_agent_diagnostic(
        CustomCheckDiagV1,
        agent_slug="external_contractor_adapter",
    )
