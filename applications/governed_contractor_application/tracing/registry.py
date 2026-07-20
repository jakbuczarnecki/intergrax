# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.runtime.observability.extension_sdk import PayloadSchemaRegistry
from applications.governed_contractor_application.tracing.example_diag import HostLifecycleDiagV1


def register_tracing_schemas() -> None:
    """Register application diagnostic schemas with the Harness observability spine."""
    PayloadSchemaRegistry.register_application_diagnostic(
        HostLifecycleDiagV1,
        app_slug="governed_contractor",
    )
