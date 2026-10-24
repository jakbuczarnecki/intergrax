# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Declared transport cancellation / termination capability (observation only — not authority)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProviderCancellationSupport(str, Enum):
    """What the adapter can physically attempt when the invoker requests cancel."""

    NONE = "none"
    COOPERATIVE_ONLY = "cooperative_only"
    CONNECTION_DROP = "connection_drop"
    QUERY_CANCEL = "query_cancel"


@dataclass(frozen=True, slots=True)
class ProviderTransportCapabilities:
    provider_family: str
    cancellation: ProviderCancellationSupport
    supports_result_byte_ceiling: bool
    supports_row_ceiling: bool
    notes: str = ""


SQL_DATABASE_CAPABILITIES = ProviderTransportCapabilities(
    provider_family="sql_database",
    cancellation=ProviderCancellationSupport.QUERY_CANCEL,
    supports_result_byte_ceiling=True,
    supports_row_ceiling=True,
    notes="Driver-level cancel when port exposes it; tool layer enforces row/byte ceilings.",
)

HTTP_REST_CAPABILITIES = ProviderTransportCapabilities(
    provider_family="http_rest",
    cancellation=ProviderCancellationSupport.CONNECTION_DROP,
    supports_result_byte_ceiling=True,
    supports_row_ceiling=False,
    notes="Attempt timeout is invoker-owned; adapter truncates response bytes.",
)

MCP_CATALOG_CAPABILITIES = ProviderTransportCapabilities(
    provider_family="mcp_catalog",
    cancellation=ProviderCancellationSupport.COOPERATIVE_ONLY,
    supports_result_byte_ceiling=True,
    supports_row_ceiling=False,
    notes="MCP hosts re-enter ToolRuntime; no second lifecycle manager.",
)

BROWSER_AUTOMATION_CAPABILITIES = ProviderTransportCapabilities(
    provider_family="browser_automation",
    cancellation=ProviderCancellationSupport.CONNECTION_DROP,
    supports_result_byte_ceiling=True,
    supports_row_ceiling=False,
    notes="Navigation timeout is integration-owned; text output is bounded at tool layer.",
)
