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
    supports_physical_result_byte_ceiling: bool
    supports_physical_row_ceiling: bool
    supports_output_result_byte_ceiling: bool = True
    supports_output_row_ceiling: bool = True
    notes: str = ""

    @property
    def supports_result_byte_ceiling(self) -> bool:
        return self.supports_physical_result_byte_ceiling

    @property
    def supports_row_ceiling(self) -> bool:
        return self.supports_physical_row_ceiling


SQL_DATABASE_CAPABILITIES = ProviderTransportCapabilities(
    provider_family="sql_database",
    cancellation=ProviderCancellationSupport.QUERY_CANCEL,
    supports_physical_result_byte_ceiling=True,
    supports_physical_row_ceiling=True,
    notes=(
        "Physical row/byte ceilings when RelationalStoreBoundedQueryPort is configured; "
        "otherwise output-only bounds (physical_bounds_enforced=false on result)."
    ),
)

HTTP_REST_CAPABILITIES = ProviderTransportCapabilities(
    provider_family="http_rest",
    cancellation=ProviderCancellationSupport.CONNECTION_DROP,
    supports_physical_result_byte_ceiling=True,
    supports_physical_row_ceiling=False,
    notes=(
        "AllowlistHttpClient enforces max_response_bytes at stream read; "
        "tool layer may apply output sanitation when transport did not enforce."
    ),
)

MCP_CATALOG_CAPABILITIES = ProviderTransportCapabilities(
    provider_family="mcp_catalog",
    cancellation=ProviderCancellationSupport.COOPERATIVE_ONLY,
    supports_physical_result_byte_ceiling=False,
    supports_physical_row_ceiling=False,
    notes=(
        "Logical invocation/catalog bounds via bound_mcp_structured_result; "
        "physical MCP transport buffering remains L3 qualification."
    ),
)

BROWSER_AUTOMATION_CAPABILITIES = ProviderTransportCapabilities(
    provider_family="browser_automation",
    cancellation=ProviderCancellationSupport.CONNECTION_DROP,
    supports_physical_result_byte_ceiling=False,
    supports_physical_row_ceiling=False,
    notes=(
        "navigation_timeout_s enforced only when navigation_timeout_integration_enforced is True; "
        "max_text_chars is output-only (DOM/page memory not hard-bounded)."
    ),
)
