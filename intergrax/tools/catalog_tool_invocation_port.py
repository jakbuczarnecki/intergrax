# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Re-export canonical execution-bound catalog tool contract (Tools-owned ABI surface)."""

from __future__ import annotations

from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
    ExecutionBoundCatalogToolInvoker,
)

CatalogToolInvocationPort = ExecutionBoundCatalogToolInvoker

__all__ = [
    "CatalogToolInvocationPort",
    "ExecutionBoundCatalogToolInvokeRequest",
    "ExecutionBoundCatalogToolInvoker",
]
