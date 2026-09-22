# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compatibility re-exports — canonical contract lives in intergrax.contracts."""

from __future__ import annotations

from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
    ExecutionBoundCatalogToolInvoker,
)

__all__ = [
    "ExecutionBoundCatalogToolInvokeRequest",
    "ExecutionBoundCatalogToolInvoker",
]
