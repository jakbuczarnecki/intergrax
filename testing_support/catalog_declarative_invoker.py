# © Artur Czarnecki. All rights reserved.

"""Test/acceptance helpers for catalog declarative invoker (not production host wiring)."""

from __future__ import annotations

from intergrax.agents.persistence.catalog_declarative_invoker import CatalogDeclarativeToolInvoker
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.registry import ToolRegistry


def build_catalog_declarative_invoker_from_registry(
    registry: ToolRegistry,
) -> CatalogDeclarativeToolInvoker:
    """Ungoverned RuntimeToolInvoker for tests only — use host wiring in production."""
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    return CatalogDeclarativeToolInvoker(tool_invoker=invoker)
