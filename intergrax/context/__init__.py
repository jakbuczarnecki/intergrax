# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 Context Engineering contracts and plugin catalog (Phase CE-1)."""

from intergrax.context.contracts import (
    AssembledContext,
    BudgetAllocationResult,
    ContextAssemblyProvenance,
    ContextAssemblyRequest,
    ContextAssemblyScope,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
    content_hash_for_text,
)
from intergrax.context.plugin import ContextPlugin, register_context_plugin
from intergrax.context.protocols import (
    ContextBudgetAllocator,
    ContextEngine,
    ContextFormatter,
    ContextRanker,
    ContextSourceProvider,
    ContextValidator,
)
from intergrax.context.registry import (
    ContextPluginRegistry,
    clear_context_plugin_catalog,
    get_context_plugin,
    iter_context_plugins,
    list_context_plugin_ids,
)

__all__ = [
    "AssembledContext",
    "BudgetAllocationResult",
    "ContextAssemblyProvenance",
    "ContextAssemblyRequest",
    "ContextAssemblyScope",
    "ContextBudgetAllocator",
    "ContextBudgetSnapshot",
    "ContextDecisionSnapshot",
    "ContextEngine",
    "ContextFormatter",
    "ContextFragment",
    "ContextFragmentSource",
    "ContextPlugin",
    "ContextPluginRegistry",
    "ContextProviderContext",
    "ContextRanker",
    "ContextSourceProvider",
    "ContextValidator",
    "clear_context_plugin_catalog",
    "content_hash_for_text",
    "get_context_plugin",
    "iter_context_plugins",
    "list_context_plugin_ids",
    "register_context_plugin",
]
