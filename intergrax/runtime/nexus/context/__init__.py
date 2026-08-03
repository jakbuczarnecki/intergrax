# © Artur Czarnecki. All rights reserved.

from typing import TYPE_CHECKING, Any

from intergrax.contracts.context_assembly import (
    ContextAssemblyMetadataKey,
    ContextSummaryTier,
    TaskContextAssemblyOptions,
    context_assembly_options_from_metadata,
)
from intergrax.runtime.nexus.context.context_models import (
    ContextProvenance,
    ContextSourceType,
    PriorOutputRecord,
)
from intergrax.runtime.nexus.context.metadata_keys import AgentContextMetadataKey

if TYPE_CHECKING:
    from intergrax.runtime.nexus.context.context_manager import (
        AgentContextBundle,
        ContextManager,
    )

_LAZY_CONTEXT_MANAGER_EXPORTS = frozenset(
    {
        "AgentContextBundle",
        "ContextManager",
    }
)

__all__ = [
    "AgentContextBundle",
    "AgentContextMetadataKey",
    "ContextAssemblyMetadataKey",
    "ContextManager",
    "ContextProvenance",
    "ContextSourceType",
    "ContextSummaryTier",
    "PriorOutputRecord",
    "TaskContextAssemblyOptions",
    "context_assembly_options_from_metadata",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_CONTEXT_MANAGER_EXPORTS:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    from intergrax.runtime.nexus.context.context_manager import (
        AgentContextBundle,
        ContextManager,
    )

    resolved = {
        "AgentContextBundle": AgentContextBundle,
        "ContextManager": ContextManager,
    }

    globals().update(resolved)
    return resolved[name]


def __dir__() -> list[str]:
    return sorted(
        set(globals()) | _LAZY_CONTEXT_MANAGER_EXPORTS
    )
