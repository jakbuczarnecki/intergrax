# © Artur Czarnecki. All rights reserved.

"""Reference production V1 process-level composition root (AGENT-CONSOLIDATION-3-ARCH).

``ProductionProcessComposition`` is the canonical owner of one
``ProductionAgentPlatformRuntime`` per production process. AP lifecycle
(prepare / project / activate) and application host serving MUST receive the
same ``agent_platform_runtime.stores`` bundle from this root.

Application ``main.py`` modules and product factories are consumers or wiring
glue — they are **not** lifecycle-store owners.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.production_agent_platform_runtime import (
    ProductionAgentPlatformRuntime,
    build_production_agent_platform_runtime,
)


@dataclass(frozen=True, slots=True)
class ProductionProcessComposition:
    """Process-level composition root for reference single-process production V1."""

    agent_platform_runtime: ProductionAgentPlatformRuntime


def create_reference_production_process_composition() -> ProductionProcessComposition:
    """Create one reference production process composition with fresh process-local stores."""
    return ProductionProcessComposition(
        agent_platform_runtime=build_production_agent_platform_runtime(),
    )


__all__ = [
    "ProductionProcessComposition",
    "create_reference_production_process_composition",
]
