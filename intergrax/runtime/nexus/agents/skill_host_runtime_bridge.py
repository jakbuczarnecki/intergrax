# © Artur Czarnecki. All rights reserved.

"""Apply host skill catalog wiring to Nexus ``RuntimeContext`` (internal composition)."""

from __future__ import annotations

from typing import Any

from intergrax.agents.persistence.skill_host_wiring import resolve_skill_host_wiring_from_metadata
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext


def apply_host_skill_wiring_to_runtime_context(
    runtime_context: object,
    request_metadata: dict[str, Any],
) -> None:
    if not isinstance(runtime_context, RuntimeContext):
        return
    wiring = resolve_skill_host_wiring_from_metadata(request_metadata)
    if wiring is None:
        return
    runtime_context.config.skill_profile = wiring.skill_profile
    runtime_context.config.skill_registry = wiring.skill_registry
    if wiring.skill_pinning_store is not None:
        runtime_context.config.skill_pinning_store = wiring.skill_pinning_store
