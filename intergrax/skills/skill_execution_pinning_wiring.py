# © Artur Czarnecki. All rights reserved.

"""Composition wiring for durable skill execution pinning (SKILL-X Batch C)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.skills.execution_binding import (
    InMemorySkillExecutionPinningStore,
    SkillExecutionPinningStore,
)
from intergrax.skills.execution_pinning_durability import (
    validate_skill_execution_pinning_durability_for_composition,
)
from intergrax.skills.execution_pinning_persistence import wire_skill_execution_pinning_store


@dataclass(frozen=True, slots=True)
class SkillExecutionPinningWiring:
    pinning_store: SkillExecutionPinningStore


def resolve_skill_execution_pinning_wiring(
    *,
    production_mode: bool,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
    pinning_store: SkillExecutionPinningStore | None = None,
) -> SkillExecutionPinningWiring:
    if pinning_store is not None:
        resolved = pinning_store
    elif kv_store is not None or document_store is not None:
        resolved = wire_skill_execution_pinning_store(
            kv_store=kv_store,
            document_store=document_store,
        )
    else:
        resolved = InMemorySkillExecutionPinningStore()

    validate_skill_execution_pinning_durability_for_composition(
        production_mode=production_mode,
        pinning_store=resolved,
    )
    return SkillExecutionPinningWiring(pinning_store=resolved)


__all__ = [
    "SkillExecutionPinningWiring",
    "resolve_skill_execution_pinning_wiring",
]
