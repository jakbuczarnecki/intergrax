# © Artur Czarnecki. All rights reserved.

"""Capability factory bindings for qualification (MEM-ENT-13)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.entity_temporal_memory import EntityTemporalMemoryStore
from intergrax.memory.contracts.long_horizon_memory import LongHorizonMemoryStore
from intergrax.memory.contracts.procedural_memory import ProcedureMemoryStore
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
from intergrax.memory.provider_qualification.factory import MemoryProviderInstanceFactory
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.runtime.nexus.session.session_storage import SessionStorage


@dataclass(frozen=True, slots=True)
class MemoryProviderCapabilityFactories:
    user_profile_store: MemoryProviderInstanceFactory[UserProfileStore] | None = None
    session_storage: MemoryProviderInstanceFactory[SessionStorage] | None = None
    session_turn_index_store: MemoryProviderInstanceFactory[SessionTurnIndexStore] | None = None
    entity_temporal_memory_store: (
        MemoryProviderInstanceFactory[EntityTemporalMemoryStore] | None
    ) = None
    procedure_memory_store: MemoryProviderInstanceFactory[ProcedureMemoryStore] | None = None
    long_horizon_memory_store: MemoryProviderInstanceFactory[LongHorizonMemoryStore] | None = None
