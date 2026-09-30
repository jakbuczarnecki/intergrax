# © Artur Czarnecki. All rights reserved.

"""EE-owned session materialization for agent authoring helpers (no Nexus surface outside EE)."""

from __future__ import annotations

from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager


def build_in_memory_session_manager() -> SessionManager:
    return SessionManager(storage=InMemorySessionStorage())


__all__ = ["build_in_memory_session_manager"]
