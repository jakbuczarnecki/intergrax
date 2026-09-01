# © Artur Czarnecki. All rights reserved.

"""Typed Collaborative Work catalog materialization factory contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories


@runtime_checkable
class CollaborativeWorkPersistenceFactory(Protocol):
    """Explicit catalog factory capability for Collaborative Work persistence."""

    def materialize_collaborative_work_repositories(
        self,
    ) -> CollaborativeWorkRepositories:
        """Materialize the authoritative Collaborative Work repository bundle."""
