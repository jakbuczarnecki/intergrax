# © Artur Czarnecki. All rights reserved.

"""Typed Collaborative Work catalog materialization factory contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories


@runtime_checkable
class CollaborativeWorkPersistenceFactory(Protocol):
    """Configured materializer capability: ready to produce Collaborative Work repositories."""

    def materialize_collaborative_work_repositories(
        self,
    ) -> CollaborativeWorkRepositories:
        """Materialize the authoritative Collaborative Work repository bundle."""


@runtime_checkable
class CollaborativeWorkMaterializationBinder(Protocol):
    """Unbound catalog factory capability: bind Integrations options to a materializer."""

    def bind_collaborative_work_materialization(
        self,
        options: Mapping[str, object],
    ) -> CollaborativeWorkPersistenceFactory:
        """Return a provider-configured materializer for Collaborative Work persistence."""
