# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed Autonomous Work catalog materialization factory contracts (AW-2C)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from intergrax.autonomous_work.persistence import AutonomousWorkRepositories


@runtime_checkable
class AutonomousWorkPersistenceFactory(Protocol):
    """Configured materializer capability: ready to produce Autonomous Work repositories."""

    def materialize_autonomous_work_repositories(self) -> AutonomousWorkRepositories:
        """Materialize the authoritative Autonomous Work repository bundle."""


@runtime_checkable
class AutonomousWorkMaterializationBinder(Protocol):
    """Unbound catalog factory capability: bind Integrations options to a materializer."""

    def bind_autonomous_work_materialization(
        self,
        options: Mapping[str, object],
    ) -> AutonomousWorkPersistenceFactory:
        """Return a provider-configured materializer for Autonomous Work persistence."""
