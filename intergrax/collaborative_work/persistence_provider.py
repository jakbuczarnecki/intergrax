# © Artur Czarnecki. All rights reserved.

"""Typed Collaborative Work persistence materialization from Integrations selection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.resolve_typed import resolve_relational_store


@runtime_checkable
class CollaborativeWorkPersistenceProvider(Protocol):
    """Domain contract: a resolved relational provider can materialize CW repositories."""

    def materialize_collaborative_work_repositories(self) -> CollaborativeWorkRepositories:
        """Construct the authoritative Collaborative Work repository bundle."""


def resolve_collaborative_work_repositories(
    profile: IntegrationProfile,
) -> CollaborativeWorkRepositories:
    """
    Resolve Collaborative Work repositories through Integrations provider selection.

    Selection order follows
    :func:`~intergrax.integrations.registry.resolve_typed.resolve_relational_store`.
    """
    relational = resolve_relational_store(profile)
    if not isinstance(relational, CollaborativeWorkPersistenceProvider):
        provider_name = type(relational).__name__
        raise IntegrationConfigurationError(
            "Selected relational store provider "
            f"({provider_name}) does not implement Collaborative Work persistence "
            "materialization."
        )
    return relational.materialize_collaborative_work_repositories()
