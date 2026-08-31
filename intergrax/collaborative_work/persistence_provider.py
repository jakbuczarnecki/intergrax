# © Artur Czarnecki. All rights reserved.

"""Typed Collaborative Work persistence materialization from Integrations selection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.collaborative_work.materialization_factory import (
    CollaborativeWorkPersistenceFactory,
    binding_from_profile_options,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.integrations._shared.config import merge_config
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.registry.catalog import get_entry
from intergrax.integrations.registry.factory import resolve_slug
from intergrax.integrations.registry.profile import IntegrationProfile


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

    Provider selection follows the same slug/options resolution as
    :func:`~intergrax.integrations.registry.factory.resolve`, but durable
    Collaborative Work materialization opens exactly one provider lifecycle.
    """
    category = IntegrationCategory.RELATIONAL_STORE
    instance = profile.instance_for_category(category)
    if instance is not None:
        if isinstance(instance, CollaborativeWorkPersistenceProvider):
            return instance.materialize_collaborative_work_repositories()
        provider_name = type(instance).__name__
        raise IntegrationConfigurationError(
            "Selected relational store provider "
            f"({provider_name}) does not implement Collaborative Work persistence "
            "materialization."
        )

    slug = resolve_slug(category, profile=profile)
    merged = merge_config(profile.options_for_slug(slug), None)
    entry = get_entry(slug)
    factory = entry.factory
    if not isinstance(factory, CollaborativeWorkPersistenceFactory):
        raise IntegrationConfigurationError(
            "Selected relational store provider "
            f"({slug}) does not implement Collaborative Work persistence "
            "materialization."
        )

    binding = binding_from_profile_options(merged)
    return factory.materialize_collaborative_work_repositories(binding)
