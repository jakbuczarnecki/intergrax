# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed Autonomous Work persistence materialization from Integrations selection (AW-2C)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.autonomous_work.materialization_factory import (
    AutonomousWorkMaterializationBinder,
    AutonomousWorkPersistenceFactory,
)
from intergrax.autonomous_work.persistence import AutonomousWorkRepositories
from intergrax.integrations._shared.config import merge_config
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.registry.catalog import get_entry
from intergrax.integrations.registry.factory import resolve_slug
from intergrax.integrations.registry.profile import IntegrationProfile


@runtime_checkable
class AutonomousWorkPersistenceProvider(Protocol):
    """Domain contract: a resolved relational provider can materialize AW repositories."""

    def materialize_autonomous_work_repositories(self) -> AutonomousWorkRepositories:
        """Construct the authoritative Autonomous Work repository bundle."""


def resolve_autonomous_work_repositories(
    profile: IntegrationProfile,
) -> AutonomousWorkRepositories:
    """Resolve Autonomous Work repositories through Integrations provider selection."""
    category = IntegrationCategory.RELATIONAL_STORE
    instance = profile.instance_for_category(category)
    if instance is not None:
        if isinstance(instance, AutonomousWorkPersistenceProvider):
            return instance.materialize_autonomous_work_repositories()
        provider_name = type(instance).__name__
        raise IntegrationConfigurationError(
            "Selected relational store provider "
            f"({provider_name}) does not implement Autonomous Work persistence "
            "materialization."
        )

    slug = resolve_slug(category, profile=profile)
    merged = merge_config(profile.options_for_slug(slug), None)
    entry = get_entry(slug)
    factory = entry.factory
    if isinstance(factory, AutonomousWorkMaterializationBinder):
        materializer = factory.bind_autonomous_work_materialization(merged)
        if not isinstance(materializer, AutonomousWorkPersistenceFactory):
            raise IntegrationConfigurationError(
                "Selected relational store provider "
                f"({slug}) bind_autonomous_work_materialization returned an object "
                "that does not implement AutonomousWorkPersistenceFactory."
            )
        return materializer.materialize_autonomous_work_repositories()

    if isinstance(factory, AutonomousWorkPersistenceFactory):
        return factory.materialize_autonomous_work_repositories()

    raise IntegrationConfigurationError(
        "Selected relational store provider "
        f"({slug}) does not implement Autonomous Work persistence "
        "materialization binder or factory."
    )
