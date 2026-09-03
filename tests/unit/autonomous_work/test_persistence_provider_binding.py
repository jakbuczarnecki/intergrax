# © Artur Czarnecki. All rights reserved.

"""Autonomous Work persistence provider binding tests (AW-2C)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.autonomous_work.in_memory_repository import InMemoryWorkerInstanceRepository
from intergrax.autonomous_work.materialization_factory import AutonomousWorkPersistenceFactory
from intergrax.autonomous_work.persistence import AutonomousWorkRepositories
from intergrax.autonomous_work.persistence_provider import resolve_autonomous_work_repositories
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationEntry,
)
from intergrax.integrations.providers.relational_store.postgresql.bundle import (
    create_postgresql_relational_store,
)
from intergrax.integrations.providers.relational_store.postgresql.register import (
    register_postgresql_integration,
)
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.catalog_manifests import POSTGRESQL
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit


def test_postgresql_factory_binds_autonomous_work_materialization() -> None:
    materializer = create_postgresql_relational_store.bind_autonomous_work_materialization({})
    assert isinstance(materializer, AutonomousWorkPersistenceFactory)


def test_resolve_autonomous_work_repositories_uses_postgresql_binder() -> None:
    register_postgresql_integration()
    profile = IntegrationProfile(
        relational_store=POSTGRESQL,
        options={POSTGRESQL.slug: {"dsn": "postgresql://localhost/test"}},
    )
    bundle = AutonomousWorkRepositories(
        worker_definition=MagicMock(),
        worker_instance=InMemoryWorkerInstanceRepository(),
        responsibility=MagicMock(),
        worker_goal=MagicMock(),
        work_continuity_state=MagicMock(),
        worker_principal_binding=MagicMock(),
        store=MagicMock(),
    )
    with patch(
        "intergrax.integrations.providers.relational_store.postgresql.bundle."
        "_materialize_postgresql_autonomous_work_repositories",
        return_value=bundle,
    ) as materialize:
        resolved = resolve_autonomous_work_repositories(profile)
    materialize.assert_called_once()
    assert resolved is bundle


def test_resolve_autonomous_work_repositories_does_not_call_aw_from_env() -> None:
    profile = IntegrationProfile(
        relational_store=POSTGRESQL,
        options={POSTGRESQL.slug: {"dsn": "postgresql://localhost/test"}},
    )
    with (
        patch(
            "intergrax.integrations.providers.relational_store.postgresql.bundle."
            "_materialize_postgresql_autonomous_work_repositories",
            side_effect=IntegrationConfigurationError("backend unavailable"),
        ),
        patch(
            "intergrax.autonomous_work.persistence_provider.merge_config",
            return_value={"dsn": "postgresql://localhost/test"},
        ),
        pytest.raises(IntegrationConfigurationError, match="backend unavailable"),
    ):
        resolve_autonomous_work_repositories(profile)


def test_resolve_autonomous_work_repositories_rejects_non_materializing_provider() -> None:
    class _UnsupportedFactory:
        def __call__(self, **_: object) -> object:
            return object()

    register_integration(
        IntegrationEntry(
            slug="unsupported-relational",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=_UnsupportedFactory(),
        )
    )
    profile = IntegrationProfile(relational_store="unsupported-relational")
    with pytest.raises(IntegrationConfigurationError, match="does not implement Autonomous Work"):
        resolve_autonomous_work_repositories(profile)
