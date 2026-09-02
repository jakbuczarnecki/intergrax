# © Artur Czarnecki. All rights reserved.

"""Provider fixtures for Collaborative Work enterprise E2E qualification."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from intergrax.collaborative_work.persistence import (
    CollaborativeWorkRepositories,
    open_postgresql_collaborative_work_repositories,
)
from intergrax.collaborative_work.persistence_provider import (
    resolve_collaborative_work_repositories,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.registry.profile import IntegrationProfile
from tests.e2e.collaborative_work.harness.composition import (
    MultiplayerE2EContext,
    build_authorization_boundary,
    open_multiplayer_e2e_context,
)
from tests.e2e.collaborative_work.harness.constants import FIXED_NOW
from tests.e2e.collaborative_work.harness.profile_factory import (
    POSTGRESQL_SCHEMA_PREFIX,
    invalid_postgresql_profile,
    postgresql_integration_profile,
    resolve_postgresql_dsn,
    sqlite_integration_profile,
)
from tests.e2e.collaborative_work.harness.runtime_policy import MutableRuntimePolicyEvaluator
from tests.e2e.collaborative_work.harness.scenario_runner import allow_runtime_decision

pytestmark = pytest.mark.e2e


@pytest.fixture
def sqlite_e2e_profile(tmp_path: Path) -> IntegrationProfile:
    return sqlite_integration_profile(tmp_path / "collaborative-work-e2e")


@pytest.fixture
def sqlite_e2e_context(sqlite_e2e_profile: IntegrationProfile) -> Generator[MultiplayerE2EContext, None, None]:
    runtime = MutableRuntimePolicyEvaluator(allow_runtime_decision())
    context = open_multiplayer_e2e_context(
        sqlite_e2e_profile,
        runtime,
        clock=lambda: FIXED_NOW,
    )
    try:
        yield context
    finally:
        context.bundle.close()


@pytest.fixture
def postgresql_e2e_profile() -> IntegrationProfile:
    dsn = resolve_postgresql_dsn()
    if not dsn:
        pytest.skip(
            "PostgreSQL E2E requires INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN "
            "or INTERGRAX_POSTGRESQL_* settings"
        )
    schema_name = f"{POSTGRESQL_SCHEMA_PREFIX}{uuid.uuid4().hex}"
    return postgresql_integration_profile(dsn=dsn, schema_name=schema_name)


@pytest.fixture
def postgresql_e2e_bundle(
    postgresql_e2e_profile: IntegrationProfile,
) -> Generator[CollaborativeWorkRepositories, None, None]:
    try:
        bundle = resolve_collaborative_work_repositories(postgresql_e2e_profile)
    except (IntegrationConfigurationError, ConnectionError, TimeoutError, OSError) as exc:
        pytest.skip(f"PostgreSQL backend unavailable: {type(exc).__name__}: {exc}")
    try:
        yield bundle
    finally:
        schema_name = bundle.store.schema_name
        config = bundle.store.config
        bundle.close()
        cleanup = open_postgresql_collaborative_work_repositories(
            config=config,
            schema_name="public",
        )
        try:
            with cleanup.store.transaction() as conn:
                conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
        finally:
            cleanup.close()


@pytest.fixture
def postgresql_e2e_context(
    postgresql_e2e_profile: IntegrationProfile,
    postgresql_e2e_bundle: CollaborativeWorkRepositories,
) -> Generator[MultiplayerE2EContext, None, None]:
    runtime = MutableRuntimePolicyEvaluator(allow_runtime_decision())
    context = MultiplayerE2EContext(
        profile=postgresql_e2e_profile,
        bundle=postgresql_e2e_bundle,
        boundary=build_authorization_boundary(
            postgresql_e2e_bundle,
            runtime,
            clock=lambda: FIXED_NOW,
        ),
        runtime_policy=runtime,
    )
    yield context


@pytest.fixture
def invalid_postgresql_profile_fixture() -> IntegrationProfile:
    return invalid_postgresql_profile()
