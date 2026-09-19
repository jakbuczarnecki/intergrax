# © Artur Czarnecki. All rights reserved.

"""MP-6G live PostgreSQL E2E qualification."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import cast

import pytest

from intergrax.collaborative_work.persistence import (
    CollaborativeWorkRepositoriesWithArtifacts,
    open_postgresql_collaborative_work_repositories,
)
from intergrax.contracts.collaborative_activity_publisher_authority import (
    restricted_collaborative_activity_workspace_authority,
)
from tests.qualification.mp6.mp6g_e2e_contract import run_mp6g_e2e_contract_suite
from tests.qualification.mp6.mp6g_harness import (
    TENANT_A,
    WS_A,
    build_mp6g_harness_from_postgresql_bundle,
    mp6g_fixed_clock,
    mp6g_fixed_utc_now,
)
from tests.integration.collaborative_work.conftest import SCHEMA_PREFIX, _drop_schema
from tests.qualification.mp6.mp6g_harness import Mp6gHarness
from tests.unit.collaborative_work.mp6c_publisher_authority_test_support import (
    platform_publisher_registration,
)

pytestmark = [pytest.mark.integration, pytest.mark.network]

_RESTRICTED_PUBLISHER = "platform-activity-publisher-restricted"


def _postgresql_store_config(
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
):
    from intergrax.collaborative_work.postgresql_repository import PostgreSQLCollaborativeWorkStore

    return cast(PostgreSQLCollaborativeWorkStore, bundle.store).config


def _isolated_schema_bundle(
    primary_bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> tuple[CollaborativeWorkRepositoriesWithArtifacts, str]:
    schema_name = f"{SCHEMA_PREFIX}{uuid.uuid4().hex}"
    bundle = open_postgresql_collaborative_work_repositories(
        config=_postgresql_store_config(primary_bundle),
        schema_name=schema_name,
    )
    return bundle, schema_name


def _concurrent_pair_harness_factory(
    primary_bundle: CollaborativeWorkRepositoriesWithArtifacts,
    *,
    build: Callable[[CollaborativeWorkRepositoriesWithArtifacts], Mp6gHarness],
    schemas_to_drop: list[str],
) -> Callable[[], Mp6gHarness]:
    shared_schema: str | None = None

    def factory() -> Mp6gHarness:
        nonlocal shared_schema
        if shared_schema is None:
            shared_schema = f"{SCHEMA_PREFIX}{uuid.uuid4().hex}"
            schemas_to_drop.append(shared_schema)
        bundle = open_postgresql_collaborative_work_repositories(
            config=_postgresql_store_config(primary_bundle),
            schema_name=shared_schema,
        )
        return build(bundle)

    return factory


def test_mp6g_postgresql_e2e_contract_suite(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> None:
    primary_bundle = postgresql_collaborative_work_bundle
    schemas_to_drop: list[str] = []

    def _build_standard(bundle: CollaborativeWorkRepositoriesWithArtifacts) -> Mp6gHarness:
        return build_mp6g_harness_from_postgresql_bundle(
            bundle,
            utc_now=mp6g_fixed_utc_now,
            clock=mp6g_fixed_clock,
        )

    def _isolated_harness(
        build: Callable[[CollaborativeWorkRepositoriesWithArtifacts], Mp6gHarness],
    ) -> Mp6gHarness:
        bundle, schema_name = _isolated_schema_bundle(primary_bundle)
        schemas_to_drop.append(schema_name)
        return build(bundle)

    def harness_factory():
        return _isolated_harness(_build_standard)

    def restricted_harness_factory():
        def build(bundle: CollaborativeWorkRepositoriesWithArtifacts) -> Mp6gHarness:
            return build_mp6g_harness_from_postgresql_bundle(
                bundle,
                utc_now=mp6g_fixed_utc_now,
                clock=mp6g_fixed_clock,
                publisher_principal_id=_RESTRICTED_PUBLISHER,
                publisher_registrations=(
                    platform_publisher_registration(
                        TENANT_A,
                        _RESTRICTED_PUBLISHER,
                        workspace_authority=restricted_collaborative_activity_workspace_authority(WS_A),
                    ),
                ),
            )

        return _isolated_harness(build)

    def policy_harness_factory(policy):
        def build(bundle: CollaborativeWorkRepositoriesWithArtifacts) -> Mp6gHarness:
            return build_mp6g_harness_from_postgresql_bundle(
                bundle,
                utc_now=mp6g_fixed_utc_now,
                clock=mp6g_fixed_clock,
                ingestion_policy=policy,
            )

        return _isolated_harness(build)

    concurrent_duplicate_pair_factory = _concurrent_pair_harness_factory(
        primary_bundle,
        build=_build_standard,
        schemas_to_drop=schemas_to_drop,
    )
    concurrent_distinct_pair_factory = _concurrent_pair_harness_factory(
        primary_bundle,
        build=_build_standard,
        schemas_to_drop=schemas_to_drop,
    )

    try:
        run_mp6g_e2e_contract_suite(
            harness_factory,
            restricted_harness_factory=restricted_harness_factory,
            policy_harness_factory=policy_harness_factory,
            concurrent_pair_harness_factory=concurrent_duplicate_pair_factory,
            concurrent_distinct_pair_harness_factory=concurrent_distinct_pair_factory,
        )
    finally:
        for schema_name in schemas_to_drop:
            _drop_schema(schema_name)
