# © Artur Czarnecki. All rights reserved.

"""MP-6G live PostgreSQL E2E qualification."""

from __future__ import annotations

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
from tests.unit.collaborative_work.mp6c_publisher_authority_test_support import (
    platform_publisher_registration,
)

pytestmark = [pytest.mark.integration, pytest.mark.network]

_RESTRICTED_PUBLISHER = "platform-activity-publisher-restricted"


def _shared_schema_bundle(
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> CollaborativeWorkRepositoriesWithArtifacts:
    from intergrax.collaborative_work.postgresql_repository import PostgreSQLCollaborativeWorkStore

    store = cast(PostgreSQLCollaborativeWorkStore, bundle.store)
    return open_postgresql_collaborative_work_repositories(
        config=store.config,
        schema_name=store.schema_name,
    )


def test_mp6g_postgresql_e2e_contract_suite(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> None:
    primary_bundle = postgresql_collaborative_work_bundle

    def harness_factory():
        bundle = _shared_schema_bundle(primary_bundle)
        return build_mp6g_harness_from_postgresql_bundle(
            bundle,
            utc_now=mp6g_fixed_utc_now,
            clock=mp6g_fixed_clock,
        )

    def restricted_harness_factory():
        bundle = _shared_schema_bundle(primary_bundle)
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

    def policy_harness_factory(policy):
        bundle = _shared_schema_bundle(primary_bundle)
        return build_mp6g_harness_from_postgresql_bundle(
            bundle,
            utc_now=mp6g_fixed_utc_now,
            clock=mp6g_fixed_clock,
            ingestion_policy=policy,
        )

    def concurrent_duplicate_pair_factory():
        bundle = _shared_schema_bundle(primary_bundle)
        return build_mp6g_harness_from_postgresql_bundle(
            bundle,
            utc_now=mp6g_fixed_utc_now,
            clock=mp6g_fixed_clock,
        )

    def concurrent_distinct_pair_factory():
        bundle = _shared_schema_bundle(primary_bundle)
        return build_mp6g_harness_from_postgresql_bundle(
            bundle,
            utc_now=mp6g_fixed_utc_now,
            clock=mp6g_fixed_clock,
        )

    run_mp6g_e2e_contract_suite(
        harness_factory,
        restricted_harness_factory=restricted_harness_factory,
        policy_harness_factory=policy_harness_factory,
        concurrent_pair_harness_factory=concurrent_duplicate_pair_factory,
        concurrent_distinct_pair_harness_factory=concurrent_distinct_pair_factory,
    )
