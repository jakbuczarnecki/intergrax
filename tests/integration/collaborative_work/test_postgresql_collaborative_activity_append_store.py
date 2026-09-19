# © Artur Czarnecki. All rights reserved.

"""PostgreSQL MP-6D append store concurrency qualification."""

from __future__ import annotations

import pytest

from intergrax.collaborative_work.persistence import (
    collaborative_activity_append_store_from_postgresql_bundle,
    open_postgresql_collaborative_work_repositories,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositoriesWithArtifacts
from tests.unit.collaborative_work.collaborative_activity_append_store_contract import (
    fixed_recorded_at,
    run_append_store_contract_suite,
    run_concurrent_distinct_contract,
    run_concurrent_duplicate_contract,
)

pytestmark = [pytest.mark.integration, pytest.mark.network]


def _append_store(bundle: CollaborativeWorkRepositoriesWithArtifacts):
    return collaborative_activity_append_store_from_postgresql_bundle(
        bundle,
        utc_now=fixed_recorded_at,
    )


def test_postgresql_collaborative_activity_append_store_contract(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> None:
    bundle = postgresql_collaborative_work_bundle

    def factory():
        return _append_store(bundle)

    run_append_store_contract_suite(factory)


def test_postgresql_collaborative_activity_concurrent_duplicate(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> None:
    bundle_a = postgresql_collaborative_work_bundle
    bundle_b = open_postgresql_collaborative_work_repositories(
        config=bundle_a.store.config,
        schema_name=bundle_a.store.schema_name,
    )
    try:
        run_concurrent_duplicate_contract(
            lambda: _append_store(bundle_a),
            open_second_connection=lambda: _append_store(bundle_b),
        )
    finally:
        bundle_b.close()


def test_postgresql_collaborative_activity_concurrent_distinct(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> None:
    bundle_a = postgresql_collaborative_work_bundle
    bundle_b = open_postgresql_collaborative_work_repositories(
        config=bundle_a.store.config,
        schema_name=bundle_a.store.schema_name,
    )
    try:
        run_concurrent_distinct_contract(
            lambda: _append_store(bundle_a),
            open_second_connection=lambda: _append_store(bundle_b),
        )
    finally:
        bundle_b.close()
