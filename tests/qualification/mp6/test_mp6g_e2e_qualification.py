# © Artur Czarnecki. All rights reserved.

"""MP-6G SQLite E2E / isolation / idempotency qualification."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from intergrax.collaborative_work.persistence import open_sqlite_collaborative_work_repositories
from intergrax.contracts.collaborative_activity_publisher_authority import (
    restricted_collaborative_activity_workspace_authority,
)
from tests.qualification.mp6.mp6g_e2e_contract import run_mp6g_e2e_contract_suite
from tests.qualification.mp6.mp6g_harness import (
    TENANT_A,
    WS_A,
    build_mp6g_harness_from_sqlite_bundle,
    mp6g_fixed_clock,
    mp6g_fixed_utc_now,
)
from tests.unit.collaborative_work.mp6c_publisher_authority_test_support import (
    platform_publisher_registration,
)

pytestmark = pytest.mark.unit

_RESTRICTED_PUBLISHER = "platform-activity-publisher-restricted"


def test_mp6g_sqlite_e2e_contract_suite(tmp_path: Path) -> None:
    concurrent_duplicate_db = str(tmp_path / "mp6g-concurrent-duplicate.sqlite")
    concurrent_distinct_db = str(tmp_path / "mp6g-concurrent-distinct.sqlite")

    def harness_factory():
        db_path = str(tmp_path / f"mp6g-{uuid.uuid4().hex}.sqlite")
        bundle = open_sqlite_collaborative_work_repositories(db_path)
        return build_mp6g_harness_from_sqlite_bundle(
            bundle,
            utc_now=mp6g_fixed_utc_now,
            clock=mp6g_fixed_clock,
        )

    def restricted_harness_factory():
        db_path = str(tmp_path / f"mp6g-restricted-{uuid.uuid4().hex}.sqlite")
        bundle = open_sqlite_collaborative_work_repositories(db_path)
        return build_mp6g_harness_from_sqlite_bundle(
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
        db_path = str(tmp_path / f"mp6g-policy-{uuid.uuid4().hex}.sqlite")
        bundle = open_sqlite_collaborative_work_repositories(db_path)
        return build_mp6g_harness_from_sqlite_bundle(
            bundle,
            utc_now=mp6g_fixed_utc_now,
            clock=mp6g_fixed_clock,
            ingestion_policy=policy,
        )

    def concurrent_duplicate_pair_factory():
        bundle = open_sqlite_collaborative_work_repositories(concurrent_duplicate_db)
        return build_mp6g_harness_from_sqlite_bundle(
            bundle,
            utc_now=mp6g_fixed_utc_now,
            clock=mp6g_fixed_clock,
        )

    def concurrent_distinct_pair_factory():
        bundle = open_sqlite_collaborative_work_repositories(concurrent_distinct_db)
        return build_mp6g_harness_from_sqlite_bundle(
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
