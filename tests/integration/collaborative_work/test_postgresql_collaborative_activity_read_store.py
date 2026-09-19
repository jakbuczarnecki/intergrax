# © Artur Czarnecki. All rights reserved.

"""PostgreSQL MP-6E read port qualification (live provider when configured)."""

from __future__ import annotations

import pytest

from intergrax.collaborative_work.persistence import (
    collaborative_activity_append_store_from_postgresql_bundle,
    collaborative_activity_read_store_from_postgresql_bundle,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositoriesWithArtifacts
from tests.unit.collaborative_work.collaborative_activity_append_store_contract import (
    fixed_recorded_at,
)
from tests.unit.collaborative_work.collaborative_activity_read_port_contract import (
    run_collaborative_activity_read_port_contract_suite,
    run_late_occurred_at_ordering_contract,
    run_tenant_isolation_read_contract,
    run_workspace_isolation_read_contract,
)

pytestmark = [pytest.mark.integration, pytest.mark.network]


def _append_store(bundle: CollaborativeWorkRepositoriesWithArtifacts):
    return collaborative_activity_append_store_from_postgresql_bundle(
        bundle,
        utc_now=fixed_recorded_at,
    )


def _read_store(bundle: CollaborativeWorkRepositoriesWithArtifacts):
    return collaborative_activity_read_store_from_postgresql_bundle(bundle)


def test_postgresql_collaborative_activity_read_port_contract(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> None:
    bundle = postgresql_collaborative_work_bundle
    run_collaborative_activity_read_port_contract_suite(
        read_port_factory=lambda: _read_store(bundle),
        append_store_factory=lambda: _append_store(bundle),
    )


def test_postgresql_collaborative_activity_read_isolation(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> None:
    bundle = postgresql_collaborative_work_bundle
    run_workspace_isolation_read_contract(
        read_port_factory=lambda: _read_store(bundle),
        append_store_factory=lambda: _append_store(bundle),
    )
    run_tenant_isolation_read_contract(
        read_port_factory=lambda: _read_store(bundle),
        append_store_factory=lambda: _append_store(bundle),
    )


def test_postgresql_collaborative_activity_read_late_occurred_at_ordering(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> None:
    bundle = postgresql_collaborative_work_bundle
    run_late_occurred_at_ordering_contract(
        read_port_factory=lambda: _read_store(bundle),
        append_store_factory=lambda: _append_store(bundle),
    )
