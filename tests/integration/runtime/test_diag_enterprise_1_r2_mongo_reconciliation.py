# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-1-R2 real Mongo list-index reconciliation proof."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.problem_lifecycle import mint_problem_id
from intergrax.runtime.diagnostics.problem_list_query import (
    encode_list_index_data,
    list_index_row_key,
    list_scopes_for_status,
)
from tests.integration.applications.architecture.harden_4f_mongo_support import (
    cleanup_proof_tenant,
    create_proof_document_store,
    ensure_mongo_running,
    require_docker_for_harden_4f_proof,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_problem_persistence_for_tests,
)

_TENANT = "diag-enterprise-1-r2-mongo"
_PARTITION = f"intergrax.diagnostic_problem.v1:{_TENANT}"
_BASE_TIME = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)
_NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
_MINIMUM_PROJECTION_AGE = timedelta(hours=1)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]


@pytest.fixture
def mongo_enterprise_r2_lifecycle(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from tests.integration.applications.architecture.harden_4f_mongo_support import proof_env

    env = proof_env()
    monkeypatch.setenv("INTERGRAX_MONGODB_URI", env["INTERGRAX_MONGODB_URI"])
    monkeypatch.setenv("INTERGRAX_MONGODB_DATABASE", env["INTERGRAX_MONGODB_DATABASE"])
    monkeypatch.setenv("INTERGRAX_MONGODB_COLLECTION", env["INTERGRAX_MONGODB_COLLECTION"])
    require_docker_for_harden_4f_proof()
    ensure_mongo_running()
    cleanup_proof_tenant(tenant_id=_TENANT)
    try:
        yield
    finally:
        cleanup_proof_tenant(tenant_id=_TENANT)
        ensure_mongo_running()


def test_mongo_reconcile_deletes_proven_orphan_and_restores_query(
    mongo_enterprise_r2_lifecycle: None,
) -> None:
    del mongo_enterprise_r2_lifecycle
    store = create_proof_document_store()
    try:
        persistence = document_store_problem_persistence_for_tests(store)
        persistence.set_clock_for_tests(lambda: _NOW)
        valid = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
        persistence.create(valid)

        orphan = replace(
            sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id()),
            last_seen_at=_BASE_TIME + timedelta(days=1),
        )
        for scope in list_scopes_for_status(orphan.status):
            store.put(
                DocumentRecord(
                    partition_key=_PARTITION,
                    row_key=list_index_row_key(scope=scope, problem=orphan),
                    data=encode_list_index_data(
                        problem_id=orphan.problem_id,
                        last_seen_at=orphan.last_seen_at,
                        status=orphan.status,
                        record_version=orphan.record_version,
                        projection_written_at=_NOW - timedelta(hours=2),
                    ),
                ),
            )

        before = persistence.query_problems(tenant_id=_TENANT, limit=10)
        assert len(before.problems) == 1
        assert before.problems[0].problem_id == valid.problem_id

        page = persistence.reconcile_list_indexes(
            tenant_id=_TENANT,
            minimum_projection_age=_MINIMUM_PROJECTION_AGE,
            limit=100,
        )
        assert page.deleted >= len(list_scopes_for_status(orphan.status))

        after = persistence.query_problems(tenant_id=_TENANT, limit=10)
        assert len(after.problems) == 1
        assert after.problems[0].problem_id == valid.problem_id
    finally:
        store.close()
